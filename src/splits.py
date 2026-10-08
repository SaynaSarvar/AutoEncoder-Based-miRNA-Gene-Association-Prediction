"""Step 5: cross-validation folds shared across negative-sampling strategies.

Run from the project root:  python3 -m src.splits

Split types (5 folds, seed from config)
  random_kfold    positives: one seeded permutation shared by all strategies; each strategy's
                  negatives get their own seeded permutation (folds stay ~50/50).
  random_holdout  fold 0 of random_kfold (80/20); a flag in load_split, no extra column.
  cold_mirna      whole miRNA seq_groups per fold (identical sequences never straddle
                  train/test); greedy balancing of positive rows, largest groups first.
  cold_gene       whole genes per fold, same balancing.
Validation (per test fold k): 10% of the remaining training data, drawn the same way
(random rows / whole seq_groups / whole genes), seeded by k. Positives' validation membership
is shared across strategies too.

Outputs
  data/processed/folds_<strategy>.parquet   row-aligned with pairs_<strategy>.parquet:
      fold_random, fold_cold_mirna, fold_cold_gene (int8, 0-4),
      val_random_k, val_cold_mirna_k, val_cold_gene_k (bool, k = 0..4)
  data/processed/folds_groups.json           group -> fold maps and validation groups
  results/step5_report.txt
"""
import json
import zlib

import numpy as np
import pandas as pd

from src.parse import Report
from src.utils import PROJECT_ROOT, load_config

STRATEGIES = ["paper", "random", "degree_aware"]
SPLITS = {"random_kfold": "random", "random_holdout": "random",
          "cold_mirna": "cold_mirna", "cold_gene": "cold_gene"}
COLD = {"cold_mirna": "mirna", "cold_gene": "gene"}
N_FOLDS = 5
VAL_FRAC = 0.10


def _paths():
    cfg = load_config()
    pr = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    return cfg, {
        "pairs": lambda s: pr / f"pairs_{s}.parquet",
        "folds": lambda s: pr / f"folds_{s}.parquet",
        "groups": pr / "folds_groups.json",
        "mirna_index": pr / "mirna_index.csv", "gene_index": pr / "gene_index.csv",
        "report": PROJECT_ROOT / cfg["paths"]["results_dir"] / "step5_report.txt",
    }


def _rng(seed, *keys):
    """Deterministic generator derived from the global seed and string/int keys."""
    return np.random.default_rng([seed] + [zlib.crc32(str(k).encode()) for k in keys])


# --------------------------------------------------------------------------- loaders
def load_split(strategy: str, split_type: str, fold: int):
    """Row indices (train_idx, val_idx, test_idx) into pairs_<strategy>.parquet.
    split_type: random_kfold | random_holdout (always fold 0) | cold_mirna | cold_gene."""
    _, P = _paths()
    if split_type == "random_holdout":
        fold = 0
    col = SPLITS[split_type]
    f = pd.read_parquet(P["folds"](strategy), columns=[f"fold_{col}", f"val_{col}_{fold}"])
    test = f[f"fold_{col}"].to_numpy() == fold
    val = f[f"val_{col}_{fold}"].to_numpy() & ~test
    train = ~test & ~val
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)


def load_cross(train_strategy: str, test_strategy: str, split_type: str, fold: int):
    """Train/val rows from train_strategy; test = the held-out positives (shared) plus
    test_strategy's negatives in that fold.

    Returns (train_idx, val_idx, test_idx): train_idx/val_idx index pairs_<train_strategy>,
    test_idx indexes pairs_<test_strategy>. Test negatives that also occur as rows in
    train_strategy's train/val (possible only for random splits) are dropped."""
    _, P = _paths()
    train_idx, val_idx, _ = load_split(train_strategy, split_type, fold)
    _, _, test_idx = load_split(test_strategy, split_type, fold)
    if train_strategy == test_strategy:
        return train_idx, val_idx, test_idx
    tr = pd.read_parquet(P["pairs"](train_strategy), columns=["mirna_idx", "gene_idx"])
    te = pd.read_parquet(P["pairs"](test_strategy), columns=["mirna_idx", "gene_idx"])
    n_g = int(max(tr["gene_idx"].max(), te["gene_idx"].max())) + 1
    seen = (tr["mirna_idx"].to_numpy(np.int64) * n_g + tr["gene_idx"].to_numpy())[
        np.concatenate([train_idx, val_idx])]
    test_keys = te["mirna_idx"].to_numpy(np.int64)[test_idx] * n_g + te["gene_idx"].to_numpy()[test_idx]
    return train_idx, val_idx, test_idx[~np.isin(test_keys, seen)]


# --------------------------------------------------------------------------- fold assignment
def greedy_groups(group_counts: pd.Series, n_folds: int) -> dict:
    """Largest groups first, each into the fold with the fewest positive rows so far."""
    order = group_counts.sort_values(ascending=False, kind="stable")
    load = np.zeros(n_folds, dtype=np.int64)
    out = {}
    for g, c in order.items():
        k = int(np.argmin(load))
        out[g] = k
        load[k] += c
    return out


def val_groups(train_counts: pd.Series, rng) -> set:
    """Whole groups in seeded random order until >= VAL_FRAC of the training positives."""
    groups = train_counts.index.to_numpy().copy()  # to_numpy() can be a view of the index
    rng.shuffle(groups)
    cum = np.cumsum(train_counts.loc[groups].to_numpy())
    n = int(np.searchsorted(cum, VAL_FRAC * cum[-1])) + 1
    return set(groups[:n].tolist())


def build(seed, pairs, mi_index, report):
    n_g = int(max(p["gene_idx"].max() for p in pairs.values())) + 1
    for p in pairs.values():
        p["key"] = p["mirna_idx"].to_numpy(np.int64) * n_g + p["gene_idx"].to_numpy()

    # ---- identical positives everywhere
    pos_keys = np.sort(pairs[STRATEGIES[0]].loc[lambda d: d["label"] == 1, "key"].to_numpy())
    for s in STRATEGIES[1:]:
        other = np.sort(pairs[s].loc[pairs[s]["label"] == 1, "key"].to_numpy())
        assert np.array_equal(pos_keys, other), f"positives differ between paper and {s}"
    report(f"positive pairs identical in all strategies: {len(pos_keys):,}")
    pos_df = pairs[STRATEGIES[0]].loc[pairs[STRATEGIES[0]]["label"] == 1,
                                      ["key", "mirna_idx", "gene_idx"]]
    pos_df = pos_df.set_index("key").loc[pos_keys]

    # ---- random folds for positives (shared)
    perm = _rng(seed, "random_kfold", "positives").permutation(len(pos_keys))
    pos_fold = np.empty(len(pos_keys), np.int8)
    pos_fold[perm] = np.arange(len(pos_keys)) % N_FOLDS
    pos_fold = pd.Series(pos_fold, index=pos_keys)
    pos_val = {}
    for k in range(N_FOLDS):
        cand = pos_keys[pos_fold.to_numpy() != k]
        r = _rng(seed, "val_random", k, "positives")
        pos_val[k] = set(r.choice(cand, size=int(round(VAL_FRAC * len(cand))), replace=False).tolist())

    # ---- cold folds (entity -> fold, from positive counts only)
    seq_group = mi_index.set_index("mirna_idx")["seq_group"]
    group_maps, val_maps = {}, {}
    ent = {"cold_mirna": pos_df["mirna_idx"].map(seq_group), "cold_gene": pos_df["gene_idx"]}
    for split, groups in ent.items():
        counts = groups.value_counts()
        gmap = greedy_groups(counts, N_FOLDS)
        group_maps[split] = gmap
        val_maps[split] = {}
        for k in range(N_FOLDS):
            train_groups = counts[[gmap[g] != k for g in counts.index]]
            val_maps[split][k] = val_groups(train_groups, _rng(seed, f"val_{split}", k))

    # ---- per strategy
    folds = {}
    for s in STRATEGIES:
        d = pairs[s]
        is_pos = d["label"].to_numpy() == 1
        f = pd.DataFrame(index=d.index)
        fr = np.empty(len(d), np.int8)
        fr[is_pos] = pos_fold.loc[d.loc[is_pos, "key"]].to_numpy()
        neg_rows = np.flatnonzero(~is_pos)
        nperm = _rng(seed, "random_kfold", "negatives", s).permutation(len(neg_rows))
        fr[neg_rows[nperm]] = np.arange(len(neg_rows)) % N_FOLDS
        f["fold_random"] = fr
        sg = d["mirna_idx"].map(seq_group)
        f["fold_cold_mirna"] = sg.map(group_maps["cold_mirna"]).astype(np.int8)
        f["fold_cold_gene"] = d["gene_idx"].map(group_maps["cold_gene"]).astype(np.int8)
        for k in range(N_FOLDS):
            v = np.zeros(len(d), bool)
            v[is_pos] = d.loc[is_pos, "key"].isin(pos_val[k]).to_numpy()
            cand = neg_rows[fr[neg_rows] != k]
            r = _rng(seed, "val_random", k, "negatives", s)
            v[r.choice(cand, size=int(round(VAL_FRAC * len(cand))), replace=False)] = True
            f[f"val_random_{k}"] = v & (fr != k)
            f[f"val_cold_mirna_{k}"] = sg.isin(val_maps["cold_mirna"][k]).to_numpy()
            f[f"val_cold_gene_{k}"] = d["gene_idx"].isin(val_maps["cold_gene"][k]).to_numpy()
        folds[s] = f
    meta = {
        "seed": seed, "n_folds": N_FOLDS, "val_frac": VAL_FRAC,
        "cold_mirna_seq_group_to_fold": {int(g): k for g, k in group_maps["cold_mirna"].items()},
        "cold_gene_gene_idx_to_fold": {int(g): k for g, k in group_maps["cold_gene"].items()},
        "val_groups": {split: {str(k): sorted(int(x) for x in v) for k, v in m.items()}
                       for split, m in val_maps.items()},
    }
    return folds, meta, pos_df, pos_fold, seq_group


# --------------------------------------------------------------------------- checks
def checks(pairs, folds, pos_df, seq_group, report):
    flagged = []
    report("=" * 70)
    report("Per-fold test sets: rows, % positive, distinct miRNAs / genes")
    report("=" * 70)
    for split, col in [("random_kfold", "random"), ("cold_mirna", "cold_mirna"),
                       ("cold_gene", "cold_gene")]:
        report(f"--- {split}")
        report(f"  {'strategy':<14}" + "".join(f"{'fold ' + str(k):>26}" for k in range(N_FOLDS)))
        for s in STRATEGIES:
            d, f = pairs[s], folds[s]
            cells = []
            for k in range(N_FOLDS):
                t = f[f"fold_{col}"].to_numpy() == k
                frac = d["label"].to_numpy()[t].mean()
                flag = "!" if frac < 0.40 or frac > 0.60 else " "
                if flag == "!":
                    flagged.append((s, split, k, frac))
                cells.append(f"{t.sum():>10,} {frac:6.1%}{flag}")
            report(f"  {s:<14}" + "".join(f"{c:>26}" for c in cells))
        for s in STRATEGIES:
            d, f = pairs[s], folds[s]
            cells = []
            for k in range(N_FOLDS):
                t = f[f"fold_{col}"].to_numpy() == k
                cells.append(f"{d['mirna_idx'][t].nunique():,} mi / {d['gene_idx'][t].nunique():,} g")
            report(f"  {s:<14}" + "".join(f"{c:>26}" for c in cells))
        # validation share and balance (fold 0 shown, all folds checked)
        for s in STRATEGIES:
            d, f = pairs[s], folds[s]
            parts = []
            for k in range(N_FOLDS):
                test = f[f"fold_{col}"].to_numpy() == k
                val = f[f"val_{col}_{k}"].to_numpy() & ~test
                assert not (f[f"val_{col}_{k}"].to_numpy() & test).any(), "val overlaps test"
                share = val.sum() / (~test).sum()
                assert 0.05 < share < 0.15, f"{s} {split} fold {k}: val share {share:.1%}"
                parts.append(f"{share:.1%}/{d['label'].to_numpy()[val].mean():.0%}")
            report(f"  val share/pos% {s:<14}" + "  ".join(parts))
    report("  ('!' = fold outside 40-60% positive; val share = val rows / non-test rows)")
    report(f"flagged folds: {len(flagged)}" + "".join(
        f"\n  {s} {sp} fold {k}: {fr:.1%} positive" for s, sp, k, fr in flagged))
    report()

    report("=" * 70)
    report("Leakage and consistency checks")
    report("=" * 70)
    for s in STRATEGIES:
        d, f = pairs[s], folds[s]
        sg = d["mirna_idx"].map(seq_group).to_numpy()
        for k in range(N_FOLDS):
            t = f["fold_cold_mirna"].to_numpy() == k
            assert not set(sg[t]) & set(sg[~t]), f"{s} cold_mirna fold {k}: seq_group overlap"
            t = f["fold_cold_gene"].to_numpy() == k
            g = d["gene_idx"].to_numpy()
            assert not set(g[t]) & set(g[~t]), f"{s} cold_gene fold {k}: gene overlap"
            for split, ent in [("cold_mirna", sg), ("cold_gene", g)]:
                test = f[f"fold_{split}"].to_numpy() == k
                val = f[f"val_{split}_{k}"].to_numpy() & ~test
                train = ~test & ~val
                assert not set(ent[val]) & set(ent[train]), f"{s} {split} fold {k}: val/train overlap"
    report("cold_mirna: zero seq_group overlap between train+val and test (all strategies, folds)")
    report("cold_gene:  zero gene overlap between train+val and test (all strategies, folds)")
    report("cold val groups are disjoint from train groups")

    for col in ["fold_random", "fold_cold_mirna", "fold_cold_gene"]:
        ref = None
        for s in STRATEGIES:
            d, f = pairs[s], folds[s]
            m = d["label"].to_numpy() == 1
            ser = pd.Series(f[col].to_numpy()[m], index=d["key"].to_numpy()[m]).sort_index()
            if ref is None:
                ref = ser
            else:
                assert ser.equals(ref), f"positives' {col} differ for {s}"
        for k in range(N_FOLDS):
            vcol = col.replace("fold_", "val_") + f"_{k}"
            vref = None
            for s in STRATEGIES:
                d, f = pairs[s], folds[s]
                m = d["label"].to_numpy() == 1
                ser = pd.Series(f[vcol].to_numpy()[m], index=d["key"].to_numpy()[m]).sort_index()
                if vref is None:
                    vref = ser
                else:
                    assert ser.equals(vref), f"positives' {vcol} differ for {s}"
    report("positives: identical test fold AND validation membership across the 3 strategies, "
           "for every split type")
    report()

    report("=" * 70)
    report("Hub concentration in cold test folds (positives involving top-1% degree entities)")
    report("=" * 70)
    deg_mi = pos_df["mirna_idx"].value_counts()
    deg_g = pos_df["gene_idx"].value_counts()
    hub_mi = set(deg_mi.index[:max(1, int(np.ceil(0.01 * len(deg_mi))))])
    hub_g = set(deg_g.index[:max(1, int(np.ceil(0.01 * len(deg_g))))])
    report(f"hub miRNAs: {len(hub_mi)} (degree >= {deg_mi.iloc[len(hub_mi) - 1]}); "
           f"hub genes: {len(hub_g)} (degree >= {deg_g.iloc[len(hub_g) - 1]})")
    f0, d0 = folds[STRATEGIES[0]], pairs[STRATEGIES[0]]
    pm = d0["label"].to_numpy() == 1
    overall_mi = d0.loc[pm, "mirna_idx"].isin(hub_mi).mean()
    overall_g = d0.loc[pm, "gene_idx"].isin(hub_g).mean()
    report(f"all positives: {overall_mi:.1%} involve a hub miRNA, {overall_g:.1%} a hub gene")
    for split in ["cold_mirna", "cold_gene"]:
        cells = []
        for k in range(N_FOLDS):
            t = (f0[f"fold_{split}"].to_numpy() == k) & pm
            hm = d0.loc[t, "mirna_idx"].isin(hub_mi)
            hg = d0.loc[t, "gene_idx"].isin(hub_g)
            nh = len(set(d0.loc[t, "mirna_idx"]) & hub_mi) if split == "cold_mirna" else \
                len(set(d0.loc[t, "gene_idx"]) & hub_g)
            cells.append(f"fold {k}: {t.sum():,} pos, hubs={nh}, "
                         f"hub-miRNA {hm.mean():.1%}, hub-gene {hg.mean():.1%}")
        report(f"--- {split}")
        for c in cells:
            report("  " + c)


def main():
    cfg, P = _paths()
    seed = cfg["seed"]
    report = Report()
    report("=" * 70)
    report(f"Step 5: folds ({N_FOLDS}-fold, seed {seed}, validation {VAL_FRAC:.0%})")
    report("=" * 70)
    pairs = {s: pd.read_parquet(P["pairs"](s)) for s in STRATEGIES}
    mi_index = pd.read_csv(P["mirna_index"])
    folds, meta, pos_df, pos_fold, seq_group = build(seed, pairs, mi_index, report)
    n_groups = {s: len(meta[k]) for s, k in [("cold_mirna", "cold_mirna_seq_group_to_fold"),
                                            ("cold_gene", "cold_gene_gene_idx_to_fold")]}
    report(f"cold_mirna: {n_groups['cold_mirna']} seq_groups (from {mi_index.shape[0]} miRNAs); "
           f"cold_gene: {n_groups['cold_gene']} genes")
    report()
    checks(pairs, folds, pos_df, seq_group, report)

    for s in STRATEGIES:
        folds[s].to_parquet(P["folds"](s), index=False)
    P["groups"].write_text(json.dumps(meta))
    report()
    report("saved: " + ", ".join(str(P["folds"](s).relative_to(PROJECT_ROOT)) for s in STRATEGIES)
           + f", {P['groups'].relative_to(PROJECT_ROOT)}")

    # loader smoke test
    tr, va, te = load_split("paper", "cold_gene", 0)
    assert len(tr) + len(va) + len(te) == len(pairs["paper"])
    tr2, va2, te2 = load_cross("paper", "degree_aware", "random_kfold", 0)
    _, _, te_da = load_split("degree_aware", "random_kfold", 0)
    report(f"load_split(paper, cold_gene, 0): train={len(tr):,} val={len(va):,} test={len(te):,}")
    report(f"load_cross(paper -> degree_aware, random_kfold, 0): train={len(tr2):,} "
           f"val={len(va2):,} test={len(te2):,} ({len(te_da) - len(te2):,} test negatives "
           f"dropped because they are train/val rows of paper)")
    report.save(P["report"])


if __name__ == "__main__":
    main()
