"""Step 4: negative sampling strategies + shortcut diagnostics.

Run from the project root:
  python3 -m src.negatives --strategy {paper,random,degree_aware,all}

Candidate pool = full (mirna_idx x gene_idx) index grid minus ALL known_pairs (every source).
The grid is never materialised: it is scanned in chunks of miRNAs, vectorised over all genes.

Strategies
  paper         AEmiGAP 4.3: candidate qualifies if its Euclidean, cosine and Mahalanobis
                distances are ALL above the mean of that metric over the positive pairs.
                Mahalanobis uses S = LedoitWolf covariance of all miRNA + gene vectors pooled
                (the paper does not define S). Sampled uniformly by two-pass rank selection.
  random        uniform over the candidate pool (two-pass rank selection).
  degree_aware  per miRNA as many negatives as positives; genes drawn without replacement with
                probability proportional to their positive degree (Gumbel top-k), excluding
                known pairs.

Outputs
  data/processed/pairs_<strategy>.parquet  mirna_idx, gene_idx, label, d_euc, d_cos, d_mah
  results/step4_report.txt, results/step4_distances.png
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.parse import Report
from src.utils import PROJECT_ROOT, load_config

STRATEGIES = ["paper", "random", "degree_aware"]
METRICS = ["d_euc", "d_cos", "d_mah"]
POS_COLOR, NEG_COLOR = "#2a78d6", "#eb6834"
# numpy 2.x + macOS Accelerate raises spurious divide/overflow flags in matmul (verified
# identical to einsum in step 3); silence only those.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def paths(cfg):
    pr = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    e = PROJECT_ROOT / cfg["paths"]["embeddings_dir"]
    r = PROJECT_ROOT / cfg["paths"]["results_dir"]
    n = cfg["negatives"]
    return {
        "positives": pr / "positives.csv", "known": pr / "known_pairs.csv",
        "mirna_index": pr / "mirna_index.csv", "gene_index": pr / "gene_index.csv",
        "genes_csv": PROJECT_ROOT / cfg["paths"]["interim_dir"] / "genes.csv",
        "mirna_emb": e / n["mirna_embedding"], "gene_emb": e / n["gene_embedding"],
        "out": lambda s: pr / f"pairs_{s}.parquet",
        "report": r / "step4_report.txt", "png": r / "step4_distances.png",
    }


# --------------------------------------------------------------------------- distances
class Distances:
    """Euclidean, cosine and Mahalanobis distances between miRNA and gene vectors."""

    def __init__(self, M, G):
        self.M, self.G = M.astype(np.float64), G.astype(np.float64)
        lw = LedoitWolf().fit(np.vstack([self.M, self.G]))
        self.W = lw.precision_                      # S^-1, computed once
        self.shrinkage = lw.shrinkage_
        self.mm = (self.M ** 2).sum(1)
        self.gg = (self.G ** 2).sum(1)
        self.Mn = self.M / np.linalg.norm(self.M, axis=1, keepdims=True)
        self.Gn = self.G / np.linalg.norm(self.G, axis=1, keepdims=True)
        self.MW = self.M @ self.W
        self.mWm = (self.MW * self.M).sum(1)
        self.gWg = ((self.G @ self.W) * self.G).sum(1)

    def grid(self, mi: np.ndarray):
        """All three distances for miRNAs `mi` against every gene: arrays (len(mi), n_genes)."""
        mg = self.M[mi] @ self.G.T
        euc = np.sqrt(np.maximum(self.mm[mi, None] + self.gg[None, :] - 2 * mg, 0))
        cos = 1.0 - self.Mn[mi] @ self.Gn.T
        mah = np.sqrt(np.maximum(self.mWm[mi, None] + self.gWg[None, :]
                                 - 2 * self.MW[mi] @ self.G.T, 0))
        return euc, cos, mah

    def pairs(self, mi: np.ndarray, gi: np.ndarray, batch: int = 500_000):
        """Distances for explicit (mi, gi) pairs, same formulas as grid()."""
        out = np.empty((len(mi), 3))
        for s in range(0, len(mi), batch):
            a, b = mi[s:s + batch], gi[s:s + batch]
            mg = (self.M[a] * self.G[b]).sum(1)
            out[s:s + batch, 0] = np.sqrt(np.maximum(self.mm[a] + self.gg[b] - 2 * mg, 0))
            out[s:s + batch, 1] = 1.0 - (self.Mn[a] * self.Gn[b]).sum(1)
            out[s:s + batch, 2] = np.sqrt(np.maximum(
                self.mWm[a] + self.gWg[b] - 2 * (self.MW[a] * self.G[b]).sum(1), 0))
        return out


def known_mask_chunk(known_m, known_g, start, stop, n_genes):
    """Boolean (stop-start, n_genes) mask of known pairs; known_* sorted by mirna_idx."""
    lo, hi = np.searchsorted(known_m, [start, stop])
    mask = np.zeros((stop - start, n_genes), dtype=bool)
    mask[known_m[lo:hi] - start, known_g[lo:hi]] = True
    return mask


# --------------------------------------------------------------------------- grid scan
def scan_grid(D, known, thresholds, n_mi, n_g, chunk, select=None):
    """Pass 1 (select=None): count candidates and qualifying pairs.
    Pass 2: `select` = {name: sorted ranks}; returns the selected (mirna_idx, gene_idx) pairs,
    where the rank of a pair is its position in row-major order within that pool."""
    km, kg = known
    counts = {k: 0 for k in ["grid", "candidates", "euc", "cos", "mah",
                             "euc&cos", "euc&mah", "cos&mah", "paper"]}
    offsets = {"random": 0, "paper": 0}
    picked = {k: [] for k in (select or {})}
    for start in range(0, n_mi, chunk):
        stop = min(start + chunk, n_mi)
        mi = np.arange(start, stop)
        euc, cos, mah = D.grid(mi)
        cand = ~known_mask_chunk(km, kg, start, stop, n_g)
        e, c, m = (euc > thresholds["d_euc"]), (cos > thresholds["d_cos"]), (mah > thresholds["d_mah"])
        e &= cand
        c &= cand
        m &= cand
        paper = e & c & m
        pools = {"random": cand, "paper": paper}
        if select is None:
            counts["grid"] += cand.size
            counts["candidates"] += int(cand.sum())
            counts["euc"] += int(e.sum())
            counts["cos"] += int(c.sum())
            counts["mah"] += int(m.sum())
            counts["euc&cos"] += int((e & c).sum())
            counts["euc&mah"] += int((e & m).sum())
            counts["cos&mah"] += int((c & m).sum())
            counts["paper"] += int(paper.sum())
            continue
        for name, ranks in select.items():
            flat = np.flatnonzero(pools[name])
            lo, hi = np.searchsorted(ranks, [offsets[name], offsets[name] + len(flat)])
            take = flat[ranks[lo:hi] - offsets[name]]
            picked[name].append(np.column_stack([take // n_g + start, take % n_g]))
            offsets[name] += len(flat)
    if select is None:
        return counts
    return {k: np.vstack(v) if v else np.empty((0, 2), int) for k, v in picked.items()}


def degree_aware(pos, known, n_mi, n_g, rng, rep):
    """Per miRNA: as many negatives as positives, genes ~ positive degree, no known pairs."""
    km, kg = known
    deg = np.bincount(pos["gene_idx"], minlength=n_g).astype(float)
    logw = np.log(np.where(deg > 0, deg, np.nan))
    quota = np.bincount(pos["mirna_idx"], minlength=n_mi)
    out, short = [], []
    for m in np.flatnonzero(quota):
        lo, hi = np.searchsorted(km, [m, m + 1])
        eligible = np.ones(n_g, bool)
        eligible[kg[lo:hi]] = False
        eligible &= deg > 0
        idx = np.flatnonzero(eligible)
        k = min(quota[m], len(idx))
        if k < quota[m]:
            short.append((m, quota[m], k))
        # Gumbel top-k = weighted sampling without replacement (successive draws ∝ degree).
        keys = logw[idx] + rng.gumbel(size=len(idx))
        top = idx[np.argpartition(-keys, k - 1)[:k]] if k else idx[:0]
        out.append(np.column_stack([np.full(k, m), top]))
    rep(f"  degree_aware: {len(short)} miRNAs could not fill their quota"
        + "".join(f"\n    mirna_idx {m}: needed {q}, got {k}" for m, q, k in short[:20]))
    return np.vstack(out)


# --------------------------------------------------------------------------- diagnostics
def class_pure(df, col):
    pos = set(df.loc[df["label"] == 1, col])
    neg = set(df.loc[df["label"] == 0, col])
    only_pos, only_neg = pos - neg, neg - pos
    rows = df[col].isin(only_pos | only_neg)
    return len(only_pos), len(only_neg), rows


def logreg_auc(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def degree_auc(df, seed):
    """miRNA and gene positive degree computed on the train split only."""
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["label"],
                              random_state=seed)
    train = df.iloc[tr]
    tp = train[train["label"] == 1]
    md = tp["mirna_idx"].value_counts()
    gd = tp["gene_idx"].value_counts()

    def feats(part):
        return np.column_stack([np.log1p(part["mirna_idx"].map(md).fillna(0).to_numpy()),
                                np.log1p(part["gene_idx"].map(gd).fillna(0).to_numpy())])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(feats(train), train["label"])
    test = df.iloc[te]
    return roc_auc_score(test["label"], clf.predict_proba(feats(test))[:, 1])


def diagnose(name, df, ctx, rep, seed):
    s = {}
    pos, neg = df[df["label"] == 1], df[df["label"] == 0]
    rep(f"--- {name}: {len(pos):,} positives + {len(neg):,} negatives")
    rep("  1. distances (pos vs neg):")
    for mcol in METRICS:
        ks = ks_2samp(pos[mcol], neg[mcol]).statistic
        s[f"KS {mcol}"] = ks
        rep(f"     {mcol}: mean {pos[mcol].mean():.3f} vs {neg[mcol].mean():.3f}; "
            f"median {pos[mcol].median():.3f} vs {neg[mcol].median():.3f}; KS={ks:.3f}")

    loglen = ctx["log_len"][df["gene_idx"].to_numpy()]
    gnorm = ctx["g_norm"][df["gene_idx"].to_numpy()]
    mnorm = ctx["m_norm"][df["mirna_idx"].to_numpy()]
    y = df["label"].to_numpy()
    rep("  2. gene bias (pos vs neg):")
    for lab, v in [("log10 gene length", loglen), ("gene vector norm", gnorm)]:
        ks = ks_2samp(v[y == 1], v[y == 0]).statistic
        s[f"KS {lab}"] = ks
        rep(f"     {lab}: median {np.median(v[y == 1]):.3f} vs {np.median(v[y == 0]):.3f}; KS={ks:.3f}")

    rep("  3. degree structure (Spearman of #pos vs #neg per entity):")
    for col, n in [("gene_idx", ctx["n_g"]), ("mirna_idx", ctx["n_mi"])]:
        p = np.bincount(pos[col], minlength=n)
        q = np.bincount(neg[col], minlength=n)
        rho = spearmanr(p, q).statistic
        s[f"rho deg {col[:-4]}"] = rho
        rep(f"     {col[:-4]}: rho={rho:+.3f}")

    rep("  4. class-pure entities:")
    any_rows = np.zeros(len(df), bool)
    for col in ["gene_idx", "mirna_idx"]:
        op, on, rows = class_pure(df, col)
        any_rows |= rows.to_numpy()
        s[f"pure {col[:-4]}s"] = op + on
        rep(f"     {col[:-4]}: only-positive={op}, only-negative={on}; "
            f"rows involving them={rows.mean():.2%}")
    s["rows w/ pure entity"] = any_rows.mean()
    rep(f"     rows involving any class-pure entity: {any_rows.mean():.2%}")

    rep("  5. shortcut baselines (logistic regression, stratified 80/20, test ROC-AUC):")
    feats = {
        "a) 3 distances": df[METRICS].to_numpy(),
        "b) log length": loglen[:, None],
        "c) len + norms": np.column_stack([loglen, gnorm, mnorm]),
        "e) embeddings (128)": np.hstack([ctx["M"][df["mirna_idx"].to_numpy()],
                                          ctx["G"][df["gene_idx"].to_numpy()]]),
    }
    for lab, X in feats.items():
        t0 = time.time()
        s[f"AUC {lab}"] = auc = logreg_auc(X, y, seed)
        rep(f"     {lab:<22} AUC={auc:.3f}  ({time.time() - t0:.0f}s)")
    s["AUC d) degree (train)"] = auc = degree_auc(df, seed)
    rep(f"     {'d) degree (train only)':<22} AUC={auc:.3f}")
    rep()
    return s


def plot(dfs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"d_euc": "Euclidean", "d_cos": "cosine distance", "d_mah": "Mahalanobis"}
    fig, axes = plt.subplots(len(dfs), 3, figsize=(13, 3.1 * len(dfs)), squeeze=False)
    for r, (name, df) in enumerate(dfs.items()):
        for c, mcol in enumerate(METRICS):
            ax = axes[r][c]
            v = df[mcol]
            bins = np.linspace(v.quantile(0.001), v.quantile(0.999), 80)
            for lab, color, txt in [(1, POS_COLOR, "positives"), (0, NEG_COLOR, "negatives")]:
                ax.hist(df.loc[df["label"] == lab, mcol], bins=bins, density=True,
                        histtype="step", linewidth=2, color=color, label=txt)
            ax.set_title(f"{name} · {labels[mcol]}", fontsize=10, loc="left")
            ax.tick_params(colors="#52514e", labelsize=8)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#bbb")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Distance distributions: positives vs sampled negatives (density)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    args = ap.parse_args()
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    cfg = load_config()
    ncfg = cfg["negatives"]
    seed = cfg["seed"]
    P = paths(cfg)
    rng = np.random.default_rng(seed)
    rep = Report()

    mi_index = pd.read_csv(P["mirna_index"])
    g_index = pd.read_csv(P["gene_index"])
    n_mi, n_g = len(mi_index), len(g_index)
    M, G = np.load(P["mirna_emb"]), np.load(P["gene_emb"])
    assert M.shape[0] == n_mi and G.shape[0] == n_g

    pos = pd.read_csv(P["positives"], usecols=["mirna_idx", "gene_idx"])
    n_pos_all = len(pos)
    if ncfg.get("n_pos_subsample"):
        pos = pos.sample(n=ncfg["n_pos_subsample"], random_state=seed).reset_index(drop=True)
    n_neg = len(pos)
    known = pd.read_csv(P["known"], usecols=["mirna_idx", "gene_idx"]).sort_values(
        ["mirna_idx", "gene_idx"])
    known = (known["mirna_idx"].to_numpy(), known["gene_idx"].to_numpy())

    genes = pd.read_csv(P["genes_csv"]).set_index("gene_id")
    ctx = {"M": M, "G": G, "n_mi": n_mi, "n_g": n_g,
           "log_len": np.log10(g_index["gene_id"].map(genes["genomic_length"]).to_numpy(float)),
           "g_norm": np.linalg.norm(G, axis=1), "m_norm": np.linalg.norm(M, axis=1)}

    rep("=" * 70)
    rep("Step 4: negative sampling")
    rep("=" * 70)
    rep(f"embeddings: miRNA={P['mirna_emb'].name} {M.shape}, gene={P['gene_emb'].name} {G.shape}")
    rep(f"positives: {n_pos_all:,}" + (f" -> subsampled to {n_neg:,}" if n_neg != n_pos_all else "")
        + f"; negatives per strategy: {n_neg:,}")
    rep(f"grid: {n_mi:,} x {n_g:,} = {n_mi * n_g:,}; known pairs excluded: {len(known[0]):,}")

    t0 = time.time()
    D = Distances(M, G)
    rep(f"Mahalanobis: S = LedoitWolf covariance of the {n_mi + n_g:,} pooled miRNA+gene vectors "
        f"(shrinkage={D.shrinkage:.4f}); S^-1 precomputed. The paper does not define S.")
    pos_d = D.pairs(pos["mirna_idx"].to_numpy(), pos["gene_idx"].to_numpy())
    thresholds = dict(zip(METRICS, pos_d.mean(0)))
    rep("thresholds (mean over positive pairs): "
        + ", ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))

    counts = scan_grid(D, known, thresholds, n_mi, n_g, ncfg["chunk_mirnas"])
    C = counts["candidates"]
    rep(f"candidate pool (grid minus known): {C:,}  (scan {time.time() - t0:.0f}s)")
    rep("candidates above threshold:")
    for k in ["euc", "cos", "mah", "euc&cos", "euc&mah", "cos&mah", "paper"]:
        lab = "all three (paper pool)" if k == "paper" else k
        rep(f"  {lab:<24}{counts[k]:>12,}  ({counts[k] / C:.2%} of candidates)")
    rep("  paper reference: 4,932,554 of ~38M")
    rep()

    # ---- sampling
    select = {}
    for name, pool in [("random", C), ("paper", counts["paper"])]:
        if name not in strategies:
            continue
        if pool < n_neg:
            rep(f"STOP: {name} pool has {pool:,} pairs < {n_neg:,} needed")
            rep.save(P["report"])
            raise SystemExit(1)
        select[name] = np.sort(rng.choice(pool, size=n_neg, replace=False))
    picked = scan_grid(D, known, thresholds, n_mi, n_g, ncfg["chunk_mirnas"], select) if select else {}
    if "degree_aware" in strategies:
        picked["degree_aware"] = degree_aware(pos, known, n_mi, n_g, rng, rep)
        rep()

    # ---- assemble, save, diagnose
    pos_rows = pos.assign(label=1, d_euc=pos_d[:, 0], d_cos=pos_d[:, 1], d_mah=pos_d[:, 2])
    summaries, dfs = {}, {}
    for name in strategies:
        neg_pairs = picked[name]
        nd = D.pairs(neg_pairs[:, 0], neg_pairs[:, 1])
        neg_rows = pd.DataFrame({"mirna_idx": neg_pairs[:, 0], "gene_idx": neg_pairs[:, 1],
                                 "label": 0, "d_euc": nd[:, 0], "d_cos": nd[:, 1],
                                 "d_mah": nd[:, 2]})
        df = pd.concat([pos_rows, neg_rows], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df[["mirna_idx", "gene_idx"]] = df[["mirna_idx", "gene_idx"]].astype(np.int32)
        df["label"] = df["label"].astype(np.int8)
        # integrity: no negative is a known pair, no duplicates
        key = df["mirna_idx"].astype(np.int64) * n_g + df["gene_idx"]
        kkey = known[0].astype(np.int64) * n_g + known[1]
        assert not np.isin(key[df["label"] == 0], kkey).any(), "negative overlaps known pairs"
        assert not key.duplicated().any(), "duplicate pairs"
        df.to_parquet(P["out"](name), index=False)
        rep(f"saved {P['out'](name).relative_to(PROJECT_ROOT)}: {len(df):,} rows")
        dfs[name] = df
    rep()

    rep("=" * 70)
    rep("Diagnostics")
    rep("=" * 70)
    for name, df in dfs.items():
        summaries[name] = diagnose(name, df, ctx, rep, seed)

    rep("=" * 70)
    rep("Summary")
    rep("=" * 70)
    tab = pd.DataFrame(summaries)
    w = max(len(i) for i in tab.index) + 2
    rep(f"{'':<{w}}" + "".join(f"{c:>15}" for c in tab.columns))
    for idx, row in tab.iterrows():
        fmt = (lambda v: f"{int(v):,}") if idx.startswith("pure") else (
            (lambda v: f"{v:.2%}") if idx.startswith("rows") else (lambda v: f"{v:+.3f}"
                                                                   if idx.startswith("rho")
                                                                   else f"{v:.3f}"))
        rep(f"{idx:<{w}}" + "".join(f"{fmt(v):>15}" for v in row))
    rep("KS = Kolmogorov-Smirnov statistic pos vs neg (0 = same distribution, 1 = disjoint).")
    rep("AUC > 0.5 from a single non-biological feature = a shortcut the model can exploit.")
    rep.save(P["report"])
    plot(dfs, P["png"])
    print(f"\nSaved {P['report'].relative_to(PROJECT_ROOT)}, {P['png'].relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
