"""Step 8: resumable experiment queue (Blocks A, B, C) + aggregation.

Run from the project root:
  python3 -m src.experiments --plan     # build jobs.csv, print counts + time estimate, stop
  python3 -m src.experiments --run      # run pending jobs in order (resumable)
  python3 -m src.experiments --report   # rebuild summary.md, CSVs and plots

Every job writes a metrics.json; a job whose metrics.json exists is skipped, so interrupted
runs continue where they stopped. Each job runs in its own subprocess (crash/memory isolation).
random_kfold fold 0 == random_holdout fold 0 (same rows): such runs are copied from the
holdout run when that run used the same settings.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.parse import Report
from src.utils import PROJECT_ROOT, load_config

# numpy 2.x + macOS Accelerate: spurious matmul FP warnings (verified harmless in step 3).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

STRATS = ["paper", "random", "degree_aware"]
SPLITS = ["random_kfold", "cold_mirna", "cold_gene"]
FOLDS = range(5)
LATENTS = [8, 16, 32, 64, 128]
BASELINES = {"distances": "a) 3 distances", "log_length": "b) log10 gene length",
             "degree": "c) train-row degree"}
EXP = PROJECT_ROOT / "results" / "experiments"
JOBS_CSV = EXP / "jobs.csv"
LOG = PROJECT_ROOT / "results" / "step8_run.log"


# --------------------------------------------------------------------------- paths
def _rel(p) -> str:
    """Project-relative path string (jobs.csv must not contain machine-specific paths)."""
    return str(Path(p).resolve().relative_to(PROJECT_ROOT)) if p else ""


def _abs(p) -> Path:
    return PROJECT_ROOT / p

def ae_out(cfg, s, sp, f, z):
    return PROJECT_ROOT / cfg["autoencoder"]["out_dir"] / f"{s}_{sp}_f{f}_z{z}"


def clf_out(cfg, s, sp, f, model, inp, z):
    return PROJECT_ROOT / cfg["classifier"]["out_dir"] / f"{s}_{sp}_f{f}_{model}_{inp}_z{z}"


def base_out(s, sp, f, b):
    return EXP / "baselines" / f"{s}_{sp}_f{f}_{b}"


# --------------------------------------------------------------------------- job list
def build_jobs(cfg):
    jobs = []

    def add(block, jtype, **kw):
        j = {"block": block, "type": jtype, "strategy": None, "split": None, "fold": None,
             "latent": None, "model": None, "input": None, "test_strategy": None,
             "batch_size": None, "max_epochs": None, **kw}
        if jtype == "ae":
            out = ae_out(cfg, j["strategy"], j["split"], j["fold"], j["latent"])
            dep = ""
        elif jtype == "clf":
            out = clf_out(cfg, j["strategy"], j["split"], j["fold"], j["model"], j["input"], j["latent"])
            dep = "" if j["input"] == "original_only" \
                else str(ae_out(cfg, j["strategy"], j["split"], j["fold"], j["latent"]) / "metrics.json")
        elif jtype == "cross":
            run = clf_out(cfg, j["strategy"], j["split"], j["fold"], j["model"], j["input"], j["latent"])
            out = run / f"cross_{j['test_strategy']}"
            dep = str(run / "metrics.json")
        else:  # baseline
            out = base_out(j["strategy"], j["split"], j["fold"], j["model"])
            dep = ""
        j["output"] = _rel(out / "metrics.json")
        j["depends_on"] = _rel(dep) if dep else ""
        j["job_id"] = (f"{block}:{jtype}:{j['strategy']}:{j['split']}:f{j['fold']}:"
                       f"{j['model'] or ''}:{j['input'] or ''}:z{j['latent'] or ''}"
                       + (f":->{j['test_strategy']}" if j["test_strategy"] else ""))
        if j["job_id"] not in {x["job_id"] for x in jobs}:
            jobs.append(j)

    # ---- Block A (paper)
    for z in LATENTS:
        add("A1", "ae", strategy="paper", split="random_holdout", fold=0, latent=z)
        for inp in ["latent_only", "original_plus_latent"]:
            add("A1", "clf", strategy="paper", split="random_holdout", fold=0, latent=z,
                model="lstm", input=inp)
    for f in FOLDS:
        add("A2", "ae", strategy="paper", split="random_kfold", fold=f, latent=64)
        for inp in ["original_plus_latent", "original_only"]:
            add("A2", "clf", strategy="paper", split="random_kfold", fold=f, latent=64,
                model="lstm", input=inp)
    for m in ["bilstm", "gru", "transformer"]:
        add("A3", "clf", strategy="paper", split="random_holdout", fold=0, latent=64, model=m,
            input="original_plus_latent")
    add("A4", "clf", strategy="paper", split="random_holdout", fold=0, latent=64,
        model="lstm_timesteps", input="original_plus_latent", batch_size=1024, max_epochs=3)

    # ---- Block B
    for s in STRATS:
        for sp in SPLITS:
            for f in FOLDS:
                add("B", "ae", strategy=s, split=sp, fold=f, latent=64)
                add("B", "clf", strategy=s, split=sp, fold=f, latent=64, model="mlp",
                    input="original_plus_latent")
                if f == 0:
                    add("B", "clf", strategy=s, split=sp, fold=0, latent=64, model="lstm",
                        input="original_plus_latent")
    for s in STRATS:
        for sp in SPLITS:
            for f in FOLDS:
                models = ["mlp"] + (["lstm"] if f == 0 else [])
                for m in models:
                    for t in STRATS:
                        if t != s:
                            add("B", "cross", strategy=s, split=sp, fold=f, latent=64,
                                model=m, input="original_plus_latent", test_strategy=t)

    # ---- Block C (baselines a-c incl. cross inside the job; d = logreg original_only)
    for s in STRATS:
        for sp in SPLITS:
            for f in FOLDS:
                for b in BASELINES:
                    add("C", "baseline", strategy=s, split=sp, fold=f, model=b)
                add("C", "clf", strategy=s, split=sp, fold=f, latent=64, model="logreg",
                    input="original_only")
    df = pd.DataFrame(jobs)
    order = {"A1": 0, "A2": 1, "A3": 2, "C": 3, "B": 4, "A4": 5}
    df["_o"] = df["block"].map(order)
    df = df.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    return df


def refresh_status(df):
    done = df["output"].map(lambda p: _abs(p).exists())
    df["status"] = np.where(done, "done", df.get("status", "pending"))
    df.loc[~done & ~df["status"].isin(["failed", "blocked"]), "status"] = "pending"
    return df


def load_or_build(cfg):
    EXP.mkdir(parents=True, exist_ok=True)
    df = build_jobs(cfg)
    if JOBS_CSV.exists():
        old = pd.read_csv(JOBS_CSV).set_index("job_id")
        for col in ["status", "seconds"]:
            if col in old:
                df[col] = df["job_id"].map(old[col])
    if "seconds" not in df:
        df["seconds"] = np.nan
    df["status"] = df.get("status", pd.Series("pending", index=df.index)).fillna("pending")
    return refresh_status(df)


# --------------------------------------------------------------------------- estimate
def estimate(cfg, df):
    """Seconds per pending job, from step 6/7 timings."""
    ae_root = PROJECT_ROOT / cfg["autoencoder"]["out_dir"]
    ae_t = [json.loads(p.read_text())["train_seconds"] + json.loads(p.read_text()).get("probe_seconds", 0)
            for p in ae_root.glob("*/metrics.json")]
    clf_root = PROJECT_ROOT / cfg["classifier"]["out_dir"]
    clf_t = {}
    for p in clf_root.glob("*/metrics.json"):
        rc = json.loads((p.parent / "config.json").read_text())
        clf_t.setdefault((rc["model"], rc["input_variant"]), []).append(
            json.loads(p.read_text())["seconds_total"])
    bench = cfg["classifier"].get("benchmark") or {}
    lstm_epochs = np.mean([json.loads(p.read_text())["epochs_run"]
                           for p in clf_root.glob("*lstm_*/metrics.json")] or [15])
    mlp_ep = bench.get("mlp_cpu", 8) + 1.0

    def est(j):
        src = _reuse_source(cfg, j)
        if src is not None and (src / "metrics.json").exists():
            rc = json.loads((src / "config.json").read_text())
            c = cfg["classifier"]
            want = (c.get("max_epochs_by_model") or {}).get(j["model"]) or c["max_epochs"]
            if j["type"] == "ae" or rc.get("max_epochs") == want:
                return 5
        if j["type"] == "ae":
            return np.mean(ae_t) if ae_t else 700
        if j["type"] == "cross":
            return 45
        if j["type"] == "baseline":
            return 40
        m, inp = j["model"], j["input"]
        if m == "lstm" and (m, inp) in clf_t:
            return np.mean(clf_t[(m, inp)])
        if m == "lstm":
            return np.mean(clf_t.get(("lstm", "original_plus_latent"), [900]))
        if m == "mlp":
            return 100 * mlp_ep * 0.75 + 30           # up to 100 epochs; assume ~75 used
        if m == "logreg":
            return 60
        if m == "lstm_timesteps":
            return 3 * timesteps_epoch_seconds(cfg)
        return bench.get(f"{m}_cpu", 60) * (lstm_epochs + 1) + 60
    return df.apply(est, axis=1)


def timesteps_epoch_seconds(cfg, n_batches=20):
    """Time a few batch-1024 batches of lstm_timesteps and extrapolate one epoch (cached)."""
    cache = EXP / "timesteps_bench.json"
    if cache.exists():
        return json.loads(cache.read_text())["epoch_seconds"]
    import torch
    from src.classifier import build_model
    torch.set_num_threads(cfg["classifier"].get("num_threads") or 4)
    model = build_model("lstm_timesteps", [64, 64, 64])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = [torch.randn(1024, 64) for _ in range(3)]
    y = torch.randint(0, 2, (1024,)).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for _ in range(n_batches):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    per_batch = (time.time() - t0) / n_batches
    n_train = 1_670_122
    sec = per_batch * np.ceil(n_train / 1024)
    cache.write_text(json.dumps({"per_batch": per_batch, "epoch_seconds": sec}))
    return sec


def plan(cfg):
    df = load_or_build(cfg)
    df["est_seconds"] = estimate(cfg, df)
    df.to_csv(JOBS_CSV, index=False)
    pend = df[df["status"] != "done"]
    print(f"jobs: {len(df)} total, {int((df['status'] == 'done').sum())} already done (reused), "
          f"{len(pend)} pending")
    tab = df.assign(pending=df["status"] != "done").groupby(["block", "type"]).agg(
        total=("job_id", "size"), pending=("pending", "sum"),
        est_hours=("est_seconds", lambda s: s[df.loc[s.index, "status"] != "done"].sum() / 3600))
    print(tab.to_string(float_format=lambda v: f"{v:.1f}"))
    blk = pend.groupby(pend["block"].map(block_group), sort=False)["est_seconds"].sum() / 3600
    print("estimated hours per block: " + ", ".join(f"{k}={v:.1f}" for k, v in blk.items()))
    print(f"estimated total: {pend['est_seconds'].sum() / 3600:.1f} h "
          f"(lstm_timesteps epoch at batch 1024 ~{timesteps_epoch_seconds(cfg) / 60:.0f} min)")
    print(f"job list saved to {JOBS_CSV.relative_to(PROJECT_ROOT)}")
    return df


# --------------------------------------------------------------------------- job execution
def _stdout_is_log() -> bool:
    try:
        return LOG.exists() and os.path.samefile(f"/dev/fd/{sys.stdout.fileno()}", LOG)
    except (OSError, ValueError):
        return False


def _log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    if _stdout_is_log():
        print(line, flush=True)
    else:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def _reuse_source(cfg, j):
    """The holdout run this random_kfold fold-0 job can be copied from, or None.
    The source need not exist yet if an earlier job in the queue will create it."""
    if j["split"] != "random_kfold" or int(j["fold"]) != 0 or j["type"] not in ("ae", "clf"):
        return None
    if j["type"] == "ae":
        return ae_out(cfg, j["strategy"], "random_holdout", 0, int(j["latent"]))
    return clf_out(cfg, j["strategy"], "random_holdout", 0, j["model"], j["input"], int(j["latent"]))


def _try_reuse_holdout(cfg, j) -> bool:
    """random_kfold fold 0 == random_holdout fold 0: copy the holdout run if settings match."""
    if j["split"] != "random_kfold" or int(j["fold"]) != 0:
        return False
    if j["type"] == "ae":
        src = ae_out(cfg, j["strategy"], "random_holdout", 0, int(j["latent"]))
    elif j["type"] == "clf":
        src = clf_out(cfg, j["strategy"], "random_holdout", 0, j["model"], j["input"], int(j["latent"]))
    else:
        return False
    if not (src / "metrics.json").exists():
        return False
    if j["type"] == "clf":
        rc = json.loads((src / "config.json").read_text())
        c = cfg["classifier"]
        want = (c.get("max_epochs_by_model") or {}).get(j["model"]) or c["max_epochs"]
        if j["model"] != "logreg" and (rc.get("max_epochs") != want or rc.get("batch_size") != c["batch_size"]):
            return False
    dst = _abs(j["output"]).parent
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("cross_*"))
    meta = json.loads((dst / "config.json").read_text())
    meta.update(reused_from=src.name, split_type="random_kfold")
    if j["type"] == "clf" and meta.get("ae_run"):
        meta["ae_run"] = meta["ae_run"].replace("random_holdout", "random_kfold")
    (dst / "config.json").write_text(json.dumps(meta, indent=1))
    return True


def run_one(job: dict):
    """Executed in a subprocess: python3 -m src.experiments --job '<json>'."""
    cfg = load_config()
    j = job
    if _try_reuse_holdout(cfg, j):
        _log(f"  reused holdout run for {j['job_id']}")
        return
    if j["type"] == "ae":
        from src.autoencoder import train_ae
        train_ae(j["strategy"], j["split"], int(j["fold"]), int(j["latent"]), cfg, _log)
    elif j["type"] == "clf":
        from src.classifier import train_clf
        train_clf(j["strategy"], j["split"], int(j["fold"]), j["model"], j["input"],
                  int(j["latent"]), cfg, _log,
                  batch_size=None if pd.isna(j["batch_size"]) else int(j["batch_size"]),
                  max_epochs=None if pd.isna(j["max_epochs"]) else int(j["max_epochs"]))
    elif j["type"] == "cross":
        from src.classifier import predict_cross
        run = _abs(j["output"]).parent.parent
        predict_cross(run, j["test_strategy"], j["split"], int(j["fold"]), _log)
    else:
        run_baseline(cfg, j["strategy"], j["split"], int(j["fold"]), j["model"])


def block_group(block: str) -> str:
    return "A4" if block == "A4" else block[0]


def run_all(cfg):
    df = load_or_build(cfg)
    df["est_seconds"] = estimate(cfg, df)
    df.to_csv(JOBS_CSV, index=False)
    _log(f"=== run start: {int((df['status'] != 'done').sum())} pending jobs")
    block_t0, cur_block = time.time(), None
    times = {}
    for i, j in df.iterrows():
        blk = block_group(j["block"])
        if blk != cur_block:
            if cur_block is not None:
                times[cur_block] = time.time() - block_t0
                _log(f"=== block {cur_block} finished in {times[cur_block] / 3600:.2f} h")
                write_summary(cfg, block_times=times)
            cur_block, block_t0 = blk, time.time()
        if _abs(j["output"]).exists():
            df.loc[i, "status"] = "done"
            continue
        if isinstance(j["depends_on"], str) and j["depends_on"] and not _abs(j["depends_on"]).exists():
            df.loc[i, "status"] = "blocked"
            _log(f"BLOCKED {j['job_id']} (missing {j['depends_on']})")
            df.to_csv(JOBS_CSV, index=False)
            continue
        _log(f"START {j['job_id']}")
        t0 = time.time()
        payload = json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                              for k, v in j.to_dict().items()})
        sys.stdout.flush()
        # stdout/stderr inherited: job logs and tracebacks go to the same stream as the queue
        r = subprocess.run([sys.executable, "-m", "src.experiments", "--job", payload],
                           cwd=PROJECT_ROOT)
        sec = time.time() - t0
        ok = r.returncode == 0 and _abs(j["output"]).exists()
        df.loc[i, ["status", "seconds"]] = ["done" if ok else "failed", sec]
        if not ok:
            _log(f"FAILED {j['job_id']} after {sec:.0f}s (exit {r.returncode}; traceback above)")
        else:
            _log(f"DONE {j['job_id']} in {sec / 60:.1f} min")
        df.to_csv(JOBS_CSV, index=False)
    if cur_block is not None:
        times[cur_block] = time.time() - block_t0
        _log(f"=== block {cur_block} finished in {times[cur_block] / 3600:.2f} h")
    write_summary(cfg, block_times=times)
    _log("=== run finished: " + df["status"].value_counts().to_json())


# --------------------------------------------------------------------------- baselines
def _ctx(cfg):
    pr = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    gi = pd.read_csv(pr / "gene_index.csv")
    genes = pd.read_csv(PROJECT_ROOT / cfg["paths"]["interim_dir"] / "genes.csv").set_index("gene_id")
    return np.log10(gi["gene_id"].map(genes["genomic_length"]).to_numpy(float))


def _pairs(cfg, s):
    return pd.read_parquet(PROJECT_ROOT / cfg["paths"]["processed_dir"] / f"pairs_{s}.parquet")


def baseline_features(name, d, rows, log_len, deg_mi=None, deg_g=None):
    if name == "distances":
        return d.loc[rows, ["d_euc", "d_cos", "d_mah"]].to_numpy()
    if name == "log_length":
        return log_len[d.loc[rows, "gene_idx"].to_numpy()][:, None]
    mi = d.loc[rows, "mirna_idx"].to_numpy()
    g = d.loc[rows, "gene_idx"].to_numpy()
    return np.column_stack([np.log1p(deg_mi[mi]), np.log1p(deg_g[g])])


def run_baseline(cfg, s, sp, f, name):
    """Logistic regression shortcut baseline; own test + cross to the other strategies."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.classifier import metrics_at
    from src.splits import load_cross, load_split
    log_len = _ctx(cfg)
    assert len(log_len) == 16796, "gene index size changed; update the 16796 constant"
    d = _pairs(cfg, s)
    tr, va, te = load_split(s, sp, f)
    n_mi, n_g = int(d["mirna_idx"].max()) + 1, 16796
    pos_tr = d.loc[tr][d.loc[tr, "label"] == 1]
    deg_mi = np.bincount(pos_tr["mirna_idx"], minlength=n_mi).astype(float)
    deg_g = np.bincount(pos_tr["gene_idx"], minlength=n_g).astype(float)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg["seed"]))
    clf.fit(baseline_features(name, d, tr, log_len, deg_mi, deg_g), d.loc[tr, "label"].to_numpy())
    out = base_out(s, sp, f, name)
    out.mkdir(parents=True, exist_ok=True)

    def evaluate(dd, rows, test_strategy):
        score = clf.predict_proba(baseline_features(name, dd, rows, log_len, deg_mi, deg_g))[:, 1]
        y = dd.loc[rows, "label"].to_numpy()
        m = metrics_at(y, score)
        m.update(extra_metrics(cfg, test_strategy, rows, y, score, dd))
        return m

    m = evaluate(d, te, s)
    m.update(baseline=name, strategy=s, split_type=sp, fold=f, test_strategy=s)
    for t in STRATS:
        if t == s:
            continue
        _, _, te_t = load_cross(s, t, sp, f)
        mc = evaluate(_pairs(cfg, t), te_t, t)
        mc.update(baseline=name, strategy=s, split_type=sp, fold=f, test_strategy=t)
        (out / f"cross_{t}").mkdir(exist_ok=True)
        (out / f"cross_{t}" / "metrics.json").write_text(json.dumps(mc, indent=1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1))


# --------------------------------------------------------------------------- extra metrics
_POS_EVIDENCE = None


def extra_metrics(cfg, test_strategy, rows, y, score, pairs_df=None):
    """strong-evidence AUC and mean per-miRNA AUC (>=10 pos and >=10 neg)."""
    from sklearn.metrics import roc_auc_score
    global _POS_EVIDENCE
    if _POS_EVIDENCE is None:
        p = pd.read_csv(PROJECT_ROOT / cfg["paths"]["processed_dir"] / "positives.csv",
                        usecols=["mirna_idx", "gene_idx", "evidence_level"])
        _POS_EVIDENCE = pd.Series(p["evidence_level"].to_numpy(),
                                  index=p["mirna_idx"].to_numpy(np.int64) * 16796 + p["gene_idx"].to_numpy())
    d = pairs_df if pairs_df is not None else _pairs(cfg, test_strategy)
    mi = d["mirna_idx"].to_numpy()[rows]
    g = d["gene_idx"].to_numpy()[rows]
    key = mi.astype(np.int64) * 16796 + g
    strong = (y == 1) & (pd.Series(key).map(_POS_EVIDENCE).to_numpy() == "strong")
    keep = strong | (y == 0)
    out = {"strong_auc": float(roc_auc_score(y[keep], score[keep])) if strong.any() else None,
           "n_strong_pos": int(strong.sum())}
    t = pd.DataFrame({"m": mi, "y": y, "s": score})
    t["r"] = t.groupby("m")["s"].rank()
    g_ = t.groupby("m").agg(n_pos=("y", "sum"), n=("y", "size"))
    g_["n_neg"] = g_["n"] - g_["n_pos"]
    rank_pos = t[t["y"] == 1].groupby("m")["r"].sum()
    ok = g_[(g_["n_pos"] >= 10) & (g_["n_neg"] >= 10)].index
    auc = ((rank_pos.loc[ok] - g_.loc[ok, "n_pos"] * (g_.loc[ok, "n_pos"] + 1) / 2)
           / (g_.loc[ok, "n_pos"] * g_.loc[ok, "n_neg"]))
    out.update(per_mirna_auc=float(auc.mean()) if len(ok) else None,
               n_mirnas_per_entity=int(len(ok)))
    return out


def ensure_extra(cfg, metrics_path: Path, test_strategy: str):
    """Extra metrics for neural/logreg runs from saved predictions (cached)."""
    ex = metrics_path.parent / "extra_metrics.json"
    if ex.exists():
        return json.loads(ex.read_text())
    pred = metrics_path.parent / "predictions.parquet"
    if not pred.exists():
        return {}
    p = pd.read_parquet(pred)
    e = extra_metrics(cfg, test_strategy, p["row"].to_numpy(), p["label"].to_numpy(),
                      p["score"].to_numpy())
    ex.write_text(json.dumps(e, indent=1))
    return e


# --------------------------------------------------------------------------- aggregation
def collect(cfg):
    rows = []
    clf_root = PROJECT_ROOT / cfg["classifier"]["out_dir"]
    for p in clf_root.glob("*/metrics.json"):
        rc = json.loads((p.parent / "config.json").read_text())
        m = json.loads(p.read_text())
        base = dict(kind="model", strategy=rc["strategy"], split=rc["split_type"], fold=rc["fold"],
                    model=rc["model"], input=rc["input_variant"], latent=rc["latent"],
                    max_epochs=rc.get("max_epochs"), batch_size=rc.get("batch_size"))
        rows.append({**base, "test_strategy": rc["strategy"], **m,
                     **ensure_extra(cfg, p, rc["strategy"])})
        for cp in p.parent.glob("cross_*/metrics.json"):
            mc = json.loads(cp.read_text())
            rows.append({**base, "test_strategy": mc["test_strategy"], **mc,
                         **ensure_extra(cfg, cp, mc["test_strategy"])})
    for p in (EXP / "baselines").glob("*/metrics.json"):
        for mp in [p] + list(p.parent.glob("cross_*/metrics.json")):
            m = json.loads(mp.read_text())
            rows.append({"kind": "baseline", "strategy": m["strategy"], "split": m["split_type"],
                         "fold": m["fold"], "model": m["baseline"], "input": None, "latent": None,
                         **m})
    df = pd.DataFrame(rows)
    for c in ["confusion"]:
        if c in df:
            df = df.drop(columns=c)
    return df


def ms(s):
    s = s.dropna()
    if len(s) == 0:
        return "–"
    return f"{s.mean():.4f}" if len(s) == 1 else f"{s.mean():.4f} ± {s.std():.4f}"


def write_summary(cfg, block_times=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EXP.mkdir(parents=True, exist_ok=True)
    df = collect(cfg)
    if df.empty:
        return
    df.to_csv(EXP / "all_runs.csv", index=False)
    ref = cfg.get("paper_reference", {})
    L = []
    L.append("# Step 8 experiment summary\n")
    if JOBS_CSV.exists():
        jobs = pd.read_csv(JOBS_CSV)
        L.append("Jobs: " + ", ".join(f"{k} {v}" for k, v in jobs["status"].value_counts().items()))
        if block_times:
            L.append("Block wall time: " + ", ".join(f"{k} {v / 3600:.2f} h" for k, v in block_times.items()))
        L.append("")
    own = df[df["test_strategy"] == df["strategy"]]
    paper_m = own[(own["kind"] == "model") & (own["strategy"] == "paper")]

    # ---- Table 2
    L.append("## Replication: paper Table 2 (lstm, random holdout, by latent size)\n")
    L.append("| latent | latent_only | paper | original_plus_latent | paper |")
    L.append("|---|---|---|---|---|")
    t2 = paper_m[(paper_m["split"] == "random_holdout") & (paper_m["model"] == "lstm")]
    t2rows = []
    for z in LATENTS:
        r = {"latent": z}
        cells = []
        for inp in ["latent_only", "original_plus_latent"]:
            v = t2[(t2["latent"] == z) & (t2["input"] == inp)]["roc_auc"]
            pv = (ref.get("table2", {}).get(z) or {}).get(inp)
            r[inp], r[f"paper_{inp}"] = (v.iloc[0] if len(v) else None), pv
            cells += [ms(v), f"{pv:.4f}" if pv else "n/a"]
        t2rows.append(r)
        L.append(f"| {z} | " + " | ".join(cells) + " |")
    pd.DataFrame(t2rows).to_csv(EXP / "table2.csv", index=False)
    L.append("\nROC-AUC. 'n/a' = paper value not entered in configs (paper_reference).\n")

    # ---- Table 3 / 11
    L.append("## Replication: paper Tables 3 and 11 (lstm, 5-fold CV, latent 64)\n")
    L.append("original_plus_latent = AEmiGAP (paper Table 3 average ± std over its 5 folds); "
             "original_only = miGAP (Table 11, average only). Paper folds are numbered 1–5, "
             "ours 0–4: compare mean ± std, not fold by fold. The paper reports no PR-AUC.\n")
    L.append("| input | metric | ours (mean ± std, 5 folds) | paper |")
    L.append("|---|---|---|---|")
    kf = paper_m[(paper_m["split"] == "random_kfold") & (paper_m["model"] == "lstm") & (paper_m["latent"] == 64)]
    t3 = []
    t3_ref = ref.get("table3") or {}
    fold_cols = ["precision", "recall", "accuracy", "f1", "roc_auc"]
    paper_folds = pd.DataFrame(t3_ref.get("folds") or [], columns=fold_cols)
    for inp, key in [("original_plus_latent", "table11_aemigap"), ("original_only", "table11_migap")]:
        sub = kf[kf["input"] == inp]
        for met in ["roc_auc", "pr_auc", "accuracy", "f1", "precision", "recall"]:
            pv = (ref.get(key) or {}).get(met)
            psd = (paper_folds[met].std() if inp == "original_plus_latent" and met in paper_folds
                   and len(paper_folds) > 1 else None)
            ptxt = "n/a" if pv is None else (f"{pv:.4f} ± {psd:.4f}" if psd is not None else f"{pv:.4f}")
            L.append(f"| {inp} | {met} | {ms(sub[met])} (n={sub[met].notna().sum()}) | {ptxt} |")
            t3.append({"input": inp, "metric": met, "mean": sub[met].mean(), "std": sub[met].std(),
                       "n": int(sub[met].notna().sum()), "paper": pv, "paper_fold_std": psd})
    pd.DataFrame(t3).to_csv(EXP / "table3_11.csv", index=False)
    L.append("")

    # ---- Table 13
    L.append("## Replication: paper Table 13 (models, random holdout, original_plus_latent, z=64)\n")
    L.append("| model | ROC-AUC | PR-AUC | acc | F1 | best/run epochs | paper ROC-AUC | note |")
    L.append("|---|---|---|---|---|---|---|---|")
    t13 = paper_m[(paper_m["split"] == "random_holdout") & (paper_m["input"] == "original_plus_latent")
                  & (paper_m["latent"] == 64)]
    rows13 = []
    for m in ["lstm", "bilstm", "gru", "transformer", "lstm_timesteps", "mlp"]:
        sub = t13[t13["model"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        pv = (ref.get("table13") or {}).get(m)
        note = (f"capped: batch {int(r['batch_size'])}, max {int(r['max_epochs'])} epochs"
                if m == "lstm_timesteps" else (f"max {int(r['max_epochs'])} epochs" if m == "mlp" else ""))
        L.append(f"| {m} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['accuracy']:.4f} | "
                 f"{r['f1']:.4f} | {int(r['best_epoch'])}/{int(r['epochs_run'])} | "
                 f"{f'{pv:.4f}' if pv else 'n/a'} | {note} |")
        rows13.append({"model": m, "roc_auc": r["roc_auc"], "paper": pv, "note": note})
    pd.DataFrame(rows13).to_csv(EXP / "table13.csv", index=False)
    L.append("")

    # ---- 3x3 matrices
    L.append("## Robustness: train strategy x test strategy (ROC-AUC)\n")
    L.append("Rows = training negatives, columns = test negatives. MLP = mean ± std over 5 folds; "
             "floor = best of shortcut baselines a–c (mean over folds, same cell); "
             "LSTM = fold 0 only.\n")
    mat_rows = []
    for sp in SPLITS:
        cell = df[df["split"] == sp]
        L.append(f"### {sp}\n")
        L.append("| train \\ test | " + " | ".join(STRATS) + " |")
        L.append("|---|" + "---|" * len(STRATS))
        for s in STRATS:
            parts = []
            for t in STRATS:
                c = cell[(cell["strategy"] == s) & (cell["test_strategy"] == t)]
                mlp = c[(c["kind"] == "model") & (c["model"] == "mlp")
                        & (c["input"] == "original_plus_latent")]["roc_auc"]
                lstm = c[(c["kind"] == "model") & (c["model"] == "lstm") & (c["fold"] == 0)
                         & (c["input"] == "original_plus_latent") & (c["latent"] == 64)]["roc_auc"]
                base = c[c["kind"] == "baseline"].groupby("model")["roc_auc"].mean()
                floor = base.max() if len(base) else np.nan
                which = base.idxmax() if len(base) else ""
                parts.append(f"MLP {ms(mlp)}<br>floor {floor:.3f} ({which})<br>LSTM f0 "
                             f"{lstm.iloc[0]:.4f}" if len(lstm) else
                             f"MLP {ms(mlp)}<br>floor {floor:.3f} ({which})<br>LSTM f0 –")
                mat_rows.append({"split": sp, "train": s, "test": t, "mlp_mean": mlp.mean(),
                                 "mlp_std": mlp.std(), "mlp_n": len(mlp), "floor": floor,
                                 "floor_baseline": which,
                                 "lstm_f0": lstm.iloc[0] if len(lstm) else np.nan,
                                 **{f"base_{k}": v for k, v in base.items()}})
            L.append(f"| **{s}** | " + " | ".join(parts) + " |")
        L.append("")
    mat = pd.DataFrame(mat_rows)
    mat.to_csv(EXP / "matrix.csv", index=False)

    # ---- strong-evidence / per-entity
    L.append("## Strong-evidence AUC and per-miRNA AUC (own test negatives)\n")
    L.append("| split | train=test strategy | model | ROC-AUC | strong-evidence AUC | per-miRNA AUC (n miRNAs) |")
    L.append("|---|---|---|---|---|---|")
    ex_rows = []
    for sp in SPLITS:
        for s in STRATS:
            c = own[(own["split"] == sp) & (own["strategy"] == s)]
            for name, sel in [("mlp", (c["kind"] == "model") & (c["model"] == "mlp")),
                              ("lstm f0", (c["kind"] == "model") & (c["model"] == "lstm") & (c["fold"] == 0)
                               & (c["input"] == "original_plus_latent") & (c["latent"] == 64)),
                              ("logreg original_only", (c["kind"] == "model") & (c["model"] == "logreg")),
                              ("floor: distances", (c["kind"] == "baseline") & (c["model"] == "distances")),
                              ("floor: degree", (c["kind"] == "baseline") & (c["model"] == "degree"))]:
                sub = c[sel]
                if sub.empty:
                    continue
                n_m = sub["n_mirnas_per_entity"].mean() if "n_mirnas_per_entity" in sub else np.nan
                L.append(f"| {sp} | {s} | {name} | {ms(sub['roc_auc'])} | {ms(sub.get('strong_auc', pd.Series(dtype=float)))} | "
                         f"{ms(sub.get('per_mirna_auc', pd.Series(dtype=float)))} ({n_m:.0f}) |")
                ex_rows.append({"split": sp, "strategy": s, "model": name,
                                "roc_auc": sub["roc_auc"].mean(),
                                "strong_auc": sub.get("strong_auc", pd.Series(dtype=float)).mean(),
                                "per_mirna_auc": sub.get("per_mirna_auc", pd.Series(dtype=float)).mean()})
    pd.DataFrame(ex_rows).to_csv(EXP / "extra_metrics.csv", index=False)
    L.append("\nper-miRNA AUC: mean ROC-AUC within each test miRNA with >=10 positives and "
             ">=10 negatives (cold_mirna splits have no such miRNA in common with training, "
             "which is fine: the metric is computed on test rows only).\n")
    (EXP / "summary.md").write_text("\n".join(L))

    # ---- plots
    if not mat.empty and mat["mlp_n"].sum() > 0:
        for sp in SPLITS:
            m = mat[mat["split"] == sp]
            if m["mlp_n"].sum() == 0:
                continue
            grid = m.pivot(index="train", columns="test", values="mlp_mean").reindex(index=STRATS, columns=STRATS)
            floor = m.pivot(index="train", columns="test", values="floor").reindex(index=STRATS, columns=STRATS)
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            im = ax.imshow(grid.to_numpy(float), cmap="Blues", vmin=0.5, vmax=1.0)
            for i in range(3):
                for k in range(3):
                    v, f = grid.iat[i, k], floor.iat[i, k]
                    if np.isnan(v):
                        continue
                    ax.text(k, i, f"{v:.3f}\nfloor {f:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.8 else "#0b0b0b")
            ax.set_xticks(range(3), STRATS)
            ax.set_yticks(range(3), STRATS)
            ax.set_xlabel("test negatives")
            ax.set_ylabel("training negatives")
            ax.set_title(f"MLP ROC-AUC, {sp} (mean of 5 folds)", fontsize=11, loc="left")
            fig.colorbar(im, ax=ax, shrink=0.8, label="ROC-AUC")
            fig.tight_layout()
            fig.savefig(EXP / f"matrix_{sp}.png", dpi=140)
            plt.close(fig)
        m = mat.dropna(subset=["mlp_mean"])
        if len(m):
            fig, axes = plt.subplots(1, len(SPLITS), figsize=(15, 4.2), sharey=True)
            for ax, sp in zip(axes, SPLITS):
                c = m[m["split"] == sp]
                x = np.arange(len(c))
                ax.bar(x - 0.2, c["mlp_mean"], 0.38, yerr=c["mlp_std"].fillna(0), color="#2a78d6",
                       label="MLP (mean ± sd)")
                ax.bar(x + 0.2, c["floor"], 0.38, color="#eb6834", label="shortcut floor")
                ax.set_xticks(x, [f"{a[:5]}→{b[:5]}" for a, b in zip(c["train"], c["test"])],
                              rotation=60, fontsize=8)
                ax.axhline(0.5, color="#999", lw=1, ls="--")
                ax.set_ylim(0.4, 1.0)
                ax.set_title(sp, fontsize=10, loc="left")
                for spn in ("top", "right"):
                    ax.spines[spn].set_visible(False)
            axes[0].set_ylabel("ROC-AUC")
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle("Model vs shortcut floor per train→test cell", fontsize=11)
            fig.tight_layout()
            fig.savefig(EXP / "model_vs_floor.png", dpi=140)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--job")
    args = ap.parse_args()
    cfg = load_config()
    if args.job:
        try:
            run_one(json.loads(args.job))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    elif args.plan:
        plan(cfg)
    elif args.run:
        run_all(cfg)
    elif args.report:
        write_summary(cfg)


if __name__ == "__main__":
    main()
