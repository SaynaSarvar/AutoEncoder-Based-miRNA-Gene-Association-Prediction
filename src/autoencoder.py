"""Step 6: autoencoder on concatenated miRNA + gene embeddings (AEmiGAP section 4.2.2).

Run from the project root:
  python3 -m src.autoencoder --benchmark                       # CPU vs MPS, writes config
  python3 -m src.autoencoder --strategy paper --split random_holdout --fold 0 --latent 64
  python3 -m src.autoencoder --replication-set                 # Table 2 + Table 3 runs
  python3 -m src.autoencoder --report                          # rebuild report + plot

Input per pair: x = [mirna_vec, gene_vec] (128). The two embedding tables live on the device;
batches are gathered from index arrays (no n_rows x 128 matrix, no DataLoader).
MinMax scaling: per feature, fit on the training rows of the fold. Each feature comes from one
table, so the column min/max over training pairs equals the min/max over the miRNAs (genes)
that occur in training pairs; the tables are scaled once and val/test values clipped to [0,1].
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.embed import perf_cores
from src.parse import Report
from src.splits import load_split
from src.utils import PROJECT_ROOT, get_device, load_config, set_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LATENTS = [8, 16, 32, 64, 128]


# --------------------------------------------------------------------------- model
class AutoEncoder(nn.Module):
    def __init__(self, n_in: int = 128, latent: int = 64):
        super().__init__()
        h1, h2 = max(112, latent), max(96, latent)
        self.encoder = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                     nn.Linear(h2, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(),
                                     nn.Linear(h1, n_in), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- data
def _emb_paths(cfg):
    e = PROJECT_ROOT / cfg["paths"]["embeddings_dir"]
    n = cfg["negatives"]
    return e / n["mirna_embedding"], e / n["gene_embedding"]


def load_pairs(cfg, strategy):
    p = PROJECT_ROOT / cfg["paths"]["processed_dir"] / f"pairs_{strategy}.parquet"
    d = pd.read_parquet(p, columns=["mirna_idx", "gene_idx", "label"])
    return d["mirna_idx"].to_numpy(np.int64), d["gene_idx"].to_numpy(np.int64), d["label"].to_numpy()


def fit_scaler(M, G, mi_train, gi_train):
    """Column min/max over the training pairs' features (= over the entities they use)."""
    um, ug = np.unique(mi_train), np.unique(gi_train)
    lo = np.concatenate([M[um].min(0), G[ug].min(0)])
    hi = np.concatenate([M[um].max(0), G[ug].max(0)])
    return lo.astype(np.float32), hi.astype(np.float32)


def scaled_tables(M, G, lo, hi, device):
    rng = np.where(hi > lo, hi - lo, 1.0)
    d_m = M.shape[1]
    Ms = np.clip((M - lo[:d_m]) / rng[:d_m], 0, 1)
    Gs = np.clip((G - lo[d_m:]) / rng[d_m:], 0, 1)
    return (torch.tensor(Ms, dtype=torch.float32, device=device),
            torch.tensor(Gs, dtype=torch.float32, device=device))


class PairBatches:
    """Gathers x = [Ms[mi], Gs[gi]] for index slices; tables already scaled on the device."""

    def __init__(self, Ms, Gs, mi, gi, device):
        self.Ms, self.Gs, self.device = Ms, Gs, device
        self.mi = torch.as_tensor(mi, device=device)
        self.gi = torch.as_tensor(gi, device=device)
        self.n = len(mi)

    def x(self, sel):
        return torch.cat([self.Ms[self.mi[sel]], self.Gs[self.gi[sel]]], dim=1)

    def iter(self, batch, perm=None):
        for s in range(0, self.n, batch):
            sel = perm[s:s + batch] if perm is not None else slice(s, s + batch)
            yield self.x(sel)


@torch.no_grad()
def eval_mse(model, data: PairBatches, batch=65536):
    model.eval()
    total = 0.0
    for x in data.iter(batch):
        total += ((model(x) - x) ** 2).sum().item()
    return total / (data.n * 128)


@torch.no_grad()
def baseline_mse(train: PairBatches, test: PairBatches, batch=65536):
    """MSE of predicting the per-feature training mean."""
    s = None
    for x in train.iter(batch):
        s = x.sum(0) if s is None else s + x.sum(0)
    mean = s / train.n
    total = 0.0
    for x in test.iter(batch):
        total += ((x - mean) ** 2).sum().item()
    return total / (test.n * 128)


def run_name(strategy, split, fold, latent):
    return f"{strategy}_{split}_f{fold}_z{latent}"


def run_dir(cfg, strategy, split, fold, latent) -> Path:
    return PROJECT_ROOT / cfg["autoencoder"]["out_dir"] / run_name(strategy, split, fold, latent)


def setup_device(cfg):
    a = cfg["autoencoder"]
    device = a.get("device") or get_device()
    if device == "cpu" and a.get("num_threads"):
        torch.set_num_threads(int(a["num_threads"]))
    return device


# --------------------------------------------------------------------------- training
def train_epoch(model, opt, data: PairBatches, batch, gen, device):
    model.train()
    perm = torch.randperm(data.n, generator=gen).to(device)
    loss_fn = nn.MSELoss()
    total, nb = 0.0, 0
    for x in data.iter(batch, perm):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()
        total += loss.detach()
        nb += 1
    return (total / nb).item()


def train_ae(strategy, split_type, fold, latent, cfg=None, log=print):
    cfg = cfg or load_config()
    a = cfg["autoencoder"]
    device = setup_device(cfg)
    seed = cfg["seed"]
    set_seed(seed)
    out = run_dir(cfg, strategy, split_type, fold, latent)
    out.mkdir(parents=True, exist_ok=True)

    M, G = (np.load(p) for p in _emb_paths(cfg))
    mi, gi, y = load_pairs(cfg, strategy)
    tr, va, te = load_split(strategy, split_type, fold)
    lo, hi = fit_scaler(M, G, mi[tr], gi[tr])
    Ms, Gs = scaled_tables(M, G, lo, hi, device)
    train = PairBatches(Ms, Gs, mi[tr], gi[tr], device)
    val = PairBatches(Ms, Gs, mi[va], gi[va], device)
    test = PairBatches(Ms, Gs, mi[te], gi[te], device)

    model = AutoEncoder(128, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a["lr"])
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, wait, rows = float("inf"), -1, None, 0, []
    t0 = time.time()
    log(f"[{out.name}] device={device} train={train.n:,} val={val.n:,} test={test.n:,}")
    for epoch in range(1, a["max_epochs"] + 1):
        te0 = time.time()
        tr_loss = train_epoch(model, opt, train, a["batch_size"], gen, device)
        v = eval_mse(model, val)
        rows.append({"epoch": epoch, "train_loss": tr_loss, "val_mse": v,
                     "seconds": time.time() - te0})
        improved = v < best
        if improved:
            best, best_epoch, wait = v, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
        log(f"[{out.name}] epoch {epoch:3d} train {tr_loss:.6f} val {v:.6f} "
            f"{rows[-1]['seconds']:.1f}s{' *' if improved else ''}")
        if wait >= a["patience"]:
            break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)

    torch.save(model.encoder.state_dict(), out / "encoder.pt")
    torch.save(model.decoder.state_dict(), out / "decoder.pt")
    np.savez(out / "scaler.npz", min=lo, max=hi)
    pd.DataFrame(rows).to_csv(out / "losses.csv", index=False)
    run_cfg = {"strategy": strategy, "split_type": split_type, "fold": fold, "latent": latent,
               "hidden": [max(112, latent), max(96, latent)], "batch_size": a["batch_size"],
               "lr": a["lr"], "max_epochs": a["max_epochs"], "patience": a["patience"],
               "seed": seed, "device": device,
               "mirna_embedding": cfg["negatives"]["mirna_embedding"],
               "gene_embedding": cfg["negatives"]["gene_embedding"]}
    (out / "config.json").write_text(json.dumps(run_cfg, indent=1))

    metrics = {
        "best_epoch": best_epoch, "epochs_run": len(rows),
        "train_mse": eval_mse(model, train), "val_mse": eval_mse(model, val),
        "test_mse": eval_mse(model, test), "baseline_test_mse": baseline_mse(train, test),
        "train_seconds": train_seconds, "seconds_per_epoch": train_seconds / len(rows),
        "n_train": train.n, "n_val": val.n, "n_test": test.n,
    }
    metrics["mse_ratio_vs_baseline"] = metrics["test_mse"] / metrics["baseline_test_mse"]
    t1 = time.time()
    metrics["probe_test_auc"] = linear_probe(model, train, test, y[tr], y[te], seed)
    metrics["probe_seconds"] = time.time() - t1
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"[{out.name}] done: best epoch {best_epoch}/{len(rows)}, test MSE "
        f"{metrics['test_mse']:.6f} (baseline {metrics['baseline_test_mse']:.6f}), "
        f"probe AUC {metrics['probe_test_auc']:.3f}, {train_seconds / 60:.1f} min")
    return metrics


@torch.no_grad()
def _encode_batches(encoder, data: PairBatches, batch=65536):
    encoder.eval()
    return np.vstack([encoder(x).cpu().numpy() for x in data.iter(batch)])


def linear_probe(model, train, test, y_train, y_test, seed):
    """Logistic regression on z (train rows -> test rows); test ROC-AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ztr, zte = _encode_batches(model.encoder, train), _encode_batches(model.encoder, test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(ztr, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(zte)[:, 1]))


# --------------------------------------------------------------------------- encode helper
def encode(run_dir, mirna_idx, gene_idx, batch=65536, device=None):
    """z for (mirna_idx, gene_idx) pairs using a trained run's encoder and scaler."""
    run_dir = Path(run_dir)
    cfg = load_config()
    rc = json.loads((run_dir / "config.json").read_text())
    device = device or setup_device(cfg)
    e = PROJECT_ROOT / cfg["paths"]["embeddings_dir"]
    M, G = np.load(e / rc["mirna_embedding"]), np.load(e / rc["gene_embedding"])
    sc = np.load(run_dir / "scaler.npz")
    Ms, Gs = scaled_tables(M, G, sc["min"], sc["max"], device)
    model = AutoEncoder(128, rc["latent"]).to(device)
    model.encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location=device))
    data = PairBatches(Ms, Gs, np.asarray(mirna_idx, np.int64), np.asarray(gene_idx, np.int64),
                       device)
    return _encode_batches(model.encoder, data, batch)


# --------------------------------------------------------------------------- benchmark
def benchmark(cfg):
    """One epoch of latent 64 (paper, random_holdout fold 0) on each device option."""
    a = cfg["autoencoder"]
    perf = perf_cores()
    M, G = (np.load(p) for p in _emb_paths(cfg))
    mi, gi, _ = load_pairs(cfg, "paper")
    tr, _, _ = load_split("paper", "random_holdout", 0)
    lo, hi = fit_scaler(M, G, mi[tr], gi[tr])
    options = [("cpu", None), ("cpu", perf)]
    if torch.backends.mps.is_available():
        options.append(("mps", None))
    default_threads = torch.get_num_threads()
    results = []
    for device, threads in options:
        torch.set_num_threads(threads or default_threads)
        set_seed(cfg["seed"])
        Ms, Gs = scaled_tables(M, G, lo, hi, device)
        train = PairBatches(Ms, Gs, mi[tr], gi[tr], device)
        model = AutoEncoder(128, 64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=a["lr"])
        gen = torch.Generator().manual_seed(cfg["seed"])
        t0 = time.time()
        train_epoch(model, opt, train, a["batch_size"], gen, device)
        if device == "mps":
            torch.mps.synchronize()
        sec = time.time() - t0
        label = f"{device}" + (f" threads={threads or default_threads}")
        results.append((device, threads, sec))
        print(f"benchmark {label:<20} 1 epoch ({train.n:,} rows, batch {a['batch_size']}): {sec:.1f}s")
    device, threads, sec = min(results, key=lambda r: r[2])
    bench = {f"{d}{'' if t is None else f'_threads{t}'}": round(s, 1) for d, t, s in results}
    # Edit the three lines in place so the YAML comments elsewhere survive.
    text = CONFIG_PATH.read_text()
    start = text.index("\nautoencoder:\n")
    head, block = text[:start], text[start:]
    for key, val in [("device", device), ("num_threads", threads),
                     ("benchmark", json.dumps(bench))]:
        lines = block.split("\n")
        i = next(j for j, l in enumerate(lines) if l.startswith(f"  {key}:"))
        lines[i] = f"  {key}: {'null' if val is None else val}"
        block = "\n".join(lines)
    CONFIG_PATH.write_text(head + block)
    print(f"chosen: {device} threads={threads} ({sec:.1f}s/epoch); written to configs/default.yaml")
    return results


# --------------------------------------------------------------------------- replication set
def replication_set(cfg):
    import logging
    log_path = PROJECT_ROOT / cfg["paths"]["results_dir"] / "step6_train.log"
    logging.basicConfig(filename=log_path, filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger("ae").info
    runs = [("paper", "random_holdout", 0, z) for z in LATENTS]
    runs += [("paper", "random_kfold", k, 64) for k in range(5)]
    for strategy, split, fold, z in runs:
        out = run_dir(cfg, strategy, split, fold, z)
        if (out / "metrics.json").exists():
            log(f"[{out.name}] already done, skipping")
            continue
        if split == "random_kfold" and fold == 0:
            # random_holdout = fold 0 of random_kfold: reuse that run instead of retraining.
            src = run_dir(cfg, strategy, "random_holdout", 0, z)
            shutil.copytree(src, out)
            meta = json.loads((out / "config.json").read_text())
            meta["reused_from"] = src.name
            meta["split_type"] = split  # same rows: random_holdout == random_kfold fold 0
            (out / "config.json").write_text(json.dumps(meta, indent=1))
            log(f"[{out.name}] reused {src.name}")
            continue
        train_ae(strategy, split, fold, z, cfg, log)
    write_report(cfg)
    log("replication set finished")


# --------------------------------------------------------------------------- report
def write_report(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = PROJECT_ROOT / cfg["autoencoder"]["out_dir"]
    rows = []
    for d in sorted(root.glob("*/metrics.json")):
        m = json.loads(d.read_text())
        c = json.loads((d.parent / "config.json").read_text())
        rows.append({**c, **m, "run": d.parent.name})
    df = pd.DataFrame(rows)
    rep = Report()
    rep("=" * 70)
    rep("Step 6: autoencoder runs (AEmiGAP 4.2.2)")
    rep("=" * 70)
    a = cfg["autoencoder"]
    rep(f"device: {a.get('device')} (threads={a.get('num_threads')}); benchmark s/epoch: "
        f"{a.get('benchmark')}")
    rep(f"input 128 = miRNA {cfg['negatives']['mirna_embedding']} + gene "
        f"{cfg['negatives']['gene_embedding']}; MinMax fit on training rows per fold")
    rep(f"Adam lr {a['lr']}, batch {a['batch_size']}, max {a['max_epochs']} epochs, early "
        f"stopping patience {a['patience']} on val MSE (best weights kept)")
    rep("note: latent=128 is overcomplete (latent = input size), hidden widths 128/128.")
    rep("note: MSE is per element on [0,1]-scaled features; baseline = predict training mean.")
    rep()
    if df.empty:
        rep("no runs yet")
    else:
        df["set"] = np.where(df["split_type"] == "random_holdout", "Table2", "Table3")
        df = df.sort_values(["set", "latent", "fold"])
        hdr = (f"{'set':<7}{'latent':>7}{'fold':>5}{'best':>6}{'run':>5}{'train MSE':>12}"
               f"{'val MSE':>12}{'test MSE':>12}{'base MSE':>12}{'ratio':>7}{'probe AUC':>10}"
               f"{'min':>7}{'s/ep':>7}")
        rep(hdr)
        rep("-" * len(hdr))
        for _, r in df.iterrows():
            reused = " (reused)" if isinstance(r.get("reused_from"), str) else ""
            rep(f"{r['set']:<7}{r['latent']:>7}{r['fold']:>5}{r['best_epoch']:>6}"
                f"{r['epochs_run']:>5}{r['train_mse']:>12.6f}{r['val_mse']:>12.6f}"
                f"{r['test_mse']:>12.6f}{r['baseline_test_mse']:>12.6f}"
                f"{r['mse_ratio_vs_baseline']:>7.3f}{r['probe_test_auc']:>10.3f}"
                f"{r['train_seconds'] / 60:>7.1f}{r['seconds_per_epoch']:>7.1f}{reused}")
        rep("ratio = test MSE / baseline MSE (lower is better; must be well below 1)")
        t3 = df[df["set"] == "Table3"]
        if len(t3) > 1:
            rep(f"Table 3 (5-fold, latent 64): test MSE {t3['test_mse'].mean():.6f} ± "
                f"{t3['test_mse'].std():.6f}; probe AUC {t3['probe_test_auc'].mean():.3f} ± "
                f"{t3['probe_test_auc'].std():.3f}")
        rep("paper reference: training converged after about 20 epochs")

        t2 = df[df["set"] == "Table2"].sort_values("latent")
        if len(t2):
            fig, axes = plt.subplots(1, len(t2), figsize=(3.4 * len(t2), 3.2), squeeze=False,
                                     sharey=True)
            for ax, (_, r) in zip(axes[0], t2.iterrows()):
                L = pd.read_csv(root / r["run"] / "losses.csv")
                ax.plot(L["epoch"], L["train_loss"], color="#2a78d6", lw=2, label="train")
                ax.plot(L["epoch"], L["val_mse"], color="#eb6834", lw=2, label="validation")
                ax.axvline(r["best_epoch"], color="#999", lw=1, ls="--")
                ax.set_yscale("log")
                ax.set_title(f"latent {r['latent']}", fontsize=10, loc="left")
                ax.set_xlabel("epoch", color="#52514e")
                ax.tick_params(colors="#52514e", labelsize=8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            axes[0][0].set_ylabel("MSE (log)", color="#52514e")
            axes[0][0].legend(frameon=False, fontsize=8)
            fig.suptitle("Autoencoder loss curves, paper negatives, random holdout "
                         "(dashed = best epoch)", fontsize=11)
            fig.tight_layout()
            fig.savefig(PROJECT_ROOT / cfg["paths"]["results_dir"] / "step6_loss_curves.png",
                        dpi=140)
    rep.save(PROJECT_ROOT / cfg["paths"]["results_dir"] / "step6_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="paper")
    ap.add_argument("--split", default="random_holdout")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--replication-set", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.benchmark:
        benchmark(cfg)
    elif args.replication_set:
        replication_set(cfg)
    elif args.report:
        write_report(cfg)
    else:
        train_ae(args.strategy, args.split, args.fold, args.latent, cfg)
        write_report(cfg)


if __name__ == "__main__":
    main()
