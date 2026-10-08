"""Step 7: miRNA-gene association classifiers on [m, g, z] (AEmiGAP 4.2.3).

Run from the project root:
  python3 -m src.classifier --benchmark          # CPU vs MPS for lstm + 1-epoch timing per model
  python3 -m src.classifier --strategy paper --split random_holdout --fold 0 \
          --model lstm --input original_plus_latent --latent 64
  python3 -m src.classifier --smoke              # benchmark + step 7 smoke runs + cross test
  python3 -m src.classifier --report

Blocks per pair: m = miRNA vector (64), g = gene vector (64), z = encoder output of the
autoencoder trained on the same strategy/split/fold/latent (computed in memory, never saved).
Each block has its own RobustScaler fit on the TRAINING rows of the fold. (RobustScaler uses
medians/quantiles, which depend on how often an entity occurs, so m and g are fit on the
training rows themselves, not on the distinct training entities; the fitted per-column
transform is then applied to the embedding tables.)
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import RobustScaler
from torch import nn

from src.autoencoder import encode, load_pairs, run_dir as ae_run_dir
from src.parse import Report
from src.splits import load_cross, load_split
from src.utils import PROJECT_ROOT, load_config, set_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
# numpy 2.x + macOS Accelerate: spurious matmul FP warnings (verified harmless in step 3).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
VARIANTS = {"original_plus_latent": ["m", "g", "z"], "original_only": ["m", "g"],
            "latent_only": ["z"]}
NEURAL = ["lstm", "bilstm", "gru", "transformer", "lstm_timesteps", "mlp"]
PAPER_REF = {("lstm", "latent_only"): 0.9231, ("lstm", "original_plus_latent"): 0.9846,
             ("lstm", "original_only"): 0.9804}
PAPER_REF_NOTE = {("lstm", "latent_only"): "Table 2, z=64",
                  ("lstm", "original_plus_latent"): "Table 2, z=64",
                  ("lstm", "original_only"): "Table 11, miGAP"}


# --------------------------------------------------------------------------- models
class RNNClassifier(nn.Module):
    """Each block -> Linear(dim, d); blocks form a token sequence; RNN; final state -> logit."""

    def __init__(self, dims, kind="lstm", d=128, hidden=128, layers=3):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(k, d) for k in dims)
        rnn = {"lstm": nn.LSTM, "bilstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.bi = kind == "bilstm"
        self.rnn = rnn(d, hidden, num_layers=layers, batch_first=True, bidirectional=self.bi)
        self.kind = kind
        self.head = nn.Linear(hidden * (2 if self.bi else 1), 1)

    def forward(self, blocks):
        x = torch.stack([p(b) for p, b in zip(self.proj, blocks)], dim=1)
        _, h = self.rnn(x)
        h = h[0] if self.kind != "gru" else h
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bi else h[-1]
        return self.head(last).squeeze(1)


class TransformerClassifier(nn.Module):
    def __init__(self, dims, d=128, heads=8, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(k, d) for k in dims)
        self.pos = nn.Parameter(torch.zeros(1, len(dims), d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)

    def forward(self, blocks):
        x = torch.stack([p(b) for p, b in zip(self.proj, blocks)], dim=1) + self.pos
        return self.head(self.enc(x).mean(1)).squeeze(1)


class TimestepLSTM(nn.Module):
    """Literal reading of the paper: the concatenated vector as a sequence of scalars."""

    def __init__(self, dims, hidden=128, layers=3):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, blocks):
        _, (h, _) = self.rnn(torch.cat(blocks, dim=1).unsqueeze(2))
        return self.head(h[-1]).squeeze(1)


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sum(dims), 256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, 1))

    def forward(self, blocks):
        return self.net(torch.cat(blocks, dim=1)).squeeze(1)


def build_model(name, dims):
    if name in ("lstm", "bilstm", "gru"):
        return RNNClassifier(dims, name)
    return {"transformer": TransformerClassifier, "lstm_timesteps": TimestepLSTM,
            "mlp": MLP}[name](dims)


# --------------------------------------------------------------------------- data
class Blocks:
    """Scaled block tensors; rows index the strategy's pairs file."""

    def __init__(self, variant, Ms, Gs, Z, mi, gi, device):
        self.names = VARIANTS[variant]
        self.t = {"m": Ms, "g": Gs, "z": Z}
        self.mi = torch.as_tensor(mi, device=device)
        self.gi = torch.as_tensor(gi, device=device)
        self.device = device

    @property
    def dims(self):
        return [self.t[n].shape[1] for n in self.names]

    def get(self, rows):
        out = []
        for n in self.names:
            if n == "m":
                out.append(self.t["m"][self.mi[rows]])
            elif n == "g":
                out.append(self.t["g"][self.gi[rows]])
            else:
                out.append(self.t["z"][rows])
        return out


def _emb(cfg):
    e = PROJECT_ROOT / cfg["paths"]["embeddings_dir"]
    n = cfg["negatives"]
    return np.load(e / n["mirna_embedding"]), np.load(e / n["gene_embedding"])


def fit_scalers(variant, M, G, Z, mi, gi, train_rows):
    sc = {}
    if "m" in VARIANTS[variant]:
        sc["m"] = RobustScaler().fit(M[mi[train_rows]])
    if "g" in VARIANTS[variant]:
        sc["g"] = RobustScaler().fit(G[gi[train_rows]])
    if "z" in VARIANTS[variant]:
        sc["z"] = RobustScaler().fit(Z[train_rows])
    return sc


def apply_scalers(sc, M, G, Z, device):
    def t(x, s):
        return None if x is None or s is None else torch.as_tensor(
            s.transform(x).astype(np.float32), device=device)
    return t(M, sc.get("m")), t(G, sc.get("g")), t(Z, sc.get("z"))


def compute_z(cfg, strategy, split, fold, latent, mi, gi, device):
    ae_dir = ae_run_dir(cfg, strategy, split, fold, latent)
    if not (ae_dir / "encoder.pt").exists():
        raise FileNotFoundError(f"autoencoder run missing: {ae_dir}")
    return encode(ae_dir, mi, gi, device="cpu").astype(np.float32), ae_dir


def setup_device(cfg):
    c = cfg["classifier"]
    if c.get("num_threads"):
        torch.set_num_threads(int(c["num_threads"]))
    return c.get("device") or "cpu"


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, blocks: Blocks, rows: np.ndarray, batch=8192):
    model.eval()
    rows_t = torch.as_tensor(rows, device=blocks.device)
    out = [torch.sigmoid(model(blocks.get(rows_t[s:s + batch]))).cpu()
           for s in range(0, len(rows), batch)]
    return torch.cat(out).numpy()


def train_epoch(model, opt, blocks, train_rows_t, batch, gen, device, max_batches=None):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    perm = train_rows_t[torch.randperm(len(train_rows_t), generator=gen).to(device)]
    total, nb = 0.0, 0
    for s in range(0, len(perm), batch):
        r = perm[s:s + batch]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(blocks.get(r)), blocks.y[r])
        loss.backward()
        opt.step()
        total += loss.detach()
        nb += 1
        if max_batches and nb >= max_batches:
            break
    return (total / nb).item()


def metrics_at(y, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "n": int(len(y)), "threshold": threshold}


def clf_dir(cfg, strategy, split, fold, model, variant, latent):
    return (PROJECT_ROOT / cfg["classifier"]["out_dir"]
            / f"{strategy}_{split}_f{fold}_{model}_{variant}_z{latent}")


def prepare(cfg, strategy, split, fold, variant, latent, device, log):
    M, G = _emb(cfg)
    mi, gi, y = load_pairs(cfg, strategy)
    tr, va, te = load_split(strategy, split, fold)
    Z, ae_dir = (None, None)
    if "z" in VARIANTS[variant]:
        t0 = time.time()
        Z, ae_dir = compute_z(cfg, strategy, split, fold, latent, mi, gi, device)
        log(f"  z from {ae_dir.name}: {Z.shape} in {time.time() - t0:.0f}s")
    sc = fit_scalers(variant, M, G, Z, mi, gi, tr)
    Ms, Gs, Zs = apply_scalers(sc, M, G, Z, device)
    del Z
    blocks = Blocks(variant, Ms, Gs, Zs, mi, gi, device)
    blocks.y = torch.as_tensor(y, dtype=torch.float32, device=device)
    return blocks, y, (tr, va, te), sc, ae_dir


def concat_np(blocks: Blocks, rows):
    out = []
    for s in range(0, len(rows), 262144):
        r = torch.as_tensor(rows[s:s + 262144], device=blocks.device)
        out.append(torch.cat(blocks.get(r), dim=1).cpu().numpy())
    return np.vstack(out)


def train_clf(strategy, split_type, fold, model_name, variant, latent, cfg=None, log=print,
              batch_size=None, max_epochs=None):
    """batch_size / max_epochs override the config (e.g. the capped lstm_timesteps run);
    otherwise max_epochs comes from classifier.max_epochs_by_model, then classifier.max_epochs."""
    cfg = cfg or load_config()
    c = dict(cfg["classifier"])
    c["batch_size"] = batch_size or c["batch_size"]
    c["max_epochs"] = max_epochs or (c.get("max_epochs_by_model") or {}).get(model_name) \
        or c["max_epochs"]
    device = setup_device(cfg)
    seed = cfg["seed"]
    set_seed(seed)
    out = clf_dir(cfg, strategy, split_type, fold, model_name, variant, latent)
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{out.name}] device={device}")
    t_start = time.time()
    blocks, y, (tr, va, te), sc, ae_dir = prepare(cfg, strategy, split_type, fold, variant,
                                                  latent, device, log)
    joblib.dump(sc, out / "scalers.joblib")
    run_cfg = {"strategy": strategy, "split_type": split_type, "fold": fold,
               "model": model_name, "input_variant": variant, "latent": latent,
               "blocks": VARIANTS[variant], "dims": blocks.dims,
               "ae_run": ae_dir.name if ae_dir else None, "seed": seed, "device": device,
               "batch_size": c["batch_size"], "lr": c["lr"], "max_epochs": c["max_epochs"],
               "patience": c["patience"], "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "mirna_embedding": cfg["negatives"]["mirna_embedding"],
               "gene_embedding": cfg["negatives"]["gene_embedding"]}
    rows_log, best_epoch = [], None

    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(concat_np(blocks, tr), y[tr])
        joblib.dump(clf, out / "model.joblib")
        val_auc = roc_auc_score(y[va], clf.predict_proba(concat_np(blocks, va))[:, 1])
        score = clf.predict_proba(concat_np(blocks, te))[:, 1]
        rows_log.append({"epoch": 1, "train_loss": float("nan"), "val_auc": val_auc,
                         "seconds": time.time() - t0})
        best_epoch = 1
    else:
        model = build_model(model_name, blocks.dims).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        gen = torch.Generator().manual_seed(seed)
        tr_t = torch.as_tensor(tr, device=device)
        best, best_state, wait = -1.0, None, 0
        for epoch in range(1, c["max_epochs"] + 1):
            t0 = time.time()
            loss = train_epoch(model, opt, blocks, tr_t, c["batch_size"], gen, device)
            t_train = time.time() - t0
            val_auc = roc_auc_score(y[va], predict(model, blocks, va))
            rows_log.append({"epoch": epoch, "train_loss": loss, "val_auc": val_auc,
                             "seconds": t_train})
            improved = val_auc > best
            if improved:
                best, best_epoch, wait = val_auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            log(f"[{out.name}] epoch {epoch:2d} loss {loss:.4f} val AUC {val_auc:.4f} "
                f"{t_train:.0f}s{' *' if improved else ''}")
            if wait >= c["patience"]:
                break
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), out / "model.pt")
        score = predict(model, blocks, te)

    m = metrics_at(y[te], score)
    m.update(best_epoch=best_epoch, epochs_run=len(rows_log),
             best_val_auc=float(max(r["val_auc"] for r in rows_log)),
             seconds_total=time.time() - t_start,
             seconds_per_epoch=float(np.mean([r["seconds"] for r in rows_log])))
    pd.DataFrame(rows_log).to_csv(out / "epochs.csv", index=False)
    pd.DataFrame({"row": te.astype(np.int64), "label": y[te].astype(np.int8),
                  "score": score.astype(np.float32)}).to_parquet(out / "predictions.parquet",
                                                                 index=False)
    (out / "config.json").write_text(json.dumps(run_cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    log(f"[{out.name}] test ROC-AUC {m['roc_auc']:.4f} PR-AUC {m['pr_auc']:.4f} "
        f"acc {m['accuracy']:.4f} F1 {m['f1']:.4f} (best epoch {best_epoch}/{len(rows_log)}, "
        f"{m['seconds_total'] / 60:.1f} min)")
    return m


def predict_cross(run_dir, test_strategy, split_type, fold, log=print):
    """Apply a trained run (its scalers, its AE) to load_cross test rows of test_strategy."""
    run_dir = Path(run_dir)
    cfg = load_config()
    rc = json.loads((run_dir / "config.json").read_text())
    if (split_type, fold) != (rc["split_type"], rc["fold"]):
        raise ValueError("predict_cross must use the run's own split and fold "
                         f"({rc['split_type']}, {rc['fold']}) to avoid leakage")
    device = setup_device(cfg)
    _, _, te = load_cross(rc["strategy"], test_strategy, split_type, fold)
    mi, gi, y = load_pairs(cfg, test_strategy)
    M, G = _emb(cfg)
    sc = joblib.load(run_dir / "scalers.joblib")
    Z = None
    if "z" in rc["blocks"]:
        ae_dir = PROJECT_ROOT / cfg["autoencoder"]["out_dir"] / rc["ae_run"]
        Z = np.zeros((len(mi), rc["latent"]), np.float32)
        Z[te] = encode(ae_dir, mi[te], gi[te], device="cpu")
    Ms, Gs, Zs = apply_scalers(sc, M, G, Z, device)
    blocks = Blocks(rc["input_variant"], Ms, Gs, Zs, mi, gi, device)
    if rc["model"] == "logreg":
        clf = joblib.load(run_dir / "model.joblib")
        score = clf.predict_proba(concat_np(blocks, te))[:, 1]
    else:
        model = build_model(rc["model"], rc["dims"]).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        score = predict(model, blocks, te)
    m = metrics_at(y[te], score)
    m.update(train_strategy=rc["strategy"], test_strategy=test_strategy,
             split_type=split_type, fold=fold)
    out = run_dir / f"cross_{test_strategy}"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    pd.DataFrame({"row": te.astype(np.int64), "label": y[te].astype(np.int8),
                  "score": score.astype(np.float32)}).to_parquet(out / "predictions.parquet",
                                                                 index=False)
    log(f"[{run_dir.name} -> {test_strategy}] ROC-AUC {m['roc_auc']:.4f} PR-AUC {m['pr_auc']:.4f} "
        f"acc {m['accuracy']:.4f} F1 {m['f1']:.4f} (n={m['n']:,})")
    return m


# --------------------------------------------------------------------------- benchmark
def benchmark(cfg, log=print):
    """CPU vs MPS for one lstm epoch, then one epoch per model on the chosen device.
    Setting: paper, random_holdout fold 0, original_plus_latent, latent 64."""
    from src.embed import perf_cores
    c = cfg["classifier"]
    perf = perf_cores()
    res = {}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    cache = {}

    def one_epoch(model_name, device, max_batches=None):
        torch.set_num_threads(perf)
        if device not in cache:
            cache[device] = prepare(cfg, "paper", "random_holdout", 0, "original_plus_latent",
                                    64, device, log)
        blocks, _, (tr, _, _), _, _ = cache[device]
        set_seed(cfg["seed"])
        model = build_model(model_name, blocks.dims).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        gen = torch.Generator().manual_seed(cfg["seed"])
        tr_t = torch.as_tensor(tr, device=device)
        t0 = time.time()
        train_epoch(model, opt, blocks, tr_t, c["batch_size"], gen, device, max_batches)
        if device == "mps":
            torch.mps.synchronize()
        sec = time.time() - t0
        n_batches = int(np.ceil(len(tr) / c["batch_size"]))
        if max_batches:
            sec = sec * n_batches / max_batches  # extrapolated to a full epoch
        return sec

    for dev in devices:
        res[f"lstm_{dev}"] = one_epoch("lstm", dev)
        log(f"benchmark lstm on {dev}: {res[f'lstm_{dev}']:.1f}s/epoch")
    best_dev = min(devices, key=lambda d: res[f"lstm_{d}"])
    cache = {k: v for k, v in cache.items() if k == best_dev}
    for name in ["bilstm", "gru", "transformer", "mlp"]:
        res[f"{name}_{best_dev}"] = one_epoch(name, best_dev)
        log(f"benchmark {name} on {best_dev}: {res[f'{name}_{best_dev}']:.1f}s/epoch")
    # lstm_timesteps: 192 steps per sample; time 300 batches and extrapolate
    res[f"lstm_timesteps_{best_dev}"] = one_epoch("lstm_timesteps", best_dev, max_batches=300)
    log(f"benchmark lstm_timesteps on {best_dev}: {res[f'lstm_timesteps_{best_dev}']:.0f}s/epoch "
        "(extrapolated from 300 batches)")

    text = CONFIG_PATH.read_text()
    start = text.index("\nclassifier:\n")
    head, block = text[:start], text[start:]
    for key, val in [("device", best_dev), ("num_threads", perf),
                     ("benchmark", json.dumps({k: round(v, 1) for k, v in res.items()}))]:
        lines = block.split("\n")
        i = next(j for j, l in enumerate(lines) if l.startswith(f"  {key}:"))
        lines[i] = f"  {key}: {val}"
        block = "\n".join(lines)
    CONFIG_PATH.write_text(head + block)
    log(f"chosen device: {best_dev} (threads={perf}); written to configs/default.yaml")
    return res


# --------------------------------------------------------------------------- smoke + report
SMOKE = [("lstm", "original_plus_latent"), ("lstm", "original_only"), ("lstm", "latent_only"),
         ("mlp", "original_plus_latent"), ("logreg", "original_plus_latent")]


def smoke(cfg):
    log_path = PROJECT_ROOT / cfg["paths"]["results_dir"] / "step7_train.log"
    logging.basicConfig(filename=log_path, filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger("clf").info
    if not cfg["classifier"].get("benchmark"):
        benchmark(cfg, log)
        cfg = load_config()
    for model, variant in SMOKE:
        d = clf_dir(cfg, "paper", "random_holdout", 0, model, variant, 64)
        if (d / "metrics.json").exists():
            log(f"[{d.name}] already done, skipping")
            continue
        train_clf("paper", "random_holdout", 0, model, variant, 64, cfg, log)
    run1 = clf_dir(cfg, "paper", "random_holdout", 0, "lstm", "original_plus_latent", 64)
    predict_cross(run1, "degree_aware", "random_holdout", 0, log)
    write_report(cfg)
    log("step 7 smoke set finished")


def write_report(cfg):
    root = PROJECT_ROOT / cfg["classifier"]["out_dir"]
    rep = Report()
    c = cfg["classifier"]
    rep("=" * 70)
    rep("Step 7: classifiers (paper negatives, random holdout fold 0, latent 64)")
    rep("=" * 70)
    rep(f"device: {c.get('device')} (threads={c.get('num_threads')}); Adam lr {c['lr']}, "
        f"batch {c['batch_size']}, max {c['max_epochs']} epochs, early stopping on val ROC-AUC "
        f"(patience {c['patience']}, best weights kept)")
    rep("note: the paper's '5,000,000 iterations' is replaced by early stopping on validation "
        "ROC-AUC.")
    rep("blocks: m/g = Doc2Vec vectors, z = encoder of the AE trained on the same "
        "strategy/split/fold; one RobustScaler per block, fit on training rows.")
    rep()
    rows = []
    for d in sorted(root.glob("*/metrics.json")):
        m = json.loads(d.read_text())
        rc = json.loads((d.parent / "config.json").read_text())
        rows.append({**rc, **m, "run": d.parent.name})
    hdr = (f"{'model':<8}{'input':<22}{'ROC-AUC':>8}{'paper':>8}{'PR-AUC':>8}{'acc':>8}"
           f"{'F1':>8}{'prec':>8}{'recall':>8}{'best':>6}{'ep':>4}{'min':>7}{'s/ep':>7}")
    rep(hdr)
    rep("-" * len(hdr))
    order = {k: i for i, k in enumerate(SMOKE)}
    for r in sorted(rows, key=lambda r: order.get((r["model"], r["input_variant"]), 99)):
        ref = PAPER_REF.get((r["model"], r["input_variant"]))
        rep(f"{r['model']:<8}{r['input_variant']:<22}{r['roc_auc']:>8.4f}"
            f"{(f'{ref:.4f}' if ref else '-'):>8}{r['pr_auc']:>8.4f}{r['accuracy']:>8.4f}"
            f"{r['f1']:>8.4f}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['best_epoch']:>6}"
            f"{r['epochs_run']:>4}{r['seconds_total'] / 60:>7.1f}{r['seconds_per_epoch']:>7.1f}")
    rep("paper reference: " + "; ".join(f"{m} {v}: {PAPER_REF[(m, v)]} ({PAPER_REF_NOTE[(m, v)]})"
                                        for m, v in PAPER_REF))
    rep("confusion matrices (test, threshold 0.5):")
    for r in rows:
        cm = r["confusion"]
        rep(f"  {r['model']:<7} {r['input_variant']:<21} TN={cm['tn']:,} FP={cm['fp']:,} "
            f"FN={cm['fn']:,} TP={cm['tp']:,}")
    rep()
    rep("Cross-strategy evaluation (trained on paper negatives):")
    for d in sorted(root.glob("*/cross_*/metrics.json")):
        m = json.loads(d.read_text())
        rep(f"  {d.parent.parent.name} -> test negatives from {m['test_strategy']} "
            f"({m['split_type']} fold {m['fold']}, n={m['n']:,}): ROC-AUC {m['roc_auc']:.4f}, "
            f"PR-AUC {m['pr_auc']:.4f}, acc {m['accuracy']:.4f}, F1 {m['f1']:.4f}, "
            f"precision {m['precision']:.4f}, recall {m['recall']:.4f}")
    rep()
    rep("Step 4 shortcut baselines, paper strategy (logistic regression, test ROC-AUC):")
    rep("  3 distances only 0.859; train-split degree only 0.894; log gene length only 0.711")
    rep()
    rep("Epoch timings (benchmark, one epoch, paper random_holdout f0, original_plus_latent):")
    for k, v in (c.get("benchmark") or {}).items():
        extra = "  (extrapolated from 300 batches; not trained)" if "timesteps" in k else ""
        rep(f"  {k:<26}{v:>8.1f} s/epoch{extra}")
    rep.save(PROJECT_ROOT / cfg["paths"]["results_dir"] / "step7_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="paper")
    ap.add_argument("--split", default="random_holdout")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="lstm", choices=NEURAL + ["logreg"])
    ap.add_argument("--input", default="original_plus_latent", choices=list(VARIANTS))
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.benchmark:
        benchmark(cfg)
    elif args.smoke:
        smoke(cfg)
    elif args.report:
        write_report(cfg)
    else:
        train_clf(args.strategy, args.split, args.fold, args.model, args.input, args.latent, cfg)
        write_report(cfg)


if __name__ == "__main__":
    main()
