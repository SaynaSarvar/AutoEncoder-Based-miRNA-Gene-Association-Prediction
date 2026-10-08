"""Step 3: embeddings for every miRNA and gene in the index files.

Run from the project root:
  python3 -m src.embed build-corpus   # data/embeddings/corpus_genomic.txt + corpus_docs.csv
  python3 -m src.embed kmer           # 3-mer frequency baseline
  python3 -m src.embed pilot          # Doc2Vec on ~2% of tokens, projects full-run cost
  python3 -m src.embed train          # full Doc2Vec run (long; run with nohup), then extract
  python3 -m src.embed extract        # gene / miRNA vectors from a trained model
  python3 -m src.embed check          # sanity checks on every embedding set present

Tokenisation (Protein2Vec style): per sequence, 3 "sentences" (reading-frame offsets 0/1/2) of
non-overlapping 3-mers, 3-mers containing N skipped, each sentence split into chunks of at
most 10,000 tokens (gensim's per-document limit). One chunk = one corpus line = one doc tag.
"""
import argparse
import json
import logging
import os
import subprocess
import time
import zlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

from src.parse import Report
from src.utils import PROJECT_ROOT, load_config, set_seed

KMERS = ["".join(k) for k in product("ACGT", repeat=3)]  # index = 16*a + 4*b + c
CODE = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    CODE[_c] = _i


def paths(cfg):
    e = PROJECT_ROOT / cfg["paths"]["embeddings_dir"]
    i = PROJECT_ROOT / cfg["paths"]["interim_dir"]
    pr = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    r = PROJECT_ROOT / cfg["paths"]["results_dir"]
    e.mkdir(parents=True, exist_ok=True)
    return {
        "corpus": e / "corpus_genomic.txt", "docs": e / "corpus_docs.csv",
        "pilot_corpus": e / "corpus_pilot.txt", "model": e / "doc2vec_genomic.model",
        "gene_d2v": e / "gene_d2v.npy", "mirna_trained": e / "mirna_d2v_trained.npy",
        "mirna_inferred": e / "mirna_d2v_inferred.npy",
        "gene_kmer": e / "gene_kmer.npy", "mirna_kmer": e / "mirna_kmer.npy",
        "genomic_fa": i / "genes_genomic.fa", "genes_csv": i / "genes.csv",
        "mirna_index": pr / "mirna_index.csv", "gene_index": pr / "gene_index.csv",
        "positives": pr / "positives.csv",
        "corpus_report": r / "step3_corpus_report.txt", "pilot_report": r / "step3_pilot_report.txt",
        "report": r / "step3_report.txt", "train_log": r / "step3_train.log",
        "pca_png": r / "step3_gene_pca_length.png", "train_times": r / "step3_train_times.json",
    }


def human_size(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def iter_entities(P):
    """Yield (entity_type, entity_idx, uppercase DNA string): miRNAs first, then genes by gene_idx."""
    mi = pd.read_csv(P["mirna_index"])
    for idx, seq in zip(mi["mirna_idx"], mi["sequence_dna"]):
        yield "mirna", int(idx), seq.upper()
    gi = pd.read_csv(P["gene_index"])
    fa = Fasta(str(P["genomic_fa"]), as_raw=True, sequence_always_upper=True)
    for idx, gid in zip(gi["gene_idx"], gi["gene_id"]):
        yield "gene", int(idx), fa[gid][:]


# --------------------------------------------------------------------------- corpus
def frame_chunks(seq: str, frame: int, max_tokens: int):
    """Non-overlapping 3-mers from offset `frame`, N-containing 3-mers dropped, as
    space-separated byte strings of at most max_tokens tokens."""
    a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)[frame:]
    n = len(a) // 3
    if n == 0:
        return []
    tri = a[:n * 3].reshape(n, 3)
    tri = tri[(CODE[tri] != 255).all(axis=1)]          # skip 3-mers with N (or any non-ACGT)
    if len(tri) == 0:
        return []
    rows = np.empty((len(tri), 4), dtype=np.uint8)
    rows[:, :3], rows[:, 3] = tri, ord(" ")
    out = []
    for s in range(0, len(rows), max_tokens):
        block = rows[s:s + max_tokens]
        out.append((block.tobytes()[:-1], len(block)))  # drop trailing space
    return out


def build_corpus(cfg):
    P = paths(cfg)
    max_tokens = cfg["embed"]["max_doc_tokens"]
    docs, line = [], 0
    with open(P["corpus"], "wb") as fh:
        for etype, idx, seq in tqdm(iter_entities(P), desc="corpus", mininterval=2):
            for frame in range(3):
                for chunk_no, (text, ntok) in enumerate(frame_chunks(seq, frame, max_tokens)):
                    fh.write(text + b"\n")
                    docs.append((line, etype, idx, frame, chunk_no, ntok))
                    line += 1
    d = pd.DataFrame(docs, columns=["line", "entity_type", "entity_idx", "frame",
                                    "chunk_number", "n_tokens"])
    d.to_csv(P["docs"], index=False)

    rep = Report()
    rep("=" * 70)
    rep("Step 3 corpus (genomic, 3-mer, 3 frames, chunks <= %d tokens)" % max_tokens)
    rep("=" * 70)
    for t, g in d.groupby("entity_type", sort=False):
        rep(f"{t:<6} entities={g['entity_idx'].nunique():>6} documents={len(g):>9,} "
            f"tokens={g['n_tokens'].sum():>15,}")
    rep(f"total  documents={len(d):,} tokens={d['n_tokens'].sum():,}")
    rep(f"corpus file: {P['corpus'].relative_to(PROJECT_ROOT)}  {human_size(P['corpus'].stat().st_size)}")
    genes = d[d["entity_type"] == "gene"]
    per_gene = genes.groupby("entity_idx").size()
    gi = pd.read_csv(P["gene_index"])
    big = per_gene.idxmax()
    rep(f"largest gene: {gi.loc[big, 'gene_id']} ({gi.loc[big, 'gene_symbol']}) "
        f"-> {per_gene.max()} chunks over 3 frames")
    rep(f"chunks per gene: median={per_gene.median():.0f} p99={per_gene.quantile(.99):.0f}")
    rep(f"documents shorter than 10 tokens: {(d['n_tokens'] < 10).sum()} "
        f"(miRNA frames have ~7 tokens)")
    rep.save(P["corpus_report"])


# --------------------------------------------------------------------------- k-mer baseline
def kmer_freq(seq: str) -> np.ndarray:
    """Overlapping 3-mer counts / valid 3-mers (64-dim, KMERS order)."""
    c = CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    if len(c) < 3:
        return np.zeros(64, dtype=np.float32)
    a, b, d = c[:-2], c[1:-1], c[2:]
    ok = (a != 255) & (b != 255) & (d != 255)
    idx = 16 * a[ok].astype(np.int64) + 4 * b[ok] + d[ok]
    cnt = np.bincount(idx, minlength=64).astype(np.float64)
    return (cnt / max(cnt.sum(), 1)).astype(np.float32)


def build_kmer(cfg):
    P = paths(cfg)
    n_mi = len(pd.read_csv(P["mirna_index"]))
    n_g = len(pd.read_csv(P["gene_index"]))
    mi, ge = np.zeros((n_mi, 64), np.float32), np.zeros((n_g, 64), np.float32)
    for etype, idx, seq in tqdm(iter_entities(P), total=n_mi + n_g, desc="kmer", mininterval=2):
        (mi if etype == "mirna" else ge)[idx] = kmer_freq(seq)
    np.save(P["mirna_kmer"], mi)
    np.save(P["gene_kmer"], ge)
    print(f"Saved {P['mirna_kmer'].name} {mi.shape}, {P['gene_kmer'].name} {ge.shape}")


# --------------------------------------------------------------------------- doc2vec
def perf_cores() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip())
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def d2v_params(cfg, epochs):
    e = cfg["embed"]
    return dict(vector_size=e["vector_size"], window=e["window"], min_count=e["min_count"],
                dm=e["dm"], sample=e["sample"], epochs=epochs,
                workers=e.get("workers") or perf_cores(), seed=cfg["seed"])


def epoch_timer(log):
    from gensim.models.callbacks import CallbackAny2Vec

    class EpochTimer(CallbackAny2Vec):
        def __init__(self):
            self.times, self._t = [], None

        def on_epoch_begin(self, model):
            self._t = time.time()

        def on_epoch_end(self, model):
            self.times.append(time.time() - self._t)
            log(f"epoch {len(self.times)}/{model.epochs} done in {self.times[-1]:.1f}s")

    return EpochTimer()


def train_doc2vec(corpus_file: Path, n_docs: int, total_words: int, params: dict, log,
                  callbacks=()):
    from gensim.models.doc2vec import Doc2Vec
    model = Doc2Vec(**params)
    t0 = time.time()
    model.build_vocab(corpus_file=str(corpus_file))
    t_vocab = time.time() - t0
    log(f"vocab built in {t_vocab:.1f}s: {len(model.wv)} words, {len(model.dv)} doc tags "
        f"(expected {n_docs}), corpus words {model.corpus_total_words:,} (expected {total_words:,})")
    t0 = time.time()
    model.train(corpus_file=str(corpus_file), total_examples=model.corpus_count,
                total_words=model.corpus_total_words, epochs=params["epochs"],
                callbacks=list(callbacks))
    t_train = time.time() - t0
    return model, t_vocab, t_train


def pilot(cfg):
    P = paths(cfg)
    d = pd.read_csv(P["docs"])
    total = int(d["n_tokens"].sum())
    frac = cfg["embed"]["pilot_token_fraction"]
    mi = d[d["entity_type"] == "mirna"]
    ge = d[d["entity_type"] == "gene"].sample(frac=1.0, random_state=cfg["seed"])
    budget = frac * total - mi["n_tokens"].sum()
    ge = ge[ge["n_tokens"].cumsum() <= budget]
    keep = set(mi["line"]) | set(ge["line"])
    with open(P["corpus"], "rb") as src, open(P["pilot_corpus"], "wb") as dst:
        for i, line in enumerate(src):
            if i in keep:
                dst.write(line)
    sub = d[d["line"].isin(keep)]
    pilot_tokens = int(sub["n_tokens"].sum())

    rep = Report()
    params = d2v_params(cfg, cfg["embed"]["pilot_epochs"])
    rep("=" * 70)
    rep("Step 3 pilot: Doc2Vec on a token subsample")
    rep("=" * 70)
    rep(f"params: {params}")
    rep(f"pilot corpus: {len(sub):,} docs ({len(mi):,} miRNA + {len(ge):,} gene chunks), "
        f"{pilot_tokens:,} tokens = {pilot_tokens / total:.2%} of {total:,}; "
        f"{human_size(P['pilot_corpus'].stat().st_size)}")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.WARNING)
    model, t_vocab, t_train = train_doc2vec(P["pilot_corpus"], len(sub), pilot_tokens, params, rep)
    words = pilot_tokens * params["epochs"]
    wps = words / t_train
    per_epoch = total / wps
    full_epochs = cfg["embed"]["epochs"]
    rep(f"training: {params['epochs']} epochs in {t_train:.1f}s -> {wps:,.0f} words/s")
    rep(f"projected full corpus: {per_epoch / 60:.1f} min/epoch; "
        f"{full_epochs} epochs = {full_epochs * per_epoch / 3600:.1f} h "
        f"(+ vocab scan ~{t_vocab * total / pilot_tokens / 60:.1f} min)")
    rep.save(P["pilot_report"])
    P["pilot_corpus"].unlink()


def train(cfg):
    P = paths(cfg)
    set_seed(cfg["seed"])
    d = pd.read_csv(P["docs"])
    params = d2v_params(cfg, cfg["embed"]["epochs"])
    logging.basicConfig(filename=P["train_log"], filemode="a", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("embed").info
    log(f"params: {params}; docs={len(d)} tokens={d['n_tokens'].sum()}")
    log(f"sample={params['sample']}: {cfg['embed']['sample_reason']}")
    log("NOTE: multi-worker gensim training is not bit-reproducible even with a fixed seed.")
    timer = epoch_timer(log)
    model, t_vocab, t_train = train_doc2vec(P["corpus"], len(d), int(d["n_tokens"].sum()),
                                            params, log, callbacks=[timer])
    log(f"training finished in {t_train / 3600:.2f} h (vocab {t_vocab:.0f}s)")
    model.save(str(P["model"]))
    log(f"saved {P['model']}")
    P["train_times"].write_text(json.dumps({
        "params": params, "vocab_seconds": t_vocab, "train_seconds": t_train,
        "epoch_seconds": timer.times, "tokens": int(d["n_tokens"].sum())}, indent=1))
    extract(cfg)
    log("extraction finished")
    check(cfg)
    log("check finished")


def extract(cfg):
    from gensim.models.doc2vec import Doc2Vec
    P = paths(cfg)
    model = Doc2Vec.load(str(P["model"]))
    d = pd.read_csv(P["docs"])
    if len(model.dv) != len(d):
        raise RuntimeError(f"model has {len(model.dv)} doc tags, corpus_docs has {len(d)}")
    dv = model.dv.vectors  # corpus_file mode: tag = line number
    n_mi = len(pd.read_csv(P["mirna_index"]))
    n_g = len(pd.read_csv(P["gene_index"]))

    g = d[d["entity_type"] == "gene"]
    w = g["n_tokens"].to_numpy(np.float64)
    gene = np.zeros((n_g, dv.shape[1]))
    np.add.at(gene, g["entity_idx"].to_numpy(), dv[g["line"].to_numpy()] * w[:, None])
    gene /= np.bincount(g["entity_idx"], weights=w, minlength=n_g)[:, None]
    np.save(P["gene_d2v"], gene.astype(np.float32))

    m = d[d["entity_type"] == "mirna"]
    trained = np.zeros((n_mi, dv.shape[1]))
    np.add.at(trained, m["entity_idx"].to_numpy(), dv[m["line"].to_numpy()])
    trained /= np.bincount(m["entity_idx"], minlength=n_mi)[:, None]
    np.save(P["mirna_trained"], trained.astype(np.float32))

    e = cfg["embed"]["infer"]
    mi = pd.read_csv(P["mirna_index"])
    inferred = np.zeros_like(trained)
    cache = {}  # identical sequences -> identical vectors
    for idx, seq in tqdm(zip(mi["mirna_idx"], mi["sequence_dna"]), total=n_mi, desc="infer"):
        seq = seq.upper()
        if seq not in cache:
            vecs = []
            for frame in range(3):
                toks = [t for text, _ in frame_chunks(seq, frame, 10**9)
                        for t in text.decode().split()]
                # Seed from the sequence itself (not the index). gensim's Cython code calls
                # model.random.randint -> needs a legacy RandomState.
                model.random = np.random.RandomState(
                    (cfg["seed"] + zlib.crc32(f"{seq}|{frame}".encode())) % 2**32)
                vecs.append(model.infer_vector(toks, epochs=e["epochs"], alpha=e["alpha"],
                                               min_alpha=e["min_alpha"]))
            cache[seq] = np.mean(vecs, axis=0)
        inferred[idx] = cache[seq]
    np.save(P["mirna_inferred"], inferred.astype(np.float32))
    print(f"Saved gene_d2v {gene.shape}, mirna trained/inferred {trained.shape}")


# --------------------------------------------------------------------------- checks
def cos_matrix(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(n == 0, 1, n)
    return Xn @ Xn.T


def mean_pairwise_cos(X):
    if len(X) < 2:
        return float("nan")
    C = cos_matrix(X)
    iu = np.triu_indices(len(X), 1)
    return float(C[iu].mean())


def random_pair_cos(X, n_pairs, rng):
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    a, b = X[i[keep]], X[j[keep]]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float((num / np.where(den == 0, 1, den)).mean())


def leakage(X, target, label, rep, seed):
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    norm_rho = spearmanr(np.linalg.norm(X, axis=1), target).statistic
    pcs = PCA(n_components=5, random_state=seed).fit_transform(X)
    pc_rho = [spearmanr(pcs[:, k], target).statistic for k in range(5)]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    r2 = cross_val_score(model, X, target, cv=KFold(5, shuffle=True, random_state=seed),
                         scoring="r2")
    rep(f"    {label:<26} rho(norm)={norm_rho:+.3f}  rho(PC1..5)="
        + " ".join(f"{r:+.2f}" for r in pc_rho)
        + f"  ridge 5-fold R2={r2.mean():.3f}±{r2.std():.3f}")
    return {"r2": float(r2.mean()), "pc1_rho": float(pc_rho[0]), "norm_rho": float(norm_rho)}


def paralog_clusters(gi, rng):
    sym = gi["gene_symbol"].astype(str)
    clusters = {
        "HOXA1-13": sym.str.fullmatch(r"HOXA([1-9]|1[0-3])"),
        "HOXB*": sym.str.fullmatch(r"HOXB\d+"),
        "KRT*": sym.str.fullmatch(r"KRT\d+[A-Z]?"),
        "OR*": sym.str.fullmatch(r"OR\d+[A-Z]+\d+[A-Z]?\d*"),
    }
    out = {k: np.flatnonzero(v.to_numpy()) for k, v in clusters.items()}
    znf = np.flatnonzero(sym.str.fullmatch(r"ZNF\d+").to_numpy())
    out["ZNF* (100)"] = rng.choice(znf, size=min(100, len(znf)), replace=False)
    return out


def pair_cos(X, i, j):
    a, b = X[i], X[j]
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(((a * b).sum(1) / np.where(den == 0, 1, den)).mean())


def random_pairs(n, n_pairs, rng):
    i, j = rng.integers(0, n, n_pairs * 2), rng.integers(0, n, n_pairs * 2)
    keep = i != j
    return i[keep][:n_pairs], j[keep][:n_pairs]


def comparison_table(metrics: dict, rep: Report):
    names = list(metrics)
    rows = [
        ("miRNA: let-7 raw cos", "let7_raw", ".3f"),
        ("miRNA: let-7 centred cos", "let7_c", ".3f"),
        ("miRNA: miR-200 raw cos", "mir200_raw", ".3f"),
        ("miRNA: miR-200 centred cos", "mir200_c", ".3f"),
        ("miRNA: random raw cos", "rand_raw", ".3f"),
        ("miRNA: random centred cos", "rand_c", ".3f"),
        ("miRNA: family margin (centred)", "fam_margin", ".3f"),
        ("shared-seq: max dist", "shared_max", ".3g"),
        ("shared-seq: max / median random", "shared_ratio", ".3f"),
    ] + [(f"gene: {c} centred cos", f"par_{c}", ".3f") for c in
         ["HOXA1-13", "HOXB*", "KRT*", "OR*", "ZNF* (100)", "random"]] + [
        ("leak: ridge R2 log length", "len_r2", ".3f"),
        ("leak: ridge R2 log degree", "deg_r2", ".3f"),
        ("leak: Spearman PC1 vs log len", "len_pc1", "+.3f"),
    ]
    w = 33
    rep(f"{'metric':<{w}}" + "".join(f"{n:>15}" for n in names))
    rep("-" * (w + 15 * len(names)))
    for label, key, fmt in rows:
        rep(f"{label:<{w}}" + "".join(
            f"{format(metrics[n][key], fmt) if key in metrics[n] else '-':>15}" for n in names))
    rep("family margin = mean over let-7/miR-200 of (centred within-family cos - centred "
        "random cos)")
    rep("d2v_trained and d2v_inferred share the same gene matrix, so gene rows are identical.")


def recommend(metrics: dict, rep: Report):
    t, i = metrics.get("d2v_trained"), metrics.get("d2v_inferred")
    if not (t and i):
        return
    rep("Recommendation (miRNA vectors downstream):")
    for n, m in (("trained", t), ("inferred", i)):
        rep(f"  {n:<9} family margin={m['fam_margin']:.3f}  "
            f"shared-seq max/median-random={m['shared_ratio']:.3f}")
    better_family = "trained" if t["fam_margin"] > i["fam_margin"] else "inferred"
    better_shared = "trained" if t["shared_ratio"] < i["shared_ratio"] else "inferred"
    if better_family == better_shared:
        rep(f"  -> use '{better_family}': better on both family structure and "
            "shared-sequence consistency.")
    else:
        rep(f"  -> mixed: '{better_family}' separates families better, '{better_shared}' is "
            "more consistent for identical sequences; see discussion.")


def check(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    import warnings
    # numpy 2.x + macOS Accelerate raises spurious divide/overflow flags in matmul (results
    # verified identical to einsum); silence only those.
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
    P = paths(cfg)
    rng = np.random.default_rng(cfg["seed"])
    mi = pd.read_csv(P["mirna_index"])
    gi = pd.read_csv(P["gene_index"])
    genes = pd.read_csv(P["genes_csv"]).set_index("gene_id")
    log_len = np.log10(gi["gene_id"].map(genes["genomic_length"]).to_numpy(float))
    degree = (pd.read_csv(P["positives"], usecols=["gene_idx"])["gene_idx"]
              .value_counts().reindex(gi["gene_idx"], fill_value=0).to_numpy(float))
    log_deg = np.log10(1 + degree)

    sets = {"kmer": (P["mirna_kmer"], P["gene_kmer"]),
            "d2v_trained": (P["mirna_trained"], P["gene_d2v"]),
            "d2v_inferred": (P["mirna_inferred"], P["gene_d2v"])}
    sets = {k: v for k, v in sets.items() if v[0].exists() and v[1].exists()}

    rep = Report()
    rep("=" * 70)
    rep("Step 3 embedding checks")
    rep("=" * 70)
    rep(f"sets present: {', '.join(sets) or 'none'}")
    rep(f"index sizes: {len(mi)} miRNAs, {len(gi)} genes")
    if P["train_times"].exists():
        tt = json.loads(P["train_times"].read_text())
        ep = np.array(tt["epoch_seconds"])
        rep(f"Doc2Vec params: {tt['params']}")
        rep(f"  sample={tt['params'].get('sample')}: {cfg['embed']['sample_reason']}")
        rep(f"  training: {len(ep)} epochs, {ep.mean() / 60:.2f} min/epoch "
            f"(min {ep.min() / 60:.2f}, max {ep.max() / 60:.2f}); total "
            f"{tt['train_seconds'] / 3600:.2f} h + vocab scan {tt['vocab_seconds'] / 60:.1f} min; "
            f"{tt['tokens'] * len(ep) / tt['train_seconds']:,.0f} words/s")
        rep("  note: multi-worker gensim Doc2Vec is not bit-reproducible even with seed=42")
        if P["train_log"].exists():
            n_dot = P["train_log"].read_text(errors="replace").count("Exception ignored in: "
                                                                     "'gensim.models.word2vec_inner.our_dot_float'")
            rep(f"  note: {n_dot} gensim 'our_dot_float' messages in the log: gensim 4.4's compiled "
                "code treats a dot product of exactly -1.0 as an error code and uses 0 for that "
                "single update (negligible vs ~1e10 updates)")
    rep()

    let7 = np.flatnonzero(mi["mirna_id"].str.startswith("hsa-let-7").to_numpy())
    mir200 = np.flatnonzero(
        mi["mirna_id"].str.fullmatch(r"hsa-miR-(200[abc]|141|429)-.*").to_numpy())
    groups = mi.groupby("seq_group")["mirna_idx"].agg(list)
    groups = [g for g in groups if len(g) > 1]
    clusters = paralog_clusters(gi, rng)
    # Same random pairs for every embedding set, so the columns are comparable.
    mi_pairs = random_pairs(len(mi), 1000, rng)
    g_pairs = random_pairs(len(gi), 1000, rng)
    gene_metrics, metrics, pca_inputs = {}, {}, []

    for name, (mpath, gpath) in sets.items():
        M, G = np.load(mpath).astype(np.float64), np.load(gpath).astype(np.float64)
        m = metrics[name] = {}
        rep(f"--- {name}  (miRNA {M.shape}, gene {G.shape})")
        ok_shape = M.shape[0] == len(mi) and G.shape[0] == len(gi)
        finite = np.isfinite(M).all() and np.isfinite(G).all()
        rep(f"  shapes match index: {ok_shape}; all finite: {finite}; "
            f"zero rows: miRNA {int((~M.any(1)).sum())}, gene {int((~G.any(1)).sum())}")
        Mc = M - M.mean(0)
        m["rand_raw"], m["rand_c"] = pair_cos(M, *mi_pairs), pair_cos(Mc, *mi_pairs)
        rep("  miRNA family test (cosine; 'centred' = after subtracting the mean vector):")
        for lab, key, idx in [("let-7 family", "let7", let7), ("miR-200 family", "mir200", mir200)]:
            m[f"{key}_raw"], m[f"{key}_c"] = mean_pairwise_cos(M[idx]), mean_pairwise_cos(Mc[idx])
            rep(f"    {lab:<16} n={len(idx):>3}  within={m[key + '_raw']:.3f}"
                f"  centred={m[key + '_c']:.3f}")
        rep(f"    {'random pairs':<16} n={len(mi_pairs[0])} mean={m['rand_raw']:.3f}"
            f"  centred={m['rand_c']:.3f}")
        m["fam_margin"] = (m["let7_c"] + m["mir200_c"]) / 2 - m["rand_c"]
        dmax = max((np.linalg.norm(M[g] - M[g][0], axis=1).max() for g in groups), default=0)
        typ = np.median(np.linalg.norm(M[mi_pairs[0]] - M[mi_pairs[1]], axis=1))
        m["shared_max"], m["shared_ratio"] = float(dmax), float(dmax / typ)
        rep(f"  shared-sequence groups ({len(groups)} groups in index): max within-group "
            f"Euclidean distance={dmax:.3g} (median random-pair distance={typ:.3g})")

        # Gene-side checks run once per distinct gene matrix.
        if gpath in gene_metrics:
            rep(f"  gene checks: same gene matrix as above")
            m.update(gene_metrics[gpath])
            rep()
            continue
        gm = gene_metrics[gpath] = {}
        Gc = G - G.mean(0)
        rep("  gene paralog test (cosine within cluster vs random pairs):")
        for lab, idx in clusters.items():
            gm[f"par_{lab}"] = mean_pairwise_cos(Gc[idx])
            rep(f"    {lab:<18} n={len(idx):>4}  within={mean_pairwise_cos(G[idx]):.3f}"
                f"  centred={gm['par_' + lab]:.3f}")
        gm["par_random"] = pair_cos(Gc, *g_pairs)
        rep(f"    {'random pairs':<18} n={len(g_pairs[0])}  mean={pair_cos(G, *g_pairs):.3f}"
            f"  centred={gm['par_random']:.3f}")
        rep("  leakage (Spearman rho; ridge predicts the target from the embedding):")
        ll = leakage(G, log_len, "log10 genomic length", rep, cfg["seed"])
        ld = leakage(G, log_deg, "log10(1+degree)", rep, cfg["seed"])
        gm.update(len_r2=ll["r2"], len_pc1=ll["pc1_rho"], deg_r2=ld["r2"])
        m.update(gm)
        pca_inputs.append(("k-mer genes" if name == "kmer" else "Doc2Vec genes", G))
        rep()

    rep("reference: Spearman rho(log10 length, log10(1+degree)) = "
        f"{pd.Series(log_len).corr(pd.Series(log_deg), method='spearman'):+.3f}")
    rep()
    rep("=" * 70)
    rep("Comparison")
    rep("=" * 70)
    comparison_table(metrics, rep)
    rep()
    recommend(metrics, rep)
    rep.save(P["report"])

    if pca_inputs:
        fig, axes = plt.subplots(1, len(pca_inputs), figsize=(5.2 * len(pca_inputs), 4.6),
                                 squeeze=False)
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import MaxNLocator
        # Single-hue sequential ramp; drop the near-white end so short genes stay visible.
        cmap = LinearSegmentedColormap.from_list(
            "blues_trim", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
        order = np.argsort(log_len)  # draw long genes on top
        for ax, (lab, G) in zip(axes[0], pca_inputs):
            pca = PCA(n_components=2, random_state=cfg["seed"])
            xy = pca.fit_transform(G)
            sc = ax.scatter(xy[order, 0], xy[order, 1], c=log_len[order], cmap=cmap,
                            s=3, linewidths=0, vmin=np.percentile(log_len, 1),
                            vmax=np.percentile(log_len, 99))
            ev = pca.explained_variance_ratio_
            ax.set_title(lab, fontsize=11, loc="left")
            ax.set_xlabel(f"PC1 ({ev[0]:.0%})", color="#555")
            ax.set_ylabel(f"PC2 ({ev[1]:.0%})", color="#555")
            ax.tick_params(colors="#777", labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            for s in ax.spines.values():
                s.set_color("#ccc")
            fig.colorbar(sc, ax=ax, label="log10 genomic length (nt)", shrink=0.85)
        fig.suptitle("Gene embeddings, PCA coloured by gene length", fontsize=12)
        fig.tight_layout()
        fig.savefig(P["pca_png"], dpi=150)
        print(f"Saved {P['pca_png'].relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build-corpus", "kmer", "pilot", "train", "extract",
                                        "check"])
    ap.add_argument("--epochs", type=int, help="override embed.epochs for train")
    args = ap.parse_args()
    cfg = load_config()
    if args.epochs:
        cfg["embed"]["epochs"] = args.epochs
    {"build-corpus": build_corpus, "kmer": build_kmer, "pilot": pilot, "train": train,
     "extract": extract, "check": check}[args.command](cfg)


if __name__ == "__main__":
    main()
