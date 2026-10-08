"""Step 2: one full gene sequence per target gene (Ensembl, local files) + index tables.

Run from the project root:  python3 -m src.genes
Expects the Ensembl files listed under `ensembl:` in configs/default.yaml in data/raw/
(downloaded separately; the genome is decompressed here if only the .fa.gz is present).

Outputs:
  data/interim/genes_genomic.fa (+ .fai)   full gene span, transcription direction
  data/interim/genes_cdna.fa (+ .fai)      canonical (else longest) cDNA per gene
  data/interim/genes.csv
  data/processed/mirna_index.csv, gene_index.csv, positives.csv, known_pairs.csv
  results/step2_report.txt
"""
import gzip
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta, Faidx
from tqdm import tqdm

from src.mirna_names import ARM_RE, Resolver, load_aliases, validate_aliases
from src.parse import LIST_SEP, Report
from src.utils import PROJECT_ROOT, load_config

PRIMARY_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
POSITIVE_LEVELS = {"strong", "weak"}
SANITY_GENES = {"TP53": "-", "EGFR": "+"}
SANITY_NT = 100
ATTR_RE = re.compile(r'(\S+) "([^"]*)"')


def p(cfg_path: str) -> Path:
    return PROJECT_ROOT / cfg_path


def clean_seq(s: str) -> str:
    s = s.upper()
    return re.sub(r"[^ACGT]", "N", s)


def write_fasta_record(fh, name: str, seq: str, width: int = 60):
    fh.write(f">{name}\n")
    for i in range(0, len(seq), width):
        fh.write(seq[i:i + width] + "\n")


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# --------------------------------------------------------------------------- files
def verify_inputs(ens: dict, report: Report):
    report("=" * 70)
    report("Ensembl input files")
    report("=" * 70)
    for key in ("gtf", "cdna", "entrez"):
        f = p(ens[key])
        ok = subprocess.run(["gzip", "-t", str(f)], capture_output=True).returncode == 0
        report(f"  {f.name:<55}{file_size(f):>10}  gzip -t {'OK' if ok else 'FAILED'}")
        if not ok:
            raise RuntimeError(f"{f} failed gzip -t")

    fa, gz = p(ens["genome_fa"]), p(ens["genome_fa_gz"])
    if not fa.exists():
        ok = subprocess.run(["gzip", "-t", str(gz)], capture_output=True).returncode == 0
        report(f"  {gz.name:<55}{file_size(gz):>10}  gzip -t {'OK' if ok else 'FAILED'}")
        if not ok:
            raise RuntimeError(f"{gz} failed gzip -t")
        print("Decompressing genome ...")
        tmp = fa.with_suffix(".fa.part")
        with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)
        tmp.replace(fa)
    if not Path(str(fa) + ".fai").exists():
        print("Indexing genome with pyfaidx ...")
    genome = Fasta(str(fa), sequence_always_upper=True, rebuild=False)
    missing = [c for c in PRIMARY_CHROMS if c not in genome.keys()]
    report(f"  {fa.name:<55}{file_size(fa):>10}  .fai {file_size(Path(str(fa) + '.fai'))}")
    report(f"  genome sequences: {len(genome.keys())} total; primary present: "
           f"{len(PRIMARY_CHROMS) - len(missing)}/{len(PRIMARY_CHROMS)}"
           + (f" MISSING {missing}" if missing else ""))
    if missing:
        raise RuntimeError(f"Primary chromosomes missing from genome: {missing}")
    report()
    return genome


def parse_gtf(path: Path, keep_exons_for: set):
    """Stream the GTF. Returns genes (all chroms), transcripts, and exons for selected genes."""
    genes, transcripts, exons = [], [], defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in tqdm(fh, desc="GTF", unit=" lines", mininterval=2):
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            feature = f[2]
            if feature not in ("gene", "transcript", "exon"):
                continue
            attrs = f[8]
            if feature == "exon":
                gname = re.search(r'gene_name "([^"]*)"', attrs)
                if not (gname and gname.group(1) in keep_exons_for):
                    continue
            kv = defaultdict(list)
            for k, v in ATTR_RE.findall(attrs):
                kv[k].append(v)
            g = lambda k: kv[k][0] if kv[k] else None
            rec = {"chrom": f[0], "start": int(f[3]), "end": int(f[4]), "strand": f[6]}
            if feature == "gene":
                genes.append({"gene_id": g("gene_id"), "gene_name": g("gene_name"),
                              "gene_biotype": g("gene_biotype"), **rec})
            elif feature == "transcript":
                transcripts.append({"transcript_id": g("transcript_id"), "gene_id": g("gene_id"),
                                    "transcript_biotype": g("transcript_biotype"),
                                    "tags": LIST_SEP.join(kv["tag"]), **rec})
            else:
                exons[g("transcript_id")].append(
                    (int(g("exon_number")), rec["start"], rec["end"]))
    genes = pd.DataFrame(genes)
    transcripts = pd.DataFrame(transcripts)
    transcripts["canonical"] = transcripts["tags"].str.split(LIST_SEP).map(
        lambda t: "Ensembl_canonical" in t)
    return genes, transcripts, exons


# --------------------------------------------------------------------------- gene mapping
def pick_gene(cands: pd.DataFrame):
    """Prefer protein_coding, then primary chromosome, then lowest gene_id."""
    if cands.empty:
        return None
    c = cands.assign(_pc=cands["gene_biotype"] != "protein_coding",
                     _prim=~cands["chrom"].isin(PRIMARY_CHROMS))
    return c.sort_values(["_pc", "_prim", "gene_id"]).iloc[0]["gene_id"]


def load_entrez(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str)
    gcol = "gene_stable_id"
    df = df[df["db_name"] == "EntrezGene"]  # EntrezGene_trans_name rows carry names, not IDs
    out = defaultdict(set)
    for gid, x in zip(df[gcol], df["xref"]):
        out[str(x).strip()].add(gid)
    return out


def map_genes(positives: pd.DataFrame, genes: pd.DataFrame, entrez: dict, report: Report):
    by_name = {n: g for n, g in genes.groupby("gene_name")}
    gene_by_id = genes.set_index("gene_id")
    sym_entrez = positives.groupby("target_gene")["entrez_ids"].agg(
        lambda s: {e for v in s.dropna() for e in str(v).split(LIST_SEP) if e})
    rows = []
    for sym, eids in sym_entrez.items():
        gid = pick_gene(by_name[sym]) if sym in by_name else None
        rule = "symbol" if gid else None
        if gid is None:
            ids = {g for e in eids for g in entrez.get(e, ()) if g in gene_by_id.index}
            gid = pick_gene(gene_by_id.loc[sorted(ids)].reset_index()) if ids else None
            rule = "entrez" if gid else "unmapped"
        rows.append({"gene_symbol": sym, "gene_id": gid, "map_rule": rule,
                     "entrez_ids": LIST_SEP.join(sorted(eids))})
    m = pd.DataFrame(rows)

    report("=" * 70)
    report("Target gene -> Ensembl gene_id")
    report("=" * 70)
    report(f"Target gene symbols in positive pairs: {len(m)}")
    for k, v in m["map_rule"].value_counts().items():
        report(f"  {k:<10}{v:>7}")
    un = m[m["map_rule"] == "unmapped"]
    n_pairs = positives["target_gene"].value_counts()
    un = un.assign(n_pairs=un["gene_symbol"].map(n_pairs)).sort_values("n_pairs", ascending=False)
    report(f"30 unmapped examples (most pairs first; {un['n_pairs'].sum()} positive pairs lost):")
    for _, r in un.head(30).iterrows():
        report(f"  {r['gene_symbol']:<20} entrez={r['entrez_ids'] or '-':<14}{r['n_pairs']:>6}")
    dup = m[m["gene_id"].notna()].groupby("gene_id")["gene_symbol"].agg(list)
    dup = dup[dup.map(len) > 1]
    report(f"gene_ids reached by >1 symbol (merged): {len(dup)}"
           + (f"  e.g. {dup.index[0]}: {dup.iloc[0]}" if len(dup) else ""))
    report()
    return m


# --------------------------------------------------------------------------- sequences
def extract_genomic(genome, table: pd.DataFrame, out: Path, max_n: float):
    """Write one record per gene; returns (lengths, n_fractions) dicts. Never holds all seqs."""
    lengths, nfrac, dropped = {}, {}, []
    with open(out, "w") as fh:
        for r in tqdm(table.itertuples(), total=len(table), desc="genomic", mininterval=2):
            rec = genome[r.chrom][r.start - 1:r.end]  # 1-based inclusive -> 0-based half-open
            seq = rec.reverse.complement.seq if r.strand == "-" else rec.seq
            seq = clean_seq(seq)
            nf = seq.count("N") / len(seq)
            nfrac[r.gene_id] = nf
            if nf > max_n:
                dropped.append(r.gene_id)
                continue
            lengths[r.gene_id] = len(seq)
            write_fasta_record(fh, r.gene_id, seq)
    Faidx(str(out), rebuild=True)
    return lengths, nfrac, dropped


def iter_fasta_gz(path: Path):
    """Yield (header, sequence) from a gzipped FASTA, one record at a time."""
    header, chunks = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].rstrip(), []
            else:
                chunks.append(line.rstrip())
    if header is not None:
        yield header, "".join(chunks)


def strip_version(x: str) -> str:
    return x.split(".")[0]


def extract_cdna(cdna_path: Path, gene_ids: set, transcripts: pd.DataFrame, out: Path):
    """Canonical transcript if present in the cDNA FASTA, else the longest cDNA of the gene."""
    canonical = dict(zip(transcripts.loc[transcripts["canonical"], "gene_id"],
                         transcripts.loc[transcripts["canonical"], "transcript_id"]))
    # Pass 1: transcript lengths per gene (headers carry gene:ENSG...).
    tx_len, tx_gene = {}, {}
    for h, s in tqdm(iter_fasta_gz(cdna_path), desc="cDNA pass 1", mininterval=2):
        m = re.search(r"gene:(ENSG\d+)", h)
        if m and m.group(1) in gene_ids:
            tid = strip_version(h.split()[0])
            tx_len[tid], tx_gene[tid] = len(s), m.group(1)
    by_gene = defaultdict(list)
    for tid, gid in tx_gene.items():
        by_gene[gid].append(tid)
    choice = {}
    for gid in gene_ids:
        cand = by_gene.get(gid, [])
        if not cand:
            continue
        c = canonical.get(gid)
        if c in tx_len:
            choice[gid] = (c, "canonical")
        else:
            choice[gid] = (max(cand, key=lambda t: (tx_len[t], t)), "longest")
    wanted = {t: g for g, (t, _) in choice.items()}
    # Pass 2: write chosen transcripts, header = gene_id.
    lengths = {}
    with open(out, "w") as fh:
        for h, s in tqdm(iter_fasta_gz(cdna_path), desc="cDNA pass 2", mininterval=2):
            tid = strip_version(h.split()[0])
            if tid in wanted:
                seq = clean_seq(s)
                lengths[wanted[tid]] = len(seq)
                write_fasta_record(fh, wanted[tid], seq)
    Faidx(str(out), rebuild=True)
    return choice, lengths


def sanity_checks(genes_tbl, transcripts, exons, genomic_fa, cdna_fa, choice, report):
    report("=" * 70)
    report("Sanity checks: genomic vs canonical cDNA at the transcription start site")
    report("=" * 70)
    gfa, cfa = Fasta(str(genomic_fa), as_raw=True), Fasta(str(cdna_fa), as_raw=True)
    tx = transcripts.set_index("transcript_id")
    all_ok = True
    for sym, expected_strand in SANITY_GENES.items():
        row = genes_tbl[genes_tbl["gene_name"] == sym]
        if row.empty:
            report(f"  {sym}: not among the extracted genes -- check skipped")
            all_ok = False
            continue
        r = row.iloc[0]
        tid, how = choice[r.gene_id]
        t = tx.loc[tid]
        offset = (t.start - r.start) if r.strand == "+" else (r.end - t.end)
        first_exon = sorted(exons.get(tid, []))
        exon1_len = (first_exon[0][2] - first_exon[0][1] + 1) if first_exon else None
        n = min(SANITY_NT, exon1_len) if exon1_len else SANITY_NT
        genomic = str(gfa[r.gene_id][offset:offset + n])
        cdna = str(cfa[r.gene_id][:n])
        plain = str(gfa[r.gene_id][:n]) == cdna
        ok = genomic == cdna
        all_ok &= ok and r.strand == expected_strand
        report(f"  {sym} ({r.gene_id}, strand {r.strand}, expected {expected_strand}) "
               f"transcript {tid} [{how}]")
        report(f"    gene span {r.chrom}:{r.start}-{r.end}; transcript {t.start}-{t.end}; "
               f"TSS offset in gene = {offset}; first exon = {exon1_len} nt; compared {n} nt")
        report(f"    TSS-adjusted comparison: {'MATCH' if ok else 'MISMATCH'}")
        report(f"    plain gene-start comparison (info): {'match' if plain else 'no match'}")
        report(f"    genomic@TSS: {genomic[:60]}")
        report(f"    cDNA[:60]  : {cdna[:60]}")
    report()
    if not all_ok:
        raise RuntimeError("Sanity check failed -- strand/coordinate handling is wrong")


# --------------------------------------------------------------------------- known pairs
def build_resolver(cfg, mirnas):
    alias_path = p(cfg["aliases"]["path"])
    if alias_path.exists() and validate_aliases(alias_path):
        return Resolver(mirnas, *load_aliases(alias_path))
    return Resolver(mirnas)


def renamed_candidates(norm: str, ids_lower: dict) -> list:
    """mature.fa IDs that extend the name with a letter suffix (optional -N copy, kept arm)."""
    arm = ARM_RE.search(norm)
    base = norm[:arm.start()] if arm else norm
    arm_pat = re.escape(arm.group(0)) if arm else r"(-[35]p)?"
    pat = re.compile(re.escape(base.lower()) + r"[a-z](-\d+)?" + arm_pat.lower() + "$")
    return sorted(i for low, i in ids_lower.items() if pat.match(low))


def build_known_pairs(cfg, mirnas, mapped_all, mirna_idx, sym_to_gidx, report):
    """All pairs that might be real, restricted to miRNAs/genes in the indexes."""
    parts = []
    # 1) all mapped pairs (any evidence level)
    m = mapped_all.assign(mirna_idx=mapped_all["mirna_id_mapped"].map(mirna_idx),
                          gene_idx=mapped_all["target_gene"].map(sym_to_gidx))
    parts.append(m[["mirna_idx", "gene_idx"]].assign(source="mapped"))

    # 2) + 3) unmapped names from interactions_raw
    raw = pd.read_csv(p(cfg["paths"]["interim_dir"]) / "interactions_raw.csv",
                      usecols=["mirna", "target_gene"], dtype=str)
    resolver = build_resolver(cfg, mirnas)
    ids_lower = {i.lower(): i for i in mirnas["mirna_id"]}
    expand, stats = {}, defaultdict(int)
    for name in raw["mirna"].unique():
        norm, mapped, rule = resolver.resolve(name)
        if rule == "ambiguous_arm":
            cands = [ids_lower[c] for c in (norm.lower() + "-5p", norm.lower() + "-3p")
                     if c in ids_lower]
            expand[name] = (cands, "ambiguous_arm")
        elif rule == "not_found":
            cands = renamed_candidates(norm, ids_lower)
            if cands:
                expand[name] = (cands, "renamed_candidate")
                stats["renamed_names"] += 1
    e = raw[raw["mirna"].isin(expand)]
    e = e.assign(cand=e["mirna"].map(lambda n: expand[n][0]),
                 source=e["mirna"].map(lambda n: expand[n][1])).explode("cand")
    e = e.assign(mirna_idx=e["cand"].map(mirna_idx), gene_idx=e["target_gene"].map(sym_to_gidx))
    parts.append(e[["mirna_idx", "gene_idx", "source"]])

    allp = pd.concat(parts, ignore_index=True)
    before = allp.groupby("source").size()
    allp = allp.dropna(subset=["mirna_idx", "gene_idx"])
    allp[["mirna_idx", "gene_idx"]] = allp[["mirna_idx", "gene_idx"]].astype(int)
    in_index = allp.groupby("source").size()
    known = allp.groupby(["mirna_idx", "gene_idx"], as_index=False)["source"].agg(
        lambda s: LIST_SEP.join(sorted(set(s))))
    return known.sort_values(["mirna_idx", "gene_idx"], ignore_index=True), before, in_index, stats


# --------------------------------------------------------------------------- main
def describe(lengths) -> str:
    a = np.fromiter(lengths, dtype=np.int64)
    q = np.percentile(a, [50, 90, 99])
    return (f"n={len(a)} min={a.min()} median={q[0]:.0f} p90={q[1]:.0f} "
            f"p99={q[2]:.0f} max={a.max()} total_nt={a.sum():,}")


def main():
    cfg = load_config()
    ens = cfg["ensembl"]
    interim, processed = p(cfg["paths"]["interim_dir"]), p(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    report = Report()

    genome = verify_inputs(ens, report)
    report(f"Ensembl release: {ens['release']}")
    report()

    # ---- positives (mapped pairs, strong/weak)
    mapped_all = pd.read_csv(interim / "interactions_mapped.csv", dtype=str)
    positives = mapped_all[mapped_all["evidence_level"].isin(POSITIVE_LEVELS)]

    # ---- GTF
    all_genes, transcripts, exons = parse_gtf(p(ens["gtf"]), set(SANITY_GENES))
    genes = all_genes[all_genes["chrom"].isin(PRIMARY_CHROMS)]
    report("=" * 70)
    report("GTF")
    report("=" * 70)
    report(f"gene records: {len(all_genes)}; on 1-22/X/Y/MT: {len(genes)}")
    report(f"transcript records: {len(transcripts)}; Ensembl_canonical: "
           f"{transcripts['canonical'].sum()}")
    report()

    # ---- symbol -> gene_id
    entrez = load_entrez(p(ens["entrez"]))
    gmap = map_genes(positives, genes, entrez, report)
    gmap = gmap[gmap["map_rule"] != "unmapped"]
    ginfo = genes.set_index("gene_id")
    per_gene = gmap.groupby("gene_id").agg(
        gene_symbol=("gene_symbol", lambda s: LIST_SEP.join(sorted(s))),
        map_rule=("map_rule", lambda s: LIST_SEP.join(sorted(set(s))))).reset_index()
    table = per_gene.join(ginfo, on="gene_id").sort_values(["chrom", "start"])

    # ---- genomic
    max_n = cfg["genes"]["max_n_fraction"]
    gfa_path = interim / "genes_genomic.fa"
    glen, nfrac, dropped = extract_genomic(genome, table, gfa_path, max_n)
    report("=" * 70)
    report("Sequences")
    report("=" * 70)
    report(f"Genes extracted (genomic): {len(table)}; dropped for >{max_n:.0%} N: {len(dropped)}")
    for gid in dropped:
        r = table[table["gene_id"] == gid].iloc[0]
        report(f"  {gid} {r.gene_symbol} {r.chrom}:{r.start}-{r.end} N={nfrac[gid]:.1%}")
    table = table[~table["gene_id"].isin(dropped)]

    # ---- cDNA
    cfa_path = interim / "genes_cdna.fa"
    choice, clen = extract_cdna(p(ens["cdna"]), set(table["gene_id"]), transcripts, cfa_path)
    how = pd.Series([c[1] for c in choice.values()]).value_counts()
    no_cdna = table[~table["gene_id"].isin(clen)]
    report(f"cDNA chosen: {how.get('canonical', 0)} canonical, {how.get('longest', 0)} longest; "
           f"{len(no_cdna)} genes have no transcript in cdna.all (e.g. ncRNA-only genes)")
    if len(no_cdna):
        report("  no-cDNA biotypes: " + ", ".join(
            f"{k}={v}" for k, v in no_cdna["gene_biotype"].value_counts().head(8).items()))
    report()

    sanity_checks(table, transcripts, exons, gfa_path, cfa_path, choice, report)

    # ---- genes.csv
    table = table.assign(
        genomic_length=table["gene_id"].map(glen),
        cdna_length=table["gene_id"].map(clen).astype("Int64"),
        cdna_transcript_id=table["gene_id"].map(lambda g: choice.get(g, (None,))[0]),
        cdna_choice=table["gene_id"].map(lambda g: choice.get(g, (None, None))[1]),
    ).rename(columns={"gene_biotype": "biotype"})
    out_cols = ["gene_symbol", "gene_id", "map_rule", "chrom", "start", "end", "strand", "biotype",
                "genomic_length", "cdna_length", "cdna_transcript_id", "cdna_choice"]
    table = table.sort_values("gene_id")
    table[out_cols].to_csv(interim / "genes.csv", index=False)

    # ---- indexes and positives
    sym_to_gene = dict(zip(gmap["gene_symbol"], gmap["gene_id"]))
    kept = set(table["gene_id"])
    pos = positives.assign(gene_id=positives["target_gene"].map(sym_to_gene))
    n_pos0 = len(pos)
    pos_mapped = pos[pos["gene_id"].notna()]
    pos = pos_mapped[pos_mapped["gene_id"].isin(kept)]

    gene_index = table[["gene_id", "gene_symbol"]].reset_index(drop=True)
    gene_index.insert(0, "gene_idx", range(len(gene_index)))
    mirnas = pd.read_csv(interim / "mirna.csv")
    mirna_index = (mirnas[mirnas["mirna_id"].isin(pos["mirna_id_mapped"])]
                   .sort_values("mirna_id").reset_index(drop=True))
    mirna_index.insert(0, "mirna_idx", range(len(mirna_index)))
    gene_index.to_csv(processed / "gene_index.csv", index=False)
    mirna_index.to_csv(processed / "mirna_index.csv", index=False)
    gidx = dict(zip(gene_index["gene_id"], gene_index["gene_idx"]))
    midx = dict(zip(mirna_index["mirna_id"], mirna_index["mirna_idx"]))

    pos = pos.assign(n_references=pos["n_references"].astype(int),
                     _rank=(pos["evidence_level"] != "strong").astype(int))
    # Two symbols can resolve to one gene_id: keep the strongest evidence, max references.
    pos_out = (pos.sort_values(["_rank", "n_references"], ascending=[True, False])
               .groupby(["mirna_id_mapped", "gene_id"], as_index=False)
               .agg(evidence_level=("evidence_level", "first"),
                    n_references=("n_references", "max"),
                    gene_symbol=("target_gene", lambda s: LIST_SEP.join(sorted(set(s))))))
    pos_out = pos_out.rename(columns={"mirna_id_mapped": "mirna_id"})
    pos_out.insert(0, "mirna_idx", pos_out["mirna_id"].map(midx))
    pos_out.insert(1, "gene_idx", pos_out["gene_id"].map(gidx))
    pos_out = pos_out[["mirna_idx", "gene_idx", "mirna_id", "gene_id", "gene_symbol",
                       "evidence_level", "n_references"]].sort_values(["mirna_idx", "gene_idx"])
    pos_out.to_csv(processed / "positives.csv", index=False)

    # ---- known pairs
    sym_to_gidx = {s: gidx[g] for s, g in sym_to_gene.items() if g in gidx}
    # Symbols outside the positives (e.g. nonfunctional_only or unmapped-miRNA pairs) count
    # only if they name a gene already in the index.
    name_to_gid = dict(zip(genes["gene_name"], genes["gene_id"]))
    for s in set(mapped_all["target_gene"]) - set(sym_to_gidx):
        if name_to_gid.get(s) in gidx:
            sym_to_gidx[s] = gidx[name_to_gid[s]]
    known, before, in_index, kstats = build_known_pairs(
        cfg, mirnas, mapped_all, midx, sym_to_gidx, report)
    known.to_csv(processed / "known_pairs.csv", index=False)

    # ---- report
    report("=" * 70)
    report("Lengths")
    report("=" * 70)
    report("genomic: " + describe(table["genomic_length"]))
    report("cdna:    " + describe(table["cdna_length"].dropna().astype(int)))
    report()

    report("=" * 70)
    report("Filtering / final counts")
    report("=" * 70)
    report(f"Mapped pairs (all evidence levels):           {len(mapped_all)}")
    report(f"Positive pairs (strong/weak):                 {n_pos0}")
    report(f"  with target gene mapped to Ensembl:         {len(pos_mapped)}")
    report(f"  after dropping high-N genes:                {len(pos)}")
    report(f"  after merging symbols on the same gene_id:  {len(pos_out)}")
    report(f"Final positives: {len(pos_out)}; miRNAs: {len(mirna_index)}; genes: {len(gene_index)}")
    report("Positives by evidence_level:")
    for k, v in pos_out["evidence_level"].value_counts().items():
        report(f"  {k:<10}{v:>9}")
    report(f"miRNAs in index belonging to a shared-sequence group: "
           f"{mirna_index['seq_group'].duplicated(keep=False).sum()}")
    report()

    report("=" * 70)
    report("known_pairs.csv (excluded from negative sampling)")
    report("=" * 70)
    report(f"not_found names with renamed candidates: {kstats['renamed_names']}")
    report(f"{'source':<20}{'rows':>10}{'in index':>10}{'only this source':>18}")
    only = known[~known["source"].str.contains(LIST_SEP)]["source"].value_counts()
    for s in before.index:
        report(f"{s:<20}{before[s]:>10}{in_index.get(s, 0):>10}{only.get(s, 0):>18}")
    report(f"Unique (mirna_idx, gene_idx) known pairs: {len(known)}")
    report(f"  of which also positives: "
           f"{len(known.merge(pos_out[['mirna_idx', 'gene_idx']]))}")
    report(f"  extra beyond positives:  {len(known) - len(pos_out)}")
    report(f"Density: known pairs cover {len(known) / (len(mirna_index) * len(gene_index)):.2%} "
           f"of the {len(mirna_index)} x {len(gene_index)} grid")
    report()

    report("=" * 70)
    report("Disk usage")
    report("=" * 70)
    for f in [p(ens["genome_fa"]), Path(str(p(ens["genome_fa"])) + ".fai"),
              gfa_path, Path(str(gfa_path) + ".fai"), cfa_path, Path(str(cfa_path) + ".fai"),
              interim / "genes.csv", processed / "gene_index.csv", processed / "mirna_index.csv",
              processed / "positives.csv", processed / "known_pairs.csv"]:
        report(f"  {str(f.relative_to(PROJECT_ROOT)):<60}{file_size(f):>10}")

    report.save(p(cfg["paths"]["results_dir"]) / "step2_report.txt")
    print("\nSaved results/step2_report.txt")


if __name__ == "__main__":
    main()
