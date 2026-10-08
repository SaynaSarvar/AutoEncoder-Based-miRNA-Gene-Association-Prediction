"""Step 1: parse mature.fa (human miRNAs) and miRTarBase hsa_MTI.csv (interactions).

Run from the project root:  python3 -m src.parse
Outputs:
  data/interim/mirna.csv
  data/interim/interactions_raw.csv
  results/step1_report.txt
"""
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src.utils import PROJECT_ROOT, load_config

HUMAN_SPECIES = {"Homo sapiens", "hsa"}  # miRTarBase versions use either form
STRONG_SUPPORT = "Functional MTI"
LIST_SEP = ";"

# Canonical column name -> lowercase substrings used to locate it in whatever header the file has.
COLUMN_PATTERNS = {
    "mirtarbase_id": ["mirtarbase id"],
    "mirna": ["mirna"],
    "species_mirna": ["species (mirna)"],
    "target_gene": ["target gene"],
    "entrez_id": ["entrez"],
    "species_target": ["species (target gene)"],
    "experiments": ["experiments"],
    "support_type": ["support type"],
    "references": ["references", "pmid"],
}


class Report:
    """Collects lines, printing them as they are added."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# --------------------------------------------------------------------------- miRNAs
def parse_mirnas(fasta_path: Path, report: Report) -> pd.DataFrame:
    rows = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        if not rec.id.startswith("hsa-"):
            continue
        tokens = rec.description.split()
        seq = str(rec.seq).upper()
        rows.append({
            "mirna_id": rec.id,
            "mimat": tokens[1] if len(tokens) > 1 else None,
            "sequence": seq,
            "sequence_dna": seq.replace("U", "T"),
        })
    df = pd.DataFrame(rows)

    report("=" * 70)
    report("miRNAs (mature.fa)")
    report("=" * 70)
    report("Human mature miRNAs (hsa-*):", len(df))

    dup_ids = df[df.duplicated("mirna_id", keep=False)]
    report("Duplicate mirna_id entries:", dup_ids["mirna_id"].nunique())
    for mid, g in dup_ids.groupby("mirna_id"):
        report(f"  {mid}: {', '.join(g['mimat'])}")

    dup_seqs = df[df.duplicated("sequence", keep=False)]
    report(f"Duplicate sequences: {dup_seqs['sequence'].nunique()} sequences "
           f"shared by {len(dup_seqs)} miRNAs")
    for seq, g in list(dup_seqs.groupby("sequence"))[:20]:
        report(f"  {seq}: {', '.join(g['mirna_id'])}")
    if dup_seqs["sequence"].nunique() > 20:
        report("  ... (first 20 shown)")
    report()
    return df


# --------------------------------------------------------------------------- interactions
def read_csv_any_encoding(path: Path, report: Report) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):  # utf-8-sig = utf-8 that also strips a BOM
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            report(f"Read {path.name} with encoding={enc}")
            return df
        except UnicodeDecodeError:
            report(f"Encoding {enc} failed, trying next")
    raise RuntimeError(f"Could not decode {path}")


def resolve_columns(columns) -> dict:
    """Map canonical names to actual column names. More specific patterns are matched first
    so that e.g. 'miRNA' does not grab 'Species (miRNA)'."""
    remaining = list(columns)
    mapping = {}
    order = ["species_mirna", "species_target", "entrez_id", "mirtarbase_id",
             "target_gene", "mirna", "experiments", "support_type", "references"]
    for key in order:
        pats = COLUMN_PATTERNS[key]
        # Prefer exact (case-insensitive) match, then substring match.
        exact = [c for c in remaining if c.strip().lower() in pats]
        sub = [c for c in remaining if any(p in c.strip().lower() for p in pats)]
        hit = (exact or sub or [None])[0]
        if hit is not None:
            mapping[key] = hit
            remaining.remove(hit)
    missing = [k for k in ("mirna", "target_gene", "support_type") if k not in mapping]
    if missing:
        raise KeyError(f"Required columns not found: {missing}. Columns: {list(columns)}")
    return mapping


def sorted_unique(values) -> str:
    return LIST_SEP.join(sorted({v for v in values if isinstance(v, str) and v.strip()}))


def clean_pmid(v):
    """'17179747.0' -> '17179747'."""
    if not isinstance(v, str) or not v.strip():
        return None
    v = v.strip()
    return v[:-2] if v.endswith(".0") else v


def parse_interactions(csv_path: Path, report: Report):
    report("=" * 70)
    report("Interactions (miRTarBase)")
    report("=" * 70)
    raw = read_csv_any_encoding(csv_path, report)

    report("Columns / dtypes:")
    for c, t in raw.dtypes.items():
        report(f"  {c!r}: {t}")
    report("Raw rows:", len(raw))
    report("Example rows:")
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.max_colwidth", 40):
        for line in raw.head(5).to_string().splitlines():
            report("  " + line)

    cols = resolve_columns(raw.columns)
    report("Column mapping:", cols)
    df = raw.rename(columns={v: k for k, v in cols.items()})
    for c in df.columns:
        df[c] = df[c].str.strip()

    for sc in ("species_mirna", "species_target"):
        if sc in df:
            report(f"{sc} values:", df[sc].value_counts(dropna=False).to_dict())

    human = df
    for sc in ("species_mirna", "species_target"):
        if sc in human:
            human = human[human[sc].isin(HUMAN_SPECIES)]
    report("Human rows (both species human):", len(human))

    human = human.copy()
    human["pmid"] = human["references"].map(clean_pmid) if "references" in human else None

    agg = {
        "support_types": ("support_type", sorted_unique),
        "n_rows": ("support_type", "size"),
    }
    if "mirtarbase_id" in human:
        agg["mirtarbase_ids"] = ("mirtarbase_id", sorted_unique)
    if "entrez_id" in human:
        agg["entrez_ids"] = ("entrez_id", lambda s: sorted_unique(s.map(clean_pmid)))
    if "experiments" in human:
        agg["experiments"] = ("experiments",
                              lambda s: sorted_unique(e for x in s.dropna() for e in x.split("//")))
    agg["pmids"] = ("pmid", sorted_unique)
    agg["n_references"] = ("pmid", lambda s: s.dropna().nunique())

    pairs = human.groupby(["mirna", "target_gene"], as_index=False).agg(**agg)
    pairs["strong_evidence"] = pairs["support_types"].map(
        lambda s: STRONG_SUPPORT in s.split(LIST_SEP))

    report("Unique (miRNA, gene) pairs:", len(pairs))
    report("Unique miRNAs:", pairs["mirna"].nunique())
    report("Unique genes:", pairs["target_gene"].nunique())
    report("Pairs with strong_evidence (Functional MTI):", int(pairs["strong_evidence"].sum()))
    report()
    report("Support type counts (human rows):")
    for k, v in human["support_type"].value_counts(dropna=False).items():
        report(f"  {k}: {v}")
    report("Support type counts (unique pairs, a pair may have several):")
    exploded = pairs["support_types"].str.split(LIST_SEP).explode()
    for k, v in exploded.value_counts().items():
        report(f"  {k}: {v}")
    report()
    return pairs


# --------------------------------------------------------------------------- matching
def report_matching(mirnas: pd.DataFrame, pairs: pd.DataFrame, report: Report):
    report("=" * 70)
    report("miRNA name matching (exact, no fixes)")
    report("=" * 70)
    known = set(mirnas["mirna_id"])
    names = pd.Series(pairs["mirna"].unique())
    matched = names.isin(known)
    report(f"Interaction miRNA names matching mature.fa mirna_id: "
           f"{matched.sum()} / {len(names)}")
    surviving = pairs["mirna"].isin(known)
    report(f"Pairs surviving the match: {surviving.sum()} / {len(pairs)}")
    report(f"Strong-evidence pairs surviving: "
           f"{(surviving & pairs['strong_evidence']).sum()} / {pairs['strong_evidence'].sum()}")
    unmatched = names[~matched].sort_values()
    report(f"Examples of unmatched miRNA names (20 of {len(unmatched)}):")
    for n in unmatched.head(20):
        report("  " + n)
    report()

    report("Top 10 miRNAs by number of interactions:")
    for k, v in pairs["mirna"].value_counts().head(10).items():
        report(f"  {k}: {v}")
    report("Top 10 genes by number of interactions:")
    for k, v in pairs["target_gene"].value_counts().head(10).items():
        report(f"  {k}: {v}")


def main():
    cfg = load_config()
    paths = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    report = Report()

    mirnas = parse_mirnas(paths["mature_fa"], report)
    out = paths["interim_dir"] / "mirna.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    mirnas.to_csv(out, index=False)

    pairs = parse_interactions(paths["mti_csv"], report)
    pairs.to_csv(paths["interim_dir"] / "interactions_raw.csv", index=False)

    report_matching(mirnas, pairs, report)

    report.save(paths["results_dir"] / "step1_report.txt")
    print(f"\nSaved: {out}, {paths['interim_dir'] / 'interactions_raw.csv'}, "
          f"{paths['results_dir'] / 'step1_report.txt'}")


if __name__ == "__main__":
    main()
