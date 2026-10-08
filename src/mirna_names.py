"""Step 1b: map miRTarBase miRNA names to mature.fa IDs.

Run from the project root:  python3 -m src.mirna_names

If data/raw/aliases.txt is missing, the miRBase sources in configs/default.yaml are tried in
order; a download is accepted only if it validates. With a valid aliases file the rule order is
    exact -> alias -> case_insensitive
otherwise (mature.fa only) it is
    exact -> case_insensitive -> single_arm   (ambiguous_arm names stay unmapped)
Dropping a valid aliases.txt into data/raw/ and re-running switches to alias mode automatically.

Outputs:
  data/interim/mirna.csv               (overwritten, adds seq_group)
  data/interim/interactions_mapped.csv
  results/step1b_report.txt
"""
import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.parse import LIST_SEP, Report, sorted_unique
from src.utils import PROJECT_ROOT, load_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

DASHES = "‐‑‒–—―−﹘﹣－"
DASH_RE = re.compile(f"[{DASHES}]")
ARM_RE = re.compile(r"-[35]p$")
# Prefixes before the first "-" that are not species codes.
NON_SPECIES_PREFIXES = {"hsa", "mir", "let", "microrna", "mirna"}

LIST_COLUMNS = ["support_types", "mirtarbase_ids", "entrez_ids", "experiments", "pmids"]


# --------------------------------------------------------------------------- aliases
def validate_aliases(path: Path) -> bool:
    """Tab-separated, many MIMAT lines, and MIMAT0000062 lists hsa-let-7a."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    mimat = [l for l in lines if l.startswith("MIMAT") and "\t" in l]
    if len(mimat) < 1000:
        return False
    let7 = [l for l in mimat if l.split("\t")[0].strip() == "MIMAT0000062"]
    return bool(let7) and "hsa-let-7a" in let7[0].split("\t")[1].split(";")


def try_download_aliases(sources: list, dest: Path, report: Report):
    """Try each source in order; keep the first that validates. Returns the source dict or None."""
    for src in sources:
        url = src["url"]
        tmp = dest.with_suffix(".download")
        try:
            r = requests.get(url, timeout=60)
            status = r.status_code
            data = r.content if r.ok else b""
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            tmp.write_bytes(data)
            ok = r.ok and validate_aliases(tmp)
        except Exception as e:  # network errors, unsupported scheme (ftp), bad gzip
            status, ok = type(e).__name__, False
        report(f"  {url} -> {status}, {'valid' if ok else 'rejected'}")
        if ok:
            tmp.replace(dest)
            return src
        tmp.unlink(missing_ok=True)
    return None


def record_alias_source(src):
    """Store the aliases source/version in configs/default.yaml."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["aliases"]["used_source"] = src["url"] if src else None
    cfg["aliases"]["used_version"] = src.get("version") if src else None
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


def load_aliases(path: Path):
    """Return (alias -> MIMAT for unambiguous human aliases, set of ambiguous aliases)."""
    alias_to_mimats = defaultdict(set)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0].startswith("MIMAT"):
            continue
        for name in parts[1].split(";"):
            name = name.strip()
            if name.startswith("hsa-"):
                alias_to_mimats[name].add(parts[0].strip())
    aliases = {a: next(iter(m)) for a, m in alias_to_mimats.items() if len(m) == 1}
    ambiguous = {a for a, m in alias_to_mimats.items() if len(m) > 1}
    return aliases, ambiguous


# --------------------------------------------------------------------------- normalisation
def normalise(name: str):
    """Return (normalised name, is_non_human)."""
    n = DASH_RE.sub("-", name.strip())
    prefix = n.split("-", 1)[0].lower()
    if prefix not in NON_SPECIES_PREFIXES and re.fullmatch(r"[a-z]{2,5}", prefix):
        return n, True
    low = n.lower()
    if low.startswith("hsa-"):
        n = "hsa-" + n[4:]
    elif low.startswith("microrna-") or low.startswith("mirna-"):
        n = "hsa-miR-" + n.split("-", 1)[1]
    elif low.startswith("mir-"):
        n = "hsa-miR-" + n[4:]
    elif low.startswith("let-"):
        n = "hsa-let-" + n[4:]
    return n, False


class Resolver:
    def __init__(self, mirnas: pd.DataFrame, aliases=None, ambiguous_aliases=None):
        self.ids = set(mirnas["mirna_id"])
        self.mimat_to_id = dict(zip(mirnas["mimat"], mirnas["mirna_id"]))
        lower = defaultdict(set)
        for i in self.ids:
            lower[i.lower()].add(i)
        self.lower = lower
        self.aliases = aliases
        self.ambiguous_aliases = ambiguous_aliases or set()

    def _ci(self, name):
        hits = self.lower.get(name.lower(), set())
        return next(iter(hits)) if len(hits) == 1 else None

    def resolve(self, raw_name: str):
        """Return (normalised, mapped_id or None, rule or unmapped reason)."""
        n, non_human = normalise(raw_name)
        if non_human:
            return n, None, "non_human"
        if n in self.ids:
            return n, n, "exact"
        if self.aliases is not None:
            mimat = self.aliases.get(n)
            if mimat in self.mimat_to_id:
                return n, self.mimat_to_id[mimat], "alias"
            hit = self._ci(n)
            if hit:
                return n, hit, "case_insensitive"
            if n in self.ambiguous_aliases:
                return n, None, "ambiguous_alias"
            return n, None, "not_found"
        # Fallback: mature.fa only.
        hit = self._ci(n)
        if hit:
            return n, hit, "case_insensitive"
        if not ARM_RE.search(n):
            arms = [a for a in (self._ci(n + "-5p"), self._ci(n + "-3p")) if a]
            if len(arms) == 1:
                return n, arms[0], "single_arm"
            if len(arms) == 2:
                return n, None, "ambiguous_arm"
        return n, None, "not_found"


# --------------------------------------------------------------------------- merging
def union_lists(s: pd.Series) -> str:
    return sorted_unique(x for v in s.dropna() for x in str(v).split(LIST_SEP))


def evidence_level(support_types: str) -> str:
    types = set(support_types.split(LIST_SEP))
    if "Functional MTI" in types:
        return "strong"
    if "Functional MTI (Weak)" in types:
        return "weak"
    return "nonfunctional_only"


def merge_pairs(mapped: pd.DataFrame) -> pd.DataFrame:
    keys = ["mirna_id_mapped", "target_gene"]
    multi = mapped.duplicated(keys, keep=False)
    # Only groups with >1 source name need the (slow) union; the rest pass through.
    single = mapped[~multi].rename(columns={"mirna": "mirna_original"})
    agg = {c: (c, union_lists) for c in LIST_COLUMNS if c in mapped}
    agg["mirna_original"] = ("mirna", union_lists)
    agg["map_rule"] = ("map_rule", union_lists)
    agg["n_rows"] = ("n_rows", "sum")
    merged = mapped[multi].groupby(keys, as_index=False).agg(**agg)
    out = pd.concat([single, merged], ignore_index=True)
    out["n_references"] = out["pmids"].fillna("").map(
        lambda s: len([p for p in s.split(LIST_SEP) if p]))
    out["strong_evidence"] = out["support_types"].map(
        lambda s: "Functional MTI" in s.split(LIST_SEP))
    out["evidence_level"] = out["support_types"].map(evidence_level)
    first = ["mirna_id_mapped", "target_gene", "mirna_original", "map_rule", "evidence_level",
             "strong_evidence", "n_references"]
    out = out[first + [c for c in out.columns if c not in first]]
    return out.sort_values(keys, ignore_index=True), int(multi.sum())


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    paths = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    alias_path = PROJECT_ROOT / cfg["aliases"]["path"]
    report = Report()

    report("=" * 70)
    report("miRBase aliases")
    report("=" * 70)
    if alias_path.exists() and validate_aliases(alias_path):
        report(f"Using existing {alias_path.relative_to(PROJECT_ROOT)} "
               f"(source recorded in config: {cfg['aliases'].get('used_source') or 'unknown/manual'})")
        src = {"url": cfg["aliases"].get("used_source") or "manual",
               "version": cfg["aliases"].get("used_version")}
    else:
        if alias_path.exists():
            report(f"{alias_path.name} exists but fails validation; ignoring it")
        report("Trying download sources:")
        src = try_download_aliases(cfg["aliases"]["sources"], alias_path, report)
        record_alias_source(src)
    if src:
        aliases, ambiguous = load_aliases(alias_path)
        report(f"Alias source: {src['url']} (version: {src.get('version') or 'unknown'})")
        report(f"Human aliases: {len(aliases)} unambiguous, {len(ambiguous)} ambiguous (excluded)")
        mode = "alias"
    else:
        aliases, ambiguous = None, None
        report("No valid aliases file -> FALLBACK mode (mature.fa only)")
        mode = "fallback"
    report()

    # ---- miRNAs + shared sequence groups
    mirnas = pd.read_csv(paths["interim_dir"] / "mirna.csv")
    mirnas["seq_group"] = pd.factorize(mirnas["sequence"])[0]
    mirnas.to_csv(paths["interim_dir"] / "mirna.csv", index=False)
    group_size = mirnas["seq_group"].map(mirnas["seq_group"].value_counts())
    shared_ids = set(mirnas.loc[group_size > 1, "mirna_id"])

    # ---- resolve names
    pairs = pd.read_csv(paths["interim_dir"] / "interactions_raw.csv", dtype=str)
    pairs["n_rows"] = pairs["n_rows"].astype(int)
    resolver = Resolver(mirnas, aliases, ambiguous)
    names = pd.DataFrame(
        [(n, *resolver.resolve(n)) for n in pairs["mirna"].unique()],
        columns=["mirna", "normalised", "mirna_id_mapped", "map_rule"])
    name_pairs = pairs["mirna"].value_counts()
    names["n_pairs"] = names["mirna"].map(name_pairs)

    pairs = pairs.merge(names[["mirna", "mirna_id_mapped", "map_rule"]], on="mirna", how="left")
    mapped = pairs[pairs["mirna_id_mapped"].notna()]
    final, n_merged_rows = merge_pairs(mapped)
    final.to_csv(paths["interim_dir"] / "interactions_mapped.csv", index=False)

    # ---- report
    report("=" * 70)
    report(f"Name resolution (mode: {mode})")
    report("=" * 70)
    report(f"Distinct miRTarBase names: {len(names)}")
    report(f"{'rule / reason':<20}{'names':>8}{'pairs':>10}")
    for rule, g in names.groupby("map_rule"):
        tag = "mapped" if g["mirna_id_mapped"].notna().all() else "unmapped"
        report(f"{rule:<20}{len(g):>8}{g['n_pairs'].sum():>10}   ({tag})")
    report()

    if mode == "fallback":
        amb = names[names["map_rule"] == "ambiguous_arm"].sort_values("n_pairs", ascending=False)
        report(f"Pairs lost to ambiguous_arm: {amb['n_pairs'].sum()} "
               f"({len(amb)} names) -- this is what the missing aliases file costs")
        report("Top 30 ambiguous_arm names by pair count:")
        for _, r in amb.head(30).iterrows():
            report(f"  {r['mirna']:<24}{r['n_pairs']:>7}")
        report()
        sa = names[names["map_rule"] == "single_arm"].sort_values("n_pairs", ascending=False)
        report("Top 10 single_arm mappings by pair count:")
        for _, r in sa.head(10).iterrows():
            report(f"  {r['mirna']:<24}-> {r['mirna_id_mapped']:<24}{r['n_pairs']:>7}")
        report()

    nf = names[names["map_rule"] == "not_found"].sort_values("n_pairs", ascending=False)
    report(f"30 example not_found names (most pairs first, of {len(nf)}):")
    for _, r in nf.head(30).iterrows():
        shown = r["mirna"] if r["mirna"] == r["normalised"] else f"{r['mirna']}  (-> {r['normalised']})"
        report(f"  {shown:<44}{r['n_pairs']:>6}")
    report()

    report("=" * 70)
    report("Pairs")
    report("=" * 70)
    report(f"Pairs before mapping (interactions_raw):   {len(pairs)}")
    report(f"Pairs with a mapped miRNA:                 {len(mapped)}")
    report(f"  of which collapsed by merging names:     {n_merged_rows} rows -> "
           f"{n_merged_rows - (len(mapped) - len(final))} pairs")
    report(f"Final unique pairs:                        {len(final)}")
    report(f"Final unique miRNAs:                       {final['mirna_id_mapped'].nunique()}")
    report(f"Final unique genes:                        {final['target_gene'].nunique()}")
    report("Pairs by evidence_level:")
    for k, v in final["evidence_level"].value_counts().items():
        report(f"  {k:<20}{v:>9}")
    report("Pairs by map_rule (merged pairs may list several rules):")
    for k, v in final["map_rule"].value_counts().items():
        report(f"  {k:<30}{v:>9}")
    report()

    report("=" * 70)
    report("Shared-sequence groups (mature.fa)")
    report("=" * 70)
    n_shared_groups = (mirnas["seq_group"].value_counts() > 1).sum()
    report(f"seq_group ids: {mirnas['seq_group'].nunique()} "
           f"({n_shared_groups} groups with >1 miRNA, covering {len(shared_ids)} miRNAs)")
    in_shared = final["mirna_id_mapped"].isin(shared_ids)
    report(f"Final pairs whose miRNA is in a shared-sequence group: {in_shared.sum()} "
           f"({final.loc[in_shared, 'mirna_id_mapped'].nunique()} miRNAs)")
    report("By evidence_level:")
    for k, v in final.loc[in_shared, "evidence_level"].value_counts().items():
        report(f"  {k:<20}{v:>9}")

    report.save(paths["results_dir"] / "step1b_report.txt")
    print(f"\nSaved: data/interim/mirna.csv, data/interim/interactions_mapped.csv, "
          f"results/step1b_report.txt")


if __name__ == "__main__":
    main()
