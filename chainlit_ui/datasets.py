"""Dataset-aware suggested-prompts for the Chainlit welcome page.

The list of AD/ADRD dataset IDs and their associated example prompts is
declarative data. Keeping it (and the function that filters by which
datasets are actually present on disk) in its own module makes it easy
to extend with new datasets without touching the UI orchestration code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Prompt templates keyed by dataset id — shown only when those files are
# locally present. Each entry is (dataset_id, prompt_text, category) where
# `dataset_id` is matched against subdirectory names of the AD data lake
# by exact equality (see `discover_present_dataset_ids`).
AD_DATASET_PROMPTS: list[tuple[str, str, str]] = [
    (
        "GCST90027158",
        "Map the top 10 AD GWAS loci from Bellenguez 2022 (GCST90027158) to nearby genes and report their putative functions",
        "GWAS",
    ),
    (
        "NG00052",
        "What are the top GWAS hits for CSF clusterin levels in the NG00052 dataset? Which of these overlap known AD risk loci?",
        "GWAS",
    ),
    (
        "NG00075",
        "Extract genome-wide significant hits from the Kunkle 2019 IGAP stage-2 summary stats (NG00075) and annotate them with gene names",
        "GWAS",
    ),
    (
        "NG00102",
        "Which proteins are measured across CSF, plasma, and brain tissue in the SomaScan 1.3k proteomic panel (NG00102)? Find any shared with known AD biomarkers",
        "Proteomics",
    ),
    (
        "NG00105",
        "Identify the top eQTL genes in prefrontal cortex (MFG) from NG00105 that overlap AD GWAS loci — load the cis-QTL file and filter by FDR < 0.05",
        "QTL",
    ),
    (
        "NG00118",
        "Find structural variant eQTLs in ROSMAP DLPFC (NG00118) for BIN1 and CLU — do they co-localize with GWAS signals?",
        "QTL",
    ),
    (
        "NG00126",
        "What rare coding variants reach exome-wide significance in the ADSP European WES dataset (NG00126)?",
        "Rare variants",
    ),
    (
        "NG00133",
        "Analyze the plasma and urine biomarker data from NG00133 — which analytes differ most between AD cases and controls?",
        "Biomarkers",
    ),
    (
        "NG00148",
        "Compare T-cell receptor CDR3 sequences between AD brain and blood samples using the NG00148 data",
        "Immunogenomics",
    ),
    (
        "NG00165",
        "Run a gene-level burden analysis summary using the CHARGE/ADSP 5k WGS results (NG00165) — list top gene hits from SKAT and CMC tests",
        "Rare variants",
    ),
    (
        "NG00166",
        "Which coding and non-coding rare variants are most significant in African American ancestry from ADSP R3 WGS (NG00166)?",
        "Rare variants",
    ),
    ("NG00172", "Summarize the structural variant associations with AD risk from NG00172", "Rare variants"),
    (
        "NG00180",
        "Identify metabolites whose MWAS weights (NG00180) are most enriched in AD-related pathways — use the EUR metabolite feature table",
        "Metabolomics",
    ),
    (
        "RADR",
        "Look up all TREM2 and APOE rare variants in the RADR database (RADR_V3.xlsx) and report their clinical classifications",
        "Rare variants",
    ),
    (
        "SingleBrain",
        "Find microglia-specific eQTLs from SingleBrain that co-localize with AD GWAS loci — load the MG top-association files",
        "QTL",
    ),
    (
        "isoMiGA_QTL",
        "Map isoMiGA microglia splicing QTLs (sQTLs) to the BIN1 and PTK2B loci — load union_leafcutter_top_assoc.tsv.gz",
        "QTL",
    ),
    (
        "isoMiGA_counts",
        "Compare microglia gene expression (TPM) for TREM2, CX3CR1, and P2RY12 across cohorts using isoMiGA count matrices",
        "Expression",
    ),
]


def discover_present_dataset_ids(ad_lake: Path) -> set[str]:
    """Return the set of dataset directory names under `ad_lake` that contain
    at least one non-README file.

    A dataset is considered "present" iff its directory exists AND has at
    least one regular file whose name does not start with `readme` (case-
    insensitive). Empty directories and README-only directories are skipped
    so the welcome page doesn't suggest prompts for datasets the user hasn't
    actually downloaded data for.
    """
    if not ad_lake.is_dir():
        return set()

    present: set[str] = set()
    for child in ad_lake.iterdir():
        if not child.is_dir():
            continue
        if any(f.is_file() and not f.name.lower().startswith("readme") for f in child.iterdir()):
            present.add(child.name)
    return present


def build_suggested_prompts_markdown(ad_lake: Path) -> str:
    """Generate the suggested-prompts markdown block for the welcome page.

    Returns an empty string when the AD data lake is missing or no
    datasets with local files match an entry in `AD_DATASET_PROMPTS`,
    so callers can no-op cleanly.
    """
    present_ids = discover_present_dataset_ids(ad_lake)
    if not present_ids:
        return ""

    by_category: dict[str, list[str]] = defaultdict(list)
    for ds_id, prompt_text, category in AD_DATASET_PROMPTS:
        if ds_id in present_ids:
            by_category[category].append(prompt_text)

    if not by_category:
        return ""

    lines = ["**Suggested prompts based on your local data:**", ""]
    for category, prompts in by_category.items():
        lines.append(f"*{category}*")
        for p in prompts:
            lines.append(f'- *"{p}"*')
        lines.append("")

    return "\n".join(lines).rstrip()
