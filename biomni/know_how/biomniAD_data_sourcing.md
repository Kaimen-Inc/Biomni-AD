# BiomniAD Data Sourcing (Alzheimer's & Related Dementias)

Short, actionable guidance for AD/dementia tasks that may benefit from additional data.

---

## When to use

Trigger keywords: "Alzheimer", "AD", "dementia", "MCI", "amyloid", "tau", "neurodegeneration", "cognition", or any ADRD-related cohort name.

## Step-by-step agent instructions

1. **Scan BIOMNI_DATA_PATH first** - List the root of the configured data directory. Datasets may live directly under it (e.g. `/mnt/dataset1/files`). Print what you find.
2. **Scan local JSON catalogs** - Load every JSON file matching `resource/BiomniAD*.json`, `resource/NIAGADS*.json`, and `resource/SinaiADRD.json` from the know-how resource directory. Each entry has: `id`, `title`, `dataset_url`, `open_access_portal_url`, optional `manifest_url`, and `files[]` with per-file `uri` download links.
3. **Summarise** - Print relevant dataset names, modality, and a one-line recommendation for the current task.
4. **Download only what is needed** - If a dataset materially improves the task, download the smallest useful subset to the data path. Cache the path and reference it in subsequent tool calls.
5. **Never fabricate data.** If local data is missing, explicitly state the gap.

### How to load catalogs (reference code)

```python
import os, glob, json
resource_dir = os.path.join(os.path.dirname(__file__), "resource")
catalog_paths = (
    glob.glob(os.path.join(resource_dir, "BiomniAD*.json"))
    + glob.glob(os.path.join(resource_dir, "NIAGADS*.json"))
    + glob.glob(os.path.join(resource_dir, "SinaiADRD.json"))
)
datasets = []
for p in catalog_paths:
    with open(p) as f:
        data = json.load(f)
    datasets.extend(data.get("datasets", []))
print(f"{len(datasets)} datasets across {len(catalog_paths)} catalogs")
for d in datasets:
    print(f"  [{d['id']}] {d['title']}")
    for fi in d.get("files", []):
        print(f"      -> {fi['name']}: {fi['uri']}")
```

## Available AD datasets (quick reference, check json files for more details)

### NIAGADS (NG*) - genetics / omics / biomarkers

| ID | Title (short) | Modality |
|----|---------------|----------|
| NG00049 | CSF Summary Statistics (Cruchaga 2013) | GWAS sumstats |
| NG00052 | CLU Endophenotype (Deming 2016) | GWAS sumstats |
| NG00067 | ADSP Umbrella | WGS / WES |
| NG00075 | IGAP Rare Variants (Kunkle 2019) | Rare-variant sumstats |
| NG00100 | AD Risk Loci in African Americans (Kunkle 2021) | GWAS sumstats |
| NG00102 | Genomic Atlas of the Proteome (Brain/CSF/Plasma) | pQTL |
| NG00103 | RBFOX1 and Brain Amyloidosis | Imaging genetics |
| NG00105 | MiGA Microglia Genomic Atlas | eQTL / sQTL |
| NG00118 | AMP-AD Structural Variant WGS and SV-xQTL | SV / xQTL |
| NG00126 | Variant-Level Artifacts in ADSP (Belloy 2022) | QC / variant |
| NG00133 | Resveratrol JOTROL Pharmacokinetics | Clinical trial |
| NG00148 | Rare Variant Aggregation ADSP Case-Control | Burden test |
| NG00156 | Resilience to AD Variants and Pathways | GWAS sumstats |
| NG00157 | Sex-Specific AD Biomarker Predictors (Deming 2018) | GWAS sumstats |
| NG00158 | Sex Differences in AD Pathology (Dumitrescu 2019) | GWAS sumstats |
| NG00159 | Longitudinal Memory Change Endophenotype (Archer 2023) | GWAS sumstats |
| NG00160 | Sex-Specific Late-Life Memory Architecture | GWAS sumstats |
| NG00161 | Sex Differences - Cognitive Resilience (Eissman 2022) | GWAS sumstats |
| NG00165 | CHARGE x ADSP R1 WGS (Wang 2024) | GWAS sumstats |
| NG00166 | ADSP R3 17k WGS (Lee 2023) | GWAS sumstats |
| NG00169 | PSP Summary Statistics (Farrell 2024) | GWAS sumstats |
| NG00172 | PSP SNVs/INDELs/SVs (Wang 2024) | GWAS sumstats |
| NG00175 | Targeted ATN Biomarker Proteomics in ADSP | Proteomics |
| NG00177 | Multi-Omic Endophenotype GWAS in ADSP | Multi-omic GWAS |
| NG00180 | Four Plasma pQTL and mQTL Atlases | pQTL / mQTL |
| NG00182 | CSF/Plasma ATN Biomarkers Multi-Ancestry | Biomarker GWAS |

### Sinai

| ID | Title | Modality |
|----|-------|----------|
| RADR | Repository for Rare AD/ADRD Variants | Curated variant table |
| SingleBrain | Single-nucleus eQTL Meta-analysis (Brain) | snRNA-seq eQTL |
| isoMiGA | Microglia Genomic Atlas (Expression/Splicing) | eQTL / sQTL |

### Major Open Discovery Datasets

| ID | Title | Modality |
|----|-------|----------|
| SEA-AD | Seattle AD Brain Cell Atlas | snRNA-seq, ATAC-seq, Pathology |
| ssREAD | Single-cell and Spatial RNA-seq DB for AD | scRNA-seq / spatial |
| LBA | Lysosomal Brain Atlas (2026) | Proteomics |
| OASIS-4 | Longitudinal Clinical AD Cohort | Neuroimaging / clinical |
| ABC Atlas | Allen Brain Cell Atlas | Cell census |
| HCP | Human Connectome Project | Structural / functional MRI |
| Pan-UKBB-AD | Pan-UK Biobank AD Proxy GWAS | GWAS sumstats |


## CRISPRbrain API

Brain and iPSC CRISPR screen data, accessible directly via Python:

```bash
pip install crisprbrain
```

```python
import crisprbrain
client = crisprbrain.Client()
print("Screens:", ", ".join(client.screens.keys()))
screen = client.screens["Glutamatergic Neuron-Survival-CRISPRi"]
df = screen.to_data_frame()
print(df.describe())
```

## Decision rule

- Download via `files[].uri` or `manifest_url`/`dataset_url` when the dataset modality aligns with the task and expected benefit > cost.
- Prefer the smallest relevant subset first; escalate to larger downloads only if needed.

## Bulk local download

To download all BiomniAD catalog files less than 100MB locally for the agent:

**Standalone Python:**
```python
from biomni.agent.ad_data_downloader import download_ad_catalog_data
# Specify your local data lake path
results = download_ad_catalog_data("/path/to/data_lake")
print(f"Downloaded {len(results['downloaded'])} files.")
```

**Via AD1 Agent:**
```python
from biomni.agent.ad1 import AD1
# This will automatically download/cache missing files < 100MB during initialization
agent = AD1(download_ad_data=True)
```

Files are stored in `<data_lake>/biomniAD/<dataset_id>/` and are automatically annotated as `[LOCAL]` in the agent's sourcing instructions.

## One-liner

> "Because this is an AD/dementia task, I first scan local data (BIOMNI_DATA_PATH) and BiomniAD catalogs, identify datasets aligned to the task, and proceed accordingly."
