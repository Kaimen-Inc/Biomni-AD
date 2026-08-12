# Biomni-AD Environment Setup

This directory contains scripts and configuration files to set up a comprehensive bioinformatics environment with various tools and packages.

## Which YAML do I use?

| File | Purpose | When to use |
|------|---------|-------------|
| `environment.yml` | **Canonical minimal env** - Python 3.11 + LangChain stack + core scientific Python | First-time install, Docker build (default), CI |
| `fixed_env.yml` | Reduced full env without R / CLI tools (~13 GB) | If you want most of `bio_env.yml` but can skip R |
| `bio_env.yml` | Layered on top of `environment.yml` for full bioinformatics tooling | When running `setup.sh` (auto-installs this) |
| `bio_env_py310.yml` | Python 3.10 side env named `biomni_py310` | Only for `analyze_copy_number_purity_ploidy_and_focal_events` (cnvkit needs Py 3.10) |
| `r_packages.yml` | R packages layered on top of the base env | When `setup.sh` installs R support |
| `adworkbench_env.yml` | AD Workbench-specific full env | Alternative Docker build: `--build-arg BIOMNI_ENV_FILE=biomni_env/adworkbench_env.yml` |

`environment.yml` is the **single source of truth for the minimal env**. The Dockerfile defaults to it; the lightweight `pip install -e .` path in the project root provides the same core Python deps without conda. Reach for the other YAMLs only when you need the extras they layer on.

**Biomni-AD** (this repository) is developed and maintained by **Kuan-lin Huang, PhD** at **[Kaimen Inc.](https://github.com/Kaimen-Inc/Biomni-AD)**, building on the foundational [Biomni](https://github.com/snap-stanford/Biomni) platform by Stanford's SNAP Lab.

## Branch Guide

| Branch | Purpose |
|--------|---------|
| [`feat/adworkbench`](https://github.com/Kaimen-Inc/Biomni-AD/tree/feat/adworkbench) | **Recommended install branch** - AD Workbench integration with tighter dataset integration and containerization that works broadly. |
| [`biomni-ad`](https://github.com/Kaimen-Inc/Biomni-AD/tree/biomni-ad) | **Primary stable branch** - Biomni-AD specialization without AD Workbench-specific deployment features. |
| `main` | Upstream [Stanford SNAP Biomni](https://github.com/snap-stanford/Biomni). Periodically merged into `biomni-ad`. Read-only from this fork. |

Install from the `feat/adworkbench` branch (recommended):

```bash
pip install git+https://github.com/Kaimen-Inc/Biomni-AD.git@feat/adworkbench
```

## AD Data Lake

Biomni-AD includes three JSON catalogs (NIAGADS, SinaiADRD, BiomniAD Discovery) covering hundreds of AD/ADRD datasets. Files ≤ 100 MB are downloaded automatically on first `AD1` initialization; larger files are accessed via catalog URIs or external portals.

See the **[AD Data Lake section in README.md](../README.md#ad-data-lake)** for full download options and catalog details.

## Environment Installation

1. Clone the repository:

   **Biomni-AD (this fork - recommended for AD research):**
   ```bash
   git clone https://github.com/Kaimen-Inc/Biomni-AD.git
   cd Biomni-AD/biomni_env
   ```

   **Upstream Biomni (Stanford SNAP Lab):**
   ```bash
   git clone https://github.com/snap-stanford/Biomni.git
   cd Biomni/biomni_env
   ```

2. Setting up the environment:
- (a) If you want to use or try out the basic agent without the full E1 or install your own softwares, run the following script:

```bash
conda env create -f environment.yml
```

- (b) If you want to use the full environment E1, run the setup script (this script takes > 10 hours to setup, and requires a disk of at least 30 GB quota). Follow the prompts to install the desired components.

```bash
bash setup.sh
```

If you already installed the base version, and just wants to add the additional packages in the new release, you can simply do:

```bash
bash new_software_v008.sh
```

Note: we have only tested this setup.sh script with Ubuntu 22.04, 64 bit.

- (c) If you want to use a reduced conda environment without R or CLI tools, run the following script:

```bash
conda env create -f fixed_env.yml
```

This contains most of the packages from environment.yml and bio_env.yml, and requires a disk of at elast 13GB quota.

- (d) **Python 3.10 Environment for Copy Number Analysis**: If you specifically need to use the `analyze_copy_number_purity_ploidy_and_focal_events` function, we provide a Python 3.10 environment option. This function has specific dependency requirements that are best met with Python 3.10. To set up this environment:

```bash
conda env create -f bio_env_py310.yml
```

This environment is optimized for copy number variation analysis and includes the necessary packages for purity, ploidy, and focal event detection.

3. Lastly, to activate the biomni environment:
```bash
conda activate biomni_e1
```

For the Python 3.10 environment specifically:
```bash
conda activate biomni_py310
```

### 📦 Langchain Package Support

The Biomni environment comes with a minimal set of langchain packages by default:
- `langchain-openai` - for OpenAI model support
- `langchain-anthropic` - for Anthropic model support
- `langchain-ollama` - for Ollama model support

If you need support for other external models or services, you'll need to install additional langchain packages manually. For example:

```bash
# For AWS Bedrock support
pip install langchain-aws

# For Google Gemini support
pip install langchain-google-genai

```
