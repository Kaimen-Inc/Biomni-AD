<p align="center">
  <img src="./figs/Biomni-AD_Logo_v2.png" alt="Biomni-AD Logo" width="560px" />
</p>
<p align="center">
  <sub>Specialized fork of&nbsp;<a href="https://github.com/snap-stanford/Biomni"><img src="./figs/biomni_logo.png" alt="Biomni" height="18" /></a>&nbsp;·&nbsp;Stanford SNAP Lab</sub>
</p>

<p align="center">
<a href="https://join.slack.com/t/biomnigroup/shared_invite/zt-3avks4913-dotMBt8D_apQnJ3mG~ak6Q">
<img src="https://img.shields.io/badge/Join-Slack-4A154B?style=for-the-badge&logo=slack" alt="Join Slack" />
</a>
<a href="https://biomni.stanford.edu">
<img src="https://img.shields.io/badge/Try-Web%20UI-blue?style=for-the-badge" alt="Web UI" />
</a>
<a href="https://x.com/ProjectBiomni">
<img src="https://img.shields.io/badge/Follow-on%20X-black?style=for-the-badge&logo=x" alt="Follow on X" />
</a>
<a href="https://www.linkedin.com/company/project-biomni">
<img src="https://img.shields.io/badge/Follow-LinkedIn-0077B5?style=for-the-badge&logo=linkedin" alt="Follow on LinkedIn" />
</a>
<a href="https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1">
<img src="https://img.shields.io/badge/Read-Paper-green?style=for-the-badge" alt="Paper" />
</a>
</p>



# Biomni-AD: AI Co-Scientist for Biomedical Research, integrated with Alzheimer's Disease Datalake

## Overview

**Biomni-AD** is an Alzheimer's disease-specialized extension of [Biomni](https://github.com/snap-stanford/Biomni) (Stanford SNAP Lab), developed and maintained by **Kuan-lin Huang, PhD** at **[Kaimen Inc.](https://github.com/Kaimen-Inc/Biomni-AD)** It adds the **AD1 agent** - a domain-expert variant of the general A1 agent - along with an AD-focused data lake, curated dataset catalogs (NIAGADS, SinaiADRD, CRISPRbrain), and a plan-then-approve Chainlit UI optimized for neurodegeneration research workflows.

The underlying **Biomni** platform is a general-purpose biomedical AI agent that integrates LLM reasoning with retrieval-augmented planning and code-based execution to help scientists enhance research productivity and generate testable hypotheses.

**Our commitment.** Biomni-AD will remain fully open source, and we are working to deploy it on the **Alzheimer's Disease Data Initiative (ADDI)** platform so it can serve as many AD researchers as possible and accelerate progress against Alzheimer's disease and related dementias.

## Branch Guide

| Branch | Purpose |
|--------|---------|
| [`feat/adworkbench`](https://github.com/Kaimen-Inc/Biomni-AD/tree/feat/adworkbench) | **Recommended install branch** - Extends `biomni-ad` with AD Workbench dataset integration and containerization. Install with `pip install git+https://github.com/Kaimen-Inc/Biomni-AD.git@feat/adworkbench`. |
| [`biomni-ad`](https://github.com/Kaimen-Inc/Biomni-AD/tree/biomni-ad) | **Primary stable branch** - Biomni-AD specialization without AD Workbench-specific deployment features. |
| `main` | Upstream [Stanford SNAP Biomni](https://github.com/snap-stanford/Biomni). Periodically merged into `biomni-ad` to track upstream. Read-only from this fork. |

## Documentation Index

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - quick start, usage, and feature overview |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, agent framework, tool ecosystem, and data lake |
| [CONTRIBUTION.md](CONTRIBUTION.md) | How to contribute tools, data, software, benchmarks, and know-how |
| [DETAILS.md](DETAILS.md) | Technical reference: module roles, code organization, and entry points |
| [chainlit.md.template](chainlit.md.template) | Chainlit welcome-page template (rendered to `chainlit.md` on launch with the local data inventory; `chainlit.md` itself is gitignored) |
| [biomni_env/README.md](biomni_env/README.md) | Environment installation instructions |
| [docs/configuration.md](docs/configuration.md) | Configuration management guide |
| [docs/known_conflicts.md](docs/known_conflicts.md) | Known package conflicts and workarounds |
| [docs/docker_vm_deployment.md](docs/docker_vm_deployment.md) | Docker and VM deployment guide |
| [docs/mcp_integration.md](docs/mcp_integration.md) | Model Context Protocol (MCP) server integration |
| [docs/building_documentation.md](docs/building_documentation.md) | Building Sphinx API documentation |


## Quick Start

### Installation

**Step 1 - Set up the environment**

The Biomni environment includes 200+ scientific Python packages, R packages, and CLI bioinformatics tools. Follow [biomni_env/README.md](biomni_env/README.md) to run the setup script (choose the option that fits your needs).

**Step 2 - Activate the environment**

```bash
conda activate biomni_e1
```

**Step 3 - Install the Biomni-AD package**

Two install paths - pick the one that fits your needs.

**Full conda env** (recommended if you want all 22 bioinformatics tool modules and R support - what `Step 2` set up):

```bash
pip install git+https://github.com/Kaimen-Inc/Biomni-AD.git@feat/adworkbench
```

**Lightweight pip-only** (agent core + LangChain stack, no conda required - good for notebooks, CI, or container images):

```bash
git clone https://github.com/Kaimen-Inc/Biomni-AD.git
cd Biomni-AD
pip install -e .                  # core: LangChain + OpenAI provider
pip install -e ".[anthropic]"     # add Claude (Anthropic) provider
pip install -e ".[all]"           # all provider + UI extras
```

Available extras: `anthropic`, `bedrock`, `ollama`, `gradio`, `chainlit`, `all`.

Or install the latest stable upstream release from PyPI (Biomni without the AD specialisation):

```bash
pip install biomni --upgrade
```

**Step 4 - Configure your API keys**

Choose one of the two methods below:

<details>
<summary>Click to expand API key setup options</summary>

#### Option 1: .env file (Recommended)

```bash
cp .env.example .env
# Then open .env and fill in your API keys
```

Your `.env` file should look like:

```env
# Set at least ONE provider profile below (leave unused keys empty)

# Profile A: Anthropic direct
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Optional custom Anthropic endpoint
# ANTHROPIC_BASE_URL=https://api.anthropic.com

# Profile B: OpenAI direct
OPENAI_API_KEY=your_openai_api_key_here
# Optional custom OpenAI-compatible endpoint
# OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Azure Anthropic (if using Claude via Azure AI Foundry)
ENDPOINT_URL=https://your-resource.services.ai.azure.com/anthropic/
DEPLOYMENT_NAME=your_claude_deployment_name
AZURE_ANTHROPIC_API_KEY=your_azure_anthropic_api_key

# Optional: Azure OpenAI (if using GPT via Azure OpenAI)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Optional: AI Studio Gemini API Key (if using Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: groq API Key (if using groq as model provider)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set the source of your LLM for example:
#"OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"
LLM_SOURCE=your_LLM_source_here
# BIOMNI_SOURCE is also accepted for backward compatibility
# BIOMNI_SOURCE=your_LLM_source_here

# Optional: AWS Bedrock Configuration (if using AWS Bedrock models)
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1

# Optional: Custom model serving configuration
# CUSTOM_MODEL_BASE_URL=http://localhost:8000/v1
# CUSTOM_MODEL_API_KEY=your_custom_api_key_here

# Optional: Biomni data path (defaults to ./data)
# BIOMNI_DATA_PATH=/path/to/your/data

# Optional: Timeout settings (defaults to 600 seconds)
# BIOMNI_TIMEOUT_SECONDS=600

# Optional: Auto-switch to local-first mode when network/API calls fail (default: true)
# BIOMNI_AUTO_NETWORK_LIMITED_MODE=true
```

#### Option 2: Shell environment variables

Add to your `~/.bashrc` (or `~/.zshrc`):

```bash
# Required - at least one LLM provider key:
export ANTHROPIC_API_KEY="your_key"   # Claude models
export OPENAI_API_KEY="your_key"      # GPT models (optional)
export GEMINI_API_KEY="your_key"      # Gemini models (optional)
export GROQ_API_KEY="your_key"        # Groq models (optional)

# Optional custom endpoints for direct providers:
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Azure OpenAI (optional):
export OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your_key"

# Azure Anthropic (optional):
export ENDPOINT_URL="https://your-resource.services.ai.azure.com/anthropic/"
export DEPLOYMENT_NAME="your_claude_deployment_name"
export AZURE_ANTHROPIC_API_KEY="your_key"

# AWS Bedrock (optional):
export AWS_BEARER_TOKEN_BEDROCK="your_key"
export AWS_REGION="us-east-1"
```
</details>


#### ⚠️ Known Package Conflicts

Some Python packages are not installed by default in the Biomni environment due to dependency conflicts. If you need these features, you must install the packages manually and may need to uncomment relevant code in the codebase. See the up-to-date list and details in [docs/known_conflicts.md](./docs/known_conflicts.md).

### AD Data Lake

Biomni-AD ships three JSON catalogs - **NIAGADS**, **SinaiADRD**, and **BiomniAD Discovery** - that describe hundreds of AD/ADRD datasets. Files ≤ 100 MB are downloaded automatically to disk; larger or controlled-access files are referenced by catalog URI for on-demand access.

| Catalog | Contents | Access |
|---------|----------|--------|
| NIAGADS (`NG*`) | Genetics, omics, biomarkers - ADSP WGS/WES, pQTL, eQTL, CSF sumstats | Controlled + open |
| SinaiADRD | Rare variants (RADR), single-nucleus eQTL (SingleBrain), microglia expression (isoMiGA) | Open |
| BiomniAD Discovery | SEA-AD, ssREAD, OASIS-4, HCP, ABC Atlas | Open |
| CRISPRbrain | CRISPR screens in neurons and microglia | Open API |

Downloaded files are cached in `<data_lake>/biomniAD/<dataset_id>/` and skipped on re-runs. Set `BIOMNI_DATA_LAKE_PATH` to control where that `<data_lake>` root lives (defaults to the repo-local `data/biomni_data/data_lake` folder).

**Option A - Skip local download, use catalog URIs and internet (default)**

```python
from biomni import AD1   # short top-level import; equivalent to `from biomni.agent.ad1 import AD1`

agent = AD1()   # download_ad_data defaults to False; datasets are fetched on demand as queries need them
```

**Option B - Bulk download on AD1 init**

```python
agent = AD1(download_ad_data=True)   # downloads files ≤ 100 MB up front, before the first query
```

**Option C - Bulk download without starting an agent**

```python
from biomni.agent.ad_data_downloader import download_ad_catalog_data

download_ad_catalog_data("/path/to/your/data_lake")
```

The agent still references catalog URIs in its system prompt and can fetch data on demand or direct you to the relevant portal (e.g., NIAGADS DAC for controlled-access datasets).

### Basic Usage & Agent Selection

Biomni-AD provides two agents:

- **AD1 (Alzheimer's Disease Agent)**: The primary agent for this fork - specialized for Alzheimer's and dementia research with AD-focused data sourcing, context-aware neurodegeneration instructions, and optimized tool selection.
- **A1 (General Agent)**: The upstream general-purpose biomedical agent. For general biomedical use without AD specialization, see the [upstream Biomni project](https://github.com/snap-stanford/Biomni).

#### 1. Chainlit Interactive UI - Default for Biomni-AD

The recommended way to run Biomni-AD is the **Chainlit UI** with its **plan-then-approve** workflow:

1. The AD1 agent generates a numbered research plan before executing.
2. You choose **Approve & Execute**, **Revise Plan**, or **Cancel**.
3. The full ReAct loop runs with each step shown as a collapsible trace (Thinking → Code → Observation → Answer).

**Setup (one-time):**
```bash
conda activate biomni_e1
pip install "chainlit>=1.0"
```

**Single instance:**
```bash
bash run_chainlit.sh                   # opens http://localhost:8000
bash run_chainlit.sh --port 8080       # custom port
bash run_chainlit.sh --headless        # no browser auto-open (servers/CI)
```

> **Note:** Always use `bash run_chainlit.sh` - not `chainlit run chainlit_app.py` directly. The script ensures the correct `biomni_e1` Python is used even when another virtual environment (`.venv`) is active in the same shell.

**Fleet deployment (multiple instances):**
```bash
bash scripts/launch_biomniAD_fleet.sh  # launches and manages a fleet of Biomni-AD instances
```

**Environment variables (optional):**

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOMNI_LLM` | `claude-sonnet-4-5` | LLM model used by both agents |
| `BIOMNI_PATH` | `./data` | Data directory for the agent |

#### 2. Running in Notebooks or CLI

**AD1 (Alzheimer's Specialized):**
```python
from biomni.agent.ad1 import AD1

agent = AD1(llm='claude-sonnet-4-5')
agent.go("Analyze Tau aggregation pathways and suggest potential inhibitors.")
```

**A1 (General - upstream Biomni):**
```python
from biomni.agent import A1

agent = A1(llm='claude-sonnet-4-5')
agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion.")
```

#### 3. Gradio UI

```python
# AD1
from biomni.agent.ad1 import AD1
AD1().launch_ui()

# A1 (general)
from biomni.agent import A1
A1().launch_gradio_demo()
```

Install Gradio 5.x first: `pip install "gradio>=5.0,<6.0"`

UI options: `share=True` (public link) · `server_name="127.0.0.1"` (localhost only) · `require_verification=True` (access code, default `"Biomni2025"`)

#### 4. Docker Deployment (Local or VM)

Biomni can be run as a containerized service with external access:

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Then open `http://localhost:8000` (or your VM public IP).

For full VM deployment instructions (firewall/security group, operations, and hardening), see [docs/docker_vm_deployment.md](docs/docker_vm_deployment.md).

#### Controlling Datalake Loading

Biomni does not download the full datalake (~11GB) when you create an agent.
Individual files are fetched lazily, the first time a query actually needs them - a session that never touches DepMap never pays to fetch it.
Set `BIOMNI_DATA_LAKE_PATH` if the datalake should live somewhere other than the repo-local `data/` folder, e.g. a dedicated volume or a different disk on the server.

```python
# Pre-fetch specific files now instead of waiting for a query that needs them
agent = A1(path='./data', llm='claude-sonnet-4-20250514', expected_data_lake_files=['gene_info.parquet'])
```

This is useful for:
- Warming the cache before a demo, so the first query isn't slowed down by a download
- Environments with limited or metered bandwidth, where you want to control exactly what gets fetched and when
If you plan on using Azure for your model, always prefix the model name with azure- (e.g. llm='azure-gpt-4o').


### Configuration Management

Biomni includes a centralized configuration system that provides flexible ways to manage settings. You can configure Biomni through environment variables, runtime modifications, or direct parameters.

```python
from biomni.config import default_config
from biomni.agent import A1

# RECOMMENDED: Modify global defaults for consistency
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200

# All agents AND database queries use these defaults
agent = A1()  # Everything uses gpt-4, 1200s timeout
```

**Note**: Direct parameters to `A1()` only affect that agent's reasoning, not database queries. For consistent configuration across all operations, use `default_config` or environment variables.

For detailed configuration options, see the **[Configuration Guide](docs/configuration.md)**.

### PDF Generation

Generate PDF reports of execution traces:

```python
from biomni.agent import A1

# Initialize agent
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# Run your task
agent.go("Your biomedical task here")

# Save conversation history as PDF
agent.save_conversation_history("my_analysis_results.pdf")
```

**PDF Generation Dependencies:**
<details>
<summary>Click to expand</summary>
For optimal PDF generation, install one of these packages:

```bash
# Option 1: WeasyPrint (recommended for best layout control)
# Conda environment (recommended)
conda install weasyprint

# System installation
brew install weasyprint  # macOS
apt install weasyprint   # Linux

# See [WeasyPrint Installation Guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html) for detailed instructions.

# Option 2: markdown2pdf (Rust-based, fast and reliable)
# macOS:
brew install theiskaa/tap/markdown2pdf

# Windows/Linux (using Cargo):
cargo install markdown2pdf

# Or download prebuilt binaries from:
# https://github.com/theiskaa/markdown2pdf/releases/latest

# Option 3: Pandoc (pip installation)
pip install pandoc
```
</details>

## MCP (Model Context Protocol) Support

Biomni-AD supports MCP servers for external tool integration:

```python
from biomni.agent.ad1 import AD1

agent = AD1()
agent.add_mcp(config_path="./mcp_config.yaml")
agent.go("Find FDA active ingredient information for donepezil")
```

For usage and implementation details, see the [MCP Integration Documentation](docs/mcp_integration.md) and examples in [`tutorials/examples/add_mcp_server/`](tutorials/examples/add_mcp_server/) and [`tutorials/examples/expose_biomni_server/`](tutorials/examples/expose_biomni_server/).

## Upstream Biomni Capabilities

Biomni-AD is a specialized fork of [Biomni](https://github.com/snap-stanford/Biomni) by Stanford's SNAP Lab. All upstream Biomni capabilities remain available - including 30+ biomedical tool domains, the Biomni-R0 reasoning model, the Biomni-Eval1 benchmark, the Know-How Library, and MCP integration. For features, models, and benchmarks not specific to Alzheimer's disease, refer to the upstream project directly:

- **Biomni-R0** reasoning model: [biomni/Biomni-R0-32B-Preview](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- **Biomni-Eval1** benchmark: [biomni/Eval1](https://huggingface.co/datasets/biomni/Eval1)
- **Know-How Library**: curated lab protocols and best practices auto-retrieved by the agent (see `biomni/know_how/`)

For general-purpose biomedical AI agent use not focused on Alzheimer's disease, use the upstream [Biomni project](https://github.com/snap-stanford/Biomni) directly.

## Tutorials

**[Biomni 101](./tutorials/biomni_101.ipynb)** - basic concepts and first steps (upstream Biomni).

AD-specific tutorials and example notebooks live alongside the AD1 agent in this repository.

## Maintainership, Scope, and Contributions

Biomni-AD is maintained by **Kuan-lin Huang, PhD** at **[Kaimen Inc.](https://github.com/Kaimen-Inc/Biomni-AD)** (`https://github.com/Kaimen-Inc/Biomni-AD.git`) as a focused, AD-specific extension of upstream Biomni.

**Open source and access commitments:**
- The Biomni-AD codebase, AD1 agent, data lake catalogs, and Chainlit UI will remain fully open source.
- We are working to deploy Biomni-AD on the **Alzheimer's Disease Data Initiative (ADDI)** platform so AD researchers worldwide can use it to advance research without needing to self-host.

This repository is **not** a general open-science platform and is not soliciting community contributions, co-author tool submissions, or paper-credit programs. For those, please engage with the upstream [Biomni](https://github.com/snap-stanford/Biomni) project.

Bug reports and targeted pull requests against the AD-specific code paths (AD1 agent, AD data lake catalogs, Chainlit UI) are welcome via GitHub issues.

## Important Notes

- **Security warning**: Biomni-AD executes LLM-generated code with full system privileges. For production or shared use, run inside an isolated/sandboxed environment. The agent can access files, the network, and system commands - be careful with sensitive data or credentials.
- **Controlled-access data**: NIAGADS and other catalog entries marked as controlled-access require independent authorization (e.g., NIAGADS DAC). Biomni-AD does not bypass access controls; the agent will reference catalog URIs and direct you to the appropriate portal.
- **Licensing**: Biomni-AD inherits upstream Biomni's Apache 2.0 license, but certain integrated tools, databases, or software may carry more restrictive licenses. Review each component before any commercial use.

## Citation

Biomni-AD builds on upstream Biomni. Please cite the original Biomni paper:

```
@article{huang2025biomni,
  title={Biomni: A General-Purpose Biomedical AI Agent},
  author={Huang, Kexin and Zhang, Serena and Wang, Hanchen and Qu, Yuanhao and Lu, Yingzhou and Roohani, Yusuf and Li, Ryan and Qiu, Lin and Zhang, Junze and Di, Yin and others},
  journal={bioRxiv},
  pages={2025--05},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

If you use Biomni-AD specifically (AD1 agent, AD data lake, or Chainlit workflow), please also credit this repository: *Biomni-AD, Kuan-lin Huang, Kaimen Inc. - https://github.com/Kaimen-Inc/Biomni-AD*
