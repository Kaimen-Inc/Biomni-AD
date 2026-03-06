# DETAILS.md — Biomni-AD Technical Reference


**Related docs:** [README.md](README.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [CONTRIBUTION.md](CONTRIBUTION.md)

---

## 1. Project Overview

### Purpose

Biomni-AD is a **biomedical AI agent platform** specialized for Alzheimer's disease and neurodegeneration research. It provides:

- Autonomous execution of complex AD research tasks via LLM-powered agents
- 180+ domain-specific biomedical tools across 20 subfields
- 53 natural-language interfaces to major biomedical databases
- A curated data lake (~11GB, 77 files) with AD-specific catalogs
- A Know-How Library of curated protocols and best practices
- An interactive Chainlit UI with plan-then-approve workflow
- Multi-provider LLM support (Claude, GPT, Gemini, Bedrock, Ollama, Groq)

### Target Users

- Biomedical researchers automating literature mining, database queries, and analysis workflows
- Bioinformaticians requiring integrated access to biological databases and computational tools
- AI researchers applying LLMs and agents to biomedical problem-solving
- AD/dementia researchers needing specialized data sourcing and analysis pipelines

---

## 2. Repository Structure

```
Biomni/
├── biomni/                        # Main library package
│   ├── agent/
│   │   ├── a1.py                  # A1: general-purpose biomedical agent (~3000 lines)
│   │   ├── ad1.py                 # AD1: Alzheimer's specialist agent (extends A1)
│   │   ├── react.py               # ReAct engine (LangGraph state machine)
│   │   ├── env_collection.py      # Environment and data retrieval utilities
│   │   ├── function_generator.py  # Dynamic function generation
│   │   └── qa_llm.py              # Question-answering LLM wrappers
│   ├── tool/
│   │   ├── tool_description/      # Declarative tool schemas (18 files, one per domain)
│   │   ├── schema_db/             # Pickled database API schemas (25 files)
│   │   ├── biochemistry.py        # Molecular structure, protein analysis
│   │   ├── bioengineering.py      # CRISPR, synthetic biology
│   │   ├── bioimaging.py          # Microscopy, histopathology
│   │   ├── biophysics.py          # Molecular dynamics
│   │   ├── cancer_biology.py      # DepMap, oncogenomics
│   │   ├── cell_biology.py        # Single-cell analysis
│   │   ├── database.py            # 53 external database API functions
│   │   ├── genetics.py            # GWAS, variant analysis
│   │   ├── genomics.py            # NGS, sequence analysis
│   │   ├── glycoengineering.py    # Glycan analysis
│   │   ├── immunology.py          # TCR, immune profiling
│   │   ├── lab_automation.py      # PyLabRobot integration
│   │   ├── literature.py          # PubMed, arXiv search
│   │   ├── microbiology.py        # Microbiome, phylogenetics
│   │   ├── molecular_biology.py   # Cloning, primers
│   │   ├── pathology.py           # Histology analysis
│   │   ├── pharmacology.py        # Drug discovery, ADMET
│   │   ├── physiology.py          # Organ systems
│   │   ├── support_tools.py       # Python REPL, Bash, R runners
│   │   ├── synthetic_biology.py   # Plasmid design
│   │   ├── systems_biology.py     # Network analysis
│   │   └── tool_registry.py       # Tool metadata management and discovery
│   ├── model/
│   │   └── retriever.py           # LLM-powered tool/dataset selection
│   ├── task/
│   │   ├── base_task.py           # Abstract benchmark task interface
│   │   ├── hle.py                 # Humanity's Last Exam benchmark
│   │   └── lab_bench.py           # Lab bench dataset evaluation
│   ├── eval/                      # BiomniEval1 evaluation framework
│   ├── biorxiv_scripts/           # Literature mining pipelines
│   ├── know_how/
│   │   ├── loader.py              # Know-How retrieval logic
│   │   ├── single_cell_annotation.md    # scRNA-seq best practices
│   │   ├── sgRNA_design_guide.md        # CRISPR guide RNA design
│   │   ├── biomniAD_data_sourcing.md    # AD data access guide
│   │   └── resource/
│   │       ├── NIAGADS_datasets_with_files.json  # 27 AD genetics datasets
│   │       ├── SinaiADRD.json                     # 5 Sinai AD datasets
│   │       ├── CRISPick_download_links.txt        # CRISPR resources
│   │       └── addgene_grna_sequences.csv         # sgRNA library
│   ├── config.py                  # Centralized configuration (BiomniConfig dataclass)
│   ├── env_desc.py                # Data lake dictionary (~77 files, 11GB)
│   ├── env_desc_cm.py             # Commercial-mode data lake variant
│   ├── llm.py                     # Multi-provider LLM factory
│   ├── utils.py                   # Utility functions
│   └── version.py                 # Package version (0.0.8)
│
├── chainlit_app.py                # Chainlit UI entry point (plan-then-approve workflow)
├── run_chainlit.sh                # Chainlit launcher script
├── chainlit.md                    # Chainlit welcome page content
├── docker/                        # Docker entrypoint scripts
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Micromamba-based container image
├── pyproject.toml                 # Package metadata (Python >=3.11, Apache 2.0)
├── biomni_env/
│   ├── README.md                  # Environment setup instructions
│   ├── environment.yml            # Base conda environment
│   ├── bio_env.yml                # Full environment with R (200+ packages)
│   ├── bio_env_py310.yml          # Python 3.10 environment for cnvkit
│   ├── r_packages.yml             # R package specifications
│   ├── setup.sh                   # Main setup script (>10 hours)
│   ├── install_cli_tools.sh       # Bioinformatics CLI tool installer
│   └── cli_tools_config.json      # CLI tools configuration
├── tutorials/
│   ├── biomni_101.ipynb           # Getting started notebook
│   └── examples/                  # Use case examples (MCP, cloning, etc.)
├── data/
│   └── biomni_data/
│       ├── data_lake/             # ~77 data files (~11GB), auto-downloaded
│       └── benchmark_data/        # Evaluation datasets
├── docs/
│   ├── configuration.md           # Configuration management guide
│   ├── known_conflicts.md         # Package conflict workarounds
│   ├── docker_vm_deployment.md    # VM/Docker deployment guide
│   ├── mcp_integration.md         # MCP server integration guide
│   └── building_documentation.md  # Sphinx documentation build guide
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTION.md
├── DETAILS.md                     # This file
├── license_info.md                # Data source licensing for commercial use
└── LICENSE                        # Apache 2.0
```

---

## 3. Core Module Descriptions

### `biomni/agent/a1.py` — A1 General Agent

The primary agent class (~3000 lines). Manages the full task lifecycle:
- Initializes the data lake and tool registry on startup
- Selects relevant tools via the `ToolRetriever`
- Executes a LangGraph-based ReAct loop (reason → act → observe → repeat)
- Supports MCP tool integration via `add_mcp()`
- Exports execution traces as PDF via `save_conversation_history()`
- Launches Gradio and Chainlit UIs

### `biomni/agent/ad1.py` — AD1 Alzheimer's Agent

Extends A1 with AD-specific capabilities (developed by Kuan-lin Huang, PhD):
- Detects AD-related keywords (Alzheimer, dementia, MCI, amyloid, tau, etc.)
- Injects curated AD dataset catalogs into the system prompt (NIAGADS, SinaiADRD, CRISPRbrain)
- Enforces local-data-first policy via `_enforce_local_data_priority()`
- Downloads AD-specific data subsets to `data/biomniad/`
- Provides `launch_ui()` for the Chainlit plan-then-approve interface

### `biomni/agent/react.py` — ReAct Engine

Core reasoning loop built on LangGraph:
- State machine: `Agent node → Tool node → Agent node → ...`
- Handles tool call dispatch and result injection
- Applies timeout management to individual tool executions
- Supports custom callback handlers for logging

### `biomni/model/retriever.py` — Tool Retriever

LLM-powered resource selector:
- Parses user queries to identify relevant tools, datasets, and libraries
- Returns ranked lists of tools/data for inclusion in the agent's context
- Uses Anthropic or OpenAI LLMs for selection

### `biomni/config.py` — Configuration

`BiomniConfig` dataclass providing centralized defaults:
- `llm`: model name (default: `claude-sonnet-4-5`)
- `source`: provider (default: `Anthropic`)
- `timeout_seconds`: tool execution timeout (default: 600)
- `commercial_mode`: filter non-commercial content (default: False)
- Reads from environment variables; overridable at runtime via `default_config`

### `biomni/llm.py` — LLM Factory

Instantiates LangChain LLM objects for multiple providers:
- Anthropic (Claude), OpenAI (GPT), Azure OpenAI, Google Gemini, AWS Bedrock, Groq, Ollama, Custom (OpenAI-compatible)

### `biomni/env_desc.py` — Data Lake Registry

Contains `data_lake_dict`: a mapping of dataset names to S3 download URLs and descriptions. Drives automatic data lake initialization on first agent run.

### `biomni/know_how/` — Know-How Library

Markdown documents with curated protocols and best practices. Loaded by `loader.py` and retrieved based on query relevance. Metadata headers track authors, affiliations, license, and commercial-use eligibility.

### `chainlit_app.py` — Chainlit UI

Interactive plan-then-approve interface:
1. Agent generates a numbered research plan (3–7 steps)
2. User reviews and chooses: Approve & Execute, Revise Plan, or Cancel
3. Full ReAct loop runs with collapsible step traces (Thinking → Code → Observation → Answer)

---

## 4. Key Entry Points

**Python API (notebooks or scripts):**

```python
# General-purpose agent
from biomni.agent import A1
agent = A1(llm='claude-sonnet-4-5', path='./data')
agent.go("Plan a CRISPR screen to identify T cell exhaustion regulators")

# Alzheimer's specialist agent
from biomni.agent.ad1 import AD1
agent = AD1(llm='claude-sonnet-4-5')
agent.go("Analyze APOE variants in Alzheimer's disease risk")

# Gradio web interface
A1().launch_gradio_demo()    # General agent
AD1().launch_ui()            # AD agent (Chainlit)
```

**Chainlit interactive UI (plan-then-approve):**

```bash
bash run_chainlit.sh                 # http://localhost:8000
bash run_chainlit.sh --port 8080     # custom port
bash run_chainlit.sh --headless      # no browser (servers/CI)
```

**Docker deployment:**

```bash
cp .env.example .env
docker compose build
docker compose up -d
# Access at http://localhost:8000
```

---

## 5. Development Patterns

| Pattern | Usage |
|---------|-------|
| **ReAct loop** | LangGraph state machine in `react.py` |
| **Factory** | LLM provider selection in `llm.py` |
| **Registry** | Tool discovery via `tool_registry.py` |
| **Declarative schemas** | Tool metadata in `tool_description/` separates spec from implementation |
| **Abstract base** | `base_task.py` enforces consistent benchmark interface |
| **Dataclass config** | `BiomniConfig` in `config.py` for centralized defaults |

### Code Style
- Python >=3.11, formatted with `ruff`
- Pre-commit hooks for linting and security checks (`ruff`, `bandit`)
- Functional style in tool modules (standalone functions, explicit I/O)
- Type annotations in core agent and config modules

### Testing
- No automated test suite; verification is via notebook examples and agent runs
- Each tool contribution requires a test prompt demonstrating correct agent behavior
- See `tutorials/biomni_101.ipynb` for interactive exploration

---

## 6. Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for Claude models |
| `OPENAI_API_KEY` | — | Required for OpenAI/Azure models |
| `GEMINI_API_KEY` | — | For Google Gemini models |
| `GROQ_API_KEY` | — | For Groq-hosted models |
| `AWS_BEARER_TOKEN_BEDROCK` | — | For AWS Bedrock models |
| `BIOMNI_DATA_PATH` | `./data` | Data directory for the agent |
| `BIOMNI_TIMEOUT_SECONDS` | `600` | Tool execution timeout |
| `BIOMNI_LLM` | `claude-sonnet-4-5` | Default LLM for Chainlit UI |
| `BIOMNI_AUTO_NETWORK_LIMITED_MODE` | `true` | Fall back to local data on network failure |

See [docs/configuration.md](docs/configuration.md) for full details.

---

*Last updated: March 2026*
