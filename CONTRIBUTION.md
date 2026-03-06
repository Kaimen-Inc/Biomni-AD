# Contributing to Biomni

Thank you for your interest in contributing to Biomni! We're building the infrastructure layer for biomedical AI agents, and we welcome contributions from the community. Contributors with significant contributions will be invited to co-author publications in top-tier journals and conferences.

**Related docs:** [README.md](README.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [DETAILS.md](DETAILS.md)

## Getting Started

Before contributing, please ensure you:
- Have tested your changes locally
- Follow the existing code style and conventions
- Include appropriate documentation

## Types of Contributions

### 🛠️ Adding a New Tool

Tools are implemented as Python functions in `biomni/tool/XXX.py`, organized by subject area.

**Steps:**
1. **Implement and test** your function locally. If it requires additional software, create installation script and append it into `biomni_env/new_software_{VERSION}.sh`

2. **Choose the appropriate subject** category (e.g. database, biochemistry, etc.)

3. **Create a tool description** in `biomni/tool/tool_description/XXX.py` following the existing format

   *Tip: Use this helper to auto-generate descriptions:*
   ```python
   from biomni.utils import function_to_api_schema
   from biomni.llm import get_llm

   llm = get_llm('claude-sonnet-4-5')
   desc = function_to_api_schema(function_code, llm)
   ```
4. **Create a test prompt** that uses your tool and verify the agent works correctly
5. **Submit a pull request** for review, don't forget to include your test prompt as well

### 📊 Adding New Data

**Persistence note (important):**
- Biomni now persists custom data registrations in a local index under the data lake directory (`_custom_data_index.json`).
- Any file already present in the local data lake is auto-discoverable by A1/AD1 in future sessions.
- To maximize portability across machines, register data paths relative to the data lake root when possible.

If the data source has web API, follow this process:

**Steps:**
1. **Verify uniqueness** - ensure no overlap with existing data
2. **Add a new query_XX function** to `biomni/tool/database.py`, follow the format from the other functions.
3. **Create a tool description** in `biomni/tool/tool_description/database.py` following the existing format

If the data source has no API access, follow the process below:

**Steps:**
1. **Verify uniqueness** - ensure no overlap with existing data
2. **Prepare download link** with verified redistribution rights
3. **Add entry** to `data_lake_dict` in `biomni/env_desc.py`
4. **Submit a pull request** with the download link

### 💻 Adding New Software

**Steps:**
1. **Test locally** to ensure no conflicts with existing environments
2. **Create installation script** and append it into `biomni_env/new_software_{VERSION}.sh`
3. **Add entry** to `library_content_dict` in `biomni/env_desc.py`
4. **Submit a pull request** including:
   - Installation bash script
   - Screenshot demonstrating no environment conflicts

### 🎯 Adding a New Benchmark

Create benchmarks in the `biomni/task/` folder.

**Required implementation:**
```python
class YourBenchmark:
    def __init__(self):
        # Initialize benchmark
        pass

    def __len__(self):
        # Return dataset size
        pass

    def get_example(self, index):
        # Return dataset item at index
        pass

    def evaluate(self):
        # Evaluation logic (flexible input format)
        pass

    def output_class(self):
        # Return expected agent output format
        pass
```

**Steps:**
1. **Create benchmark file** in `biomni/task/[benchmark_name].py`
2. **Implement required methods** as shown above
3. **Provide data download link** for associated datasets
4. **Submit a pull request**

### 📚 Adding Know-How Documents

The Know-How Library (`biomni/know_how/`) provides curated protocols, best practices, and troubleshooting guides automatically retrieved by agents during reasoning. Community contributions expand the library's coverage and quality.

**Suitable contributions:**
- Lab protocols (cell culture, flow cytometry, western blotting, etc.)
- Computational analysis best practices (NGS workflows, microscopy, single-cell, etc.)
- Troubleshooting guides (common pitfalls and solutions)
- Experimental design guidelines (sample size, controls, validation)
- Domain-specific knowledge (drug formulation, animal models, clinical trials, etc.)

**Steps:**
1. **Write a markdown file** following the format of [biomni/know_how/single_cell_annotation.md](biomni/know_how/single_cell_annotation.md) — include a metadata header (authors, affiliations, license, commercial use flag) and practical, succinct content
2. **Place it** in `biomni/know_how/`
3. **Submit a pull request** with a brief description of the domain and source material

### 🧠 Biomni-AD Contributions

Contributions specific to Alzheimer's disease and neurodegeneration research are especially welcome in this repository. Biomni-AD is maintained by **Kuan-lin Huang, PhD**.

**AD-specific contribution types:**
- **AD Tools**: New analysis functions for AD/dementia data (e.g., GWAS, eQTL, proteomics, imaging) in `biomni/tool/`
- **AD Data**: New AD dataset registrations in `biomni/env_desc.py`, or new dataset catalogs in `biomni/know_how/resource/`
- **AD Know-How**: Protocols and best practices for AD research in `biomni/know_how/biomniAD_data_sourcing.md` or new know-how files
- **AD Benchmarks**: Evaluation tasks for AD reasoning in `biomni/task/`

For Biomni-AD-specific issues, please open issues on the Biomni-AD repository (this repo) rather than the upstream snap-stanford/Biomni repository.

### 🐛 Bug Fixes & Enhancements

We welcome all bug fixes and enhancements to the existing codebase!

**Create an issue to discuss with the Biomni team first.**

**Guidelines:**
- Clearly describe the issue or enhancement
- Include tests when applicable
- Follow existing code patterns
- Update documentation if needed

## Submission Process

1. **Fork** the repository
2. **Create a feature branch** from `main`
3. **Make your changes** following the guidelines above
4. **Test thoroughly** in your local environment
5. **Submit a pull request** with a clear description

## Review Process

The Biomni team will review all pull requests promptly. We may request changes or provide feedback to ensure code quality and consistency.

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

---

*Together, let's build the future of biomedical AI agents!*
