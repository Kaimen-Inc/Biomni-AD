"""
Biomni Chainlit UI
==================
Interactive biomedical AI agent with plan-then-approve workflow.

Usage:
    chainlit run chainlit_app.py

Environment variables:
    BIOMNI_LLM      LLM model name (default: claude-sonnet-4-5)
    BIOMNI_PATH     Data directory (default: ./data)
    Provider keys as needed: ANTHROPIC_API_KEY / OPENAI_API_KEY /
    AZURE_ANTHROPIC_API_KEY / AZURE_OPENAI_API_KEY
"""

import asyncio
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

if sys.version_info >= (3, 14):
    print(
        "\n"
        "ERROR: Biomni Chainlit UI is not currently supported on Python 3.14+.\n"
        "\n"
        "  Reason: current Chainlit/AnyIO stack may fail with NoEventLoopError.\n"
        "\n"
        "  Please run with Python 3.11/3.12 (recommended: conda env 'biomni_e1').\n"
        "\n"
        "  Launch command:\n"
        "    bash run_chainlit.sh\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Environment guard — all biomni dependencies live in the biomni_e1 conda env.
# Catch the most common mistake (running from the bare .venv) early.
# ---------------------------------------------------------------------------
try:
    import pandas  # noqa: F401 – representative heavy dep; not used directly here
except ModuleNotFoundError:
    print(
        "\n"
        "ERROR: Required biomedical packages are not installed in the current Python.\n"
        "\n"
        "  This app must run inside the 'biomni_e1' conda environment.\n"
        "\n"
        "  Activate it first:\n"
        "    conda activate biomni_e1\n"
        "\n"
        "  Then launch:\n"
        "    bash run_chainlit.sh\n"
        "  or:\n"
        "    chainlit run chainlit_app.py\n",
        file=sys.stderr,
    )
    sys.exit(1)

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------

def _extract_final_answer(state: dict) -> str:
    """Extract the final answer text from the last agent state, for history storage."""
    if not state or "messages" not in state:
        return ""
    last_content = state["messages"][-1].content
    if not isinstance(last_content, str):
        return ""
    solution_match = re.search(r"<solution>(.*?)</solution>", last_content, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()
    # Fall back to stripped content (remove XML tags)
    cleaned = re.sub(r"<execute>.*?</execute>", "", last_content, flags=re.DOTALL)
    cleaned = re.sub(r"<observation>.*?</observation>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = (
    "You are a biomedical research assistant planning a task. "
    "Given the user's research question, write a concise numbered plan "
    "of 3 to 7 steps describing exactly how you will solve it. "
    "IMPORTANT: Always prefer this tool order in your plan: "
    "(1) web/literature search tools first (advanced_web_search, search_pubmed, search_biorxiv), "
    "(2) local data and database query tools second, "
    "(3) custom code generation only as a last resort. "
    "Mention specific tools, databases, or analyses you will use. "
    "Be specific but brief. Do not execute any code yet."
)

AD1_PLANNING_SYSTEM_PROMPT = (
    "You are an expert Alzheimer's disease research assistant planning a task. "
    "Given the user's research question, write a concise numbered plan "
    "of several steps describing exactly how you will solve it. "
    "IMPORTANT — LOCAL-FIRST RULE: "
    "(1) ALWAYS start by scanning and listing files available in the local user data path "
    "(BIOMNI_USER_DATA_PATH / /app/user-data) and the AD data lake "
    "(data/biomni_data/data_lake/biomniAD/) BEFORE any other action. "
    "Use only locally identified files for as much of the analysis as possible. "
    "Do NOT download, fetch, or call external APIs when the needed data is already present locally. "
    "(2) Built-in domain tools second (query databases, tool functions); "
    "(3) Custom code generation to execute these analyses using your tools. "
    "Do NOT simulate or fabricate data. "
    "Mention specific datasets, tools, or analyses you will use. "
    "Be specific and tailor the plan to user's question. Do not execute any code yet."
)

# Auto-detect Azure OpenAI setup: require deployment + endpoint + Azure OpenAI key.
# This avoids misrouting users who configure Azure Anthropic with the same
# ENDPOINT_URL/DEPLOYMENT_NAME fields.
_azure_deployment = os.getenv("DEPLOYMENT_NAME")
_azure_endpoint = os.getenv("ENDPOINT_URL")
_azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
_azure_default = f"azure-{_azure_deployment}" if (_azure_deployment and _azure_endpoint and _azure_openai_key) else None
_azure_anthropic_key = os.getenv("AZURE_ANTHROPIC_API_KEY")
_azure_anthropic_default = (
    _azure_deployment
    if (_azure_deployment and _azure_endpoint and "anthropic" in _azure_endpoint and _azure_anthropic_key)
    else None
)
DEFAULT_LLM = os.getenv("BIOMNI_LLM") or _azure_default or _azure_anthropic_default or "claude-sonnet-4-5"
DEFAULT_PATH = os.getenv("BIOMNI_PATH", "./data")
# Set BIOMNI_AGENT=a1 to force the A1 agent on startup (skips the profile selector)
FORCE_AGENT = os.getenv("BIOMNI_AGENT", "").lower()  # "a1" | "ad1" | ""

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

CHAINLIT_MD_PATH = Path(__file__).with_name("chainlit.md")
_WELCOME_DATASET_BLOCK_START = "<!-- BIOMNI_LOCAL_DATASET_SECTION_START -->"
_WELCOME_DATASET_BLOCK_END = "<!-- BIOMNI_LOCAL_DATASET_SECTION_END -->"
_SUGGESTED_PROMPTS_BLOCK_START = "<!-- BIOMNI_SUGGESTED_PROMPTS_START -->"
_SUGGESTED_PROMPTS_BLOCK_END = "<!-- BIOMNI_SUGGESTED_PROMPTS_END -->"


# Prompt templates keyed by dataset id — shown only when those files are locally present.
# Each entry is (prompt_text, category).
_AD_DATASET_PROMPTS: list[tuple[str, str, str]] = [
    # (dataset_id_prefix, prompt_text, category)
    ("GCST90027158", "Map the top 10 AD GWAS loci from Bellenguez 2022 (GCST90027158) to nearby genes and report their putative functions", "GWAS"),
    ("NG00052",  "What are the top GWAS hits for CSF clusterin levels in the NG00052 dataset? Which of these overlap known AD risk loci?", "GWAS"),
    ("NG00075",  "Extract genome-wide significant hits from the Kunkle 2019 IGAP stage-2 summary stats (NG00075) and annotate them with gene names", "GWAS"),
    ("NG00102",  "Which proteins are measured across CSF, plasma, and brain tissue in the SomaScan 1.3k proteomic panel (NG00102)? Find any shared with known AD biomarkers", "Proteomics"),
    ("NG00105",  "Identify the top eQTL genes in prefrontal cortex (MFG) from NG00105 that overlap AD GWAS loci — load the cis-QTL file and filter by FDR < 0.05", "QTL"),
    ("NG00118",  "Find structural variant eQTLs in ROSMAP DLPFC (NG00118) for BIN1 and CLU — do they co-localize with GWAS signals?", "QTL"),
    ("NG00126",  "What rare coding variants reach exome-wide significance in the ADSP European WES dataset (NG00126)?", "Rare variants"),
    ("NG00133",  "Analyze the plasma and urine biomarker data from NG00133 — which analytes differ most between AD cases and controls?", "Biomarkers"),
    ("NG00148",  "Compare T-cell receptor CDR3 sequences between AD brain and blood samples using the NG00148 data", "Immunogenomics"),
    ("NG00165",  "Run a gene-level burden analysis summary using the CHARGE/ADSP 5k WGS results (NG00165) — list top gene hits from SKAT and CMC tests", "Rare variants"),
    ("NG00166",  "Which coding and non-coding rare variants are most significant in African American ancestry from ADSP R3 WGS (NG00166)?", "Rare variants"),
    ("NG00172",  "Summarize the structural variant associations with AD risk from NG00172", "Rare variants"),
    ("NG00180",  "Identify metabolites whose MWAS weights (NG00180) are most enriched in AD-related pathways — use the EUR metabolite feature table", "Metabolomics"),
    ("RADR",     "Look up all TREM2 and APOE rare variants in the RADR database (RADR_V3.xlsx) and report their clinical classifications", "Rare variants"),
    ("SingleBrain", "Find microglia-specific eQTLs from SingleBrain that co-localize with AD GWAS loci — load the MG top-association files", "QTL"),
    ("isoMiGA_QTL", "Map isoMiGA microglia splicing QTLs (sQTLs) to the BIN1 and PTK2B loci — load union_leafcutter_top_assoc.tsv.gz", "QTL"),
    ("isoMiGA_counts", "Compare microglia gene expression (TPM) for TREM2, CX3CR1, and P2RY12 across cohorts using isoMiGA count matrices", "Expression"),
]


def _build_ad_suggested_prompts() -> str:
    """Generate suggested prompts based on which BiomniAD datasets are locally present."""
    repo_root = Path(__file__).resolve().parent
    ad_lake = repo_root / "data" / "biomni_data" / "data_lake" / "biomniAD"

    if not ad_lake.is_dir():
        return ""

    present_ids = {d.name for d in ad_lake.iterdir() if d.is_dir() and any(
        f for f in d.iterdir() if f.is_file() and not f.name.lower().startswith("readme")
    )}

    # Collect prompts for available datasets, grouped by category
    from collections import defaultdict
    by_category: dict[str, list[str]] = defaultdict(list)
    for ds_id, prompt_text, category in _AD_DATASET_PROMPTS:
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


def _list_path_entries(path: str, max_items: int = 40) -> list[str]:
    """List non-hidden entries for a path (brief, non-recursive)."""
    if not path or not os.path.isdir(path):
        return []
    try:
        entries = sorted(name for name in os.listdir(path) if not name.startswith("."))
    except OSError:
        return []
    return entries[:max_items]


def _resolve_user_data_roots() -> list[tuple[str, str]]:
    """Resolve configured user data roots from supported env vars (deduplicated)."""
    user_data_host_path = os.getenv("BIOMNI_USER_DATA_HOST_PATH", "").strip()
    user_data_path = os.getenv("BIOMNI_USER_DATA_PATH", "").strip()
    legacy_user_data_path = os.getenv("BIOMNI_DATA_PATH", "").strip()
    biomni_path = os.getenv("BIOMNI_PATH", "").strip()

    candidates: list[tuple[str, str]] = []

    # Host-mounted path is usually a compose substitution variable and may not
    # exist inside the running container. Only include if it is actually visible.
    if user_data_host_path and os.path.isdir(user_data_host_path):
        candidates.append(("BIOMNI_USER_DATA_HOST_PATH", user_data_host_path))

    if user_data_path:
        candidates.append(("BIOMNI_USER_DATA_PATH", user_data_path))

    if legacy_user_data_path:
        candidates.append(("BIOMNI_DATA_PATH", legacy_user_data_path))

    # BIOMNI_PATH is commonly the built-in app data root (/app/data) in Docker.
    # Treat it as a user-data fallback only when explicit user paths are absent.
    if biomni_path and not (user_data_path or legacy_user_data_path):
        candidates.append(("BIOMNI_PATH", biomni_path))

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    builtin_root = os.path.abspath(_resolve_builtin_data_lake_root())

    def _is_builtin_or_parent(path_value: str) -> bool:
        root = os.path.abspath(path_value)
        if root == builtin_root:
            return True
        if builtin_root.startswith(root + os.sep):
            return True
        return False

    for env_name, raw_path in candidates:
        if not raw_path:
            continue
        abs_path = os.path.abspath(raw_path)
        if env_name == "BIOMNI_PATH" and _is_builtin_or_parent(abs_path):
            continue
        if abs_path in seen:
            continue
        seen.add(abs_path)
        resolved.append((env_name, abs_path))
    return resolved


def _display_data_root_label(env_name: str) -> str:
    """Map env var names to concise sidebar labels."""
    if env_name == "BIOMNI_USER_DATA_HOST_PATH":
        return "AD_WORKBENCH_DATASETS"
    return env_name


def _list_path_entries_recursive(path: str, max_items: int = 80, max_depth: int = 10, exclude_top_subdirs: set[str] | None = None) -> tuple[list[str], int]:
    """Recursively list non-hidden files under a directory.

    Returns a (preview_items, total_file_count) tuple. Preview items are
    relative POSIX-style paths suitable for UI display.

    ``exclude_top_subdirs`` names top-level subdirectories to skip entirely.
    """
    if not path or not os.path.isdir(path):
        return [], 0

    excluded_dirs = {
        ".git", "__pycache__", ".venv", "venv", "env", "node_modules", "site-packages"
    }

    preview: list[str] = []
    total_count = 0

    for root, dirs, files in os.walk(path):
        rel_root = os.path.relpath(root, path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue

        visible = [d for d in dirs if not d.startswith(".") and d not in excluded_dirs]
        if depth == 0 and exclude_top_subdirs:
            visible = [d for d in visible if d not in exclude_top_subdirs]
        dirs[:] = sorted(visible)

        for file_name in sorted(files):
            if file_name.startswith("."):
                continue
            total_count += 1
            rel = file_name if rel_root == "." else f"{rel_root}/{file_name}"
            rel = rel.replace(os.sep, "/")
            if len(preview) < max_items:
                preview.append(rel)

    return preview, total_count


def _collect_path_stats(path: str, max_depth: int = 10, exclude_top_subdirs: set[str] | None = None) -> dict:
    """Collect compact stats for a directory tree for sidebar summaries.

    ``exclude_top_subdirs`` names top-level subdirectories to skip entirely
    (useful for counting the datalake root without the biomniAD subfolder).
    """
    stats = {
        "total_files": 0,
        "total_dirs": 0,
        "top_level_counts": {},
        "extension_counts": {},
    }
    if not path or not os.path.isdir(path):
        return stats

    excluded_dirs = {
        ".git", "__pycache__", ".venv", "venv", "env", "node_modules", "site-packages"
    }

    for root, dirs, files in os.walk(path):
        rel_root = os.path.relpath(root, path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue

        visible_dirs = [d for d in dirs if not d.startswith(".") and d not in excluded_dirs]
        if depth == 0 and exclude_top_subdirs:
            visible_dirs = [d for d in visible_dirs if d not in exclude_top_subdirs]
        dirs[:] = sorted(visible_dirs)
        stats["total_dirs"] += len(visible_dirs)

        for file_name in files:
            if file_name.startswith("."):
                continue

            stats["total_files"] += 1

            ext = Path(file_name).suffix.lower() or "[no_ext]"
            ext_counts = stats["extension_counts"]
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

            if rel_root == ".":
                top = "[root]"
            else:
                top = rel_root.split(os.sep, 1)[0]
            top_counts = stats["top_level_counts"]
            top_counts[top] = top_counts.get(top, 0) + 1

    return stats


def _format_compact_counts(counts: dict[str, int], max_items: int = 8) -> str:
    """Format a frequency map as a compact markdown bullet list."""
    if not counts:
        return "- *(none)*"
    top_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_items]
    return "\n".join(f"- `{name}`: {value}" for name, value in top_items)


def _build_tree_preview_lines(paths: list[str], max_lines: int = 60, max_depth: int = 3) -> list[str]:
    """Render relative file paths as a compact folder tree preview.

    Directory counts are computed from the provided path sample.
    """
    # Tree node shape: {"dirs": {name: node}, "files": [name, ...]}
    tree: dict[str, object] = {"dirs": {}, "files": []}

    for rel_path in sorted(paths):
        parts = [p for p in rel_path.split("/") if p]
        if not parts:
            continue

        node = tree
        for idx, part in enumerate(parts):
            is_file = idx == len(parts) - 1
            if is_file:
                files = node.setdefault("files", [])
                if isinstance(files, list):
                    files.append(part)
            else:
                dirs = node.setdefault("dirs", {})
                if not isinstance(dirs, dict):
                    break
                if part not in dirs:
                    dirs[part] = {"dirs": {}, "files": []}
                child = dirs.get(part)
                if not isinstance(child, dict):
                    break
                node = child

    lines: list[str] = []

    def _count_files(node: dict[str, object]) -> int:
        count = 0
        files = node.get("files", [])
        dirs = node.get("dirs", {})

        if isinstance(files, list):
            count += len(files)
        if isinstance(dirs, dict):
            for child in dirs.values():
                if isinstance(child, dict):
                    count += _count_files(child)
        return count

    def _render(node: dict[str, object], prefix: str, depth: int) -> bool:
        if len(lines) >= max_lines:
            return False
        dirs = node.get("dirs", {})
        files = node.get("files", [])

        dir_names = sorted(dirs.keys()) if isinstance(dirs, dict) else []
        file_names = sorted(str(f) for f in files) if isinstance(files, list) else []
        entries: list[tuple[str, str, object | None]] = []
        for dirname in dir_names:
            child = dirs.get(dirname) if isinstance(dirs, dict) else None
            entries.append(("dir", dirname, child))
        for filename in file_names:
            entries.append(("file", filename, None))

        for idx, (kind, name, child) in enumerate(entries):
            is_last = idx == len(entries) - 1
            branch = "└─ " if is_last else "├─ "
            next_prefix = prefix + ("   " if is_last else "│  ")

            if kind == "dir":
                child_count = _count_files(child) if isinstance(child, dict) else 0
                lines.append(f"{prefix}{branch}📁 {name}/ ({child_count})")
                if len(lines) >= max_lines:
                    return False
                if isinstance(child, dict):
                    if depth + 1 < max_depth:
                        if not _render(child, next_prefix, depth + 1):
                            return False
                    elif child_count > 0:
                        lines.append(f"{next_prefix}…")
                        if len(lines) >= max_lines:
                            return False
            else:
                lines.append(f"{prefix}{branch}📄 {name}")
                if len(lines) >= max_lines:
                    return False

        return True

    _render(tree, prefix="", depth=0)
    return lines


def _build_user_data_tree_content(root_path: str, preview_files: int = 300) -> tuple[str, int]:
    """Build concise per-root content for sidebar readability."""
    preview, total_files = _list_path_entries_recursive(root_path, max_items=preview_files)
    if total_files == 0:
        return "(no files found)", 0

    tree_lines = _build_tree_preview_lines(preview, max_lines=70, max_depth=3)
    lines: list[str] = [
        f"Directory structure preview (showing first {len(preview)} files)",
        "",
        *tree_lines,
    ]
    if total_files > len(preview):
        lines.append(f"... and {total_files - len(preview)} more files")

    return "\n".join(lines), total_files


def _build_sidebar_overview_content() -> str:
    """Build an at-a-glance overview across all configured roots.

    Sections (in order, only shown when they have files):
    1. AD Workbench Datasets  – when BIOMNI_USER_DATA_HOST_PATH is defined,
       read file counts from the container-side mount (BIOMNI_USER_DATA_PATH).
    2. Biomni-AD Datalake     – data_lake/biomniAD/ subfolder.
    3. Biomni Datalake        – root of data_lake/ excluding biomniAD.
    """
    user_data_host_path = os.getenv("BIOMNI_USER_DATA_HOST_PATH", "").strip()
    user_data_path = os.getenv("BIOMNI_USER_DATA_PATH", "").strip()

    lines: list[str] = ["At-a-glance overview", ""]
    grand_files = 0
    grand_dirs = 0

    # --- 1. AD Workbench Datasets -------------------------------------------
    # When HOST_PATH is defined this is an AD-Workbench deployment. The host
    # directory is bind-mounted into the container at BIOMNI_USER_DATA_PATH
    # (/app/user-data), so count files from the container-side path.
    if user_data_host_path:
        # Prefer the container mount; fall back to the host path only if the
        # mount path is absent or empty.
        ad_path = ""
        if user_data_path and os.path.isdir(user_data_path):
            ad_path = user_data_path
        elif os.path.isdir(user_data_host_path):
            ad_path = user_data_host_path
        if ad_path:
            stats = _collect_path_stats(ad_path)
            if stats["total_files"] > 0:
                grand_files += stats["total_files"]
                grand_dirs += stats["total_dirs"]
                lines.append(f"AD Workbench Datasets: {stats['total_files']} files, {stats['total_dirs']} folders")
                lines.append("")
    else:
        # Local / non-AD-Workbench deployment: show regular user data roots.
        for env_name, root in _resolve_user_data_roots():
            stats = _collect_path_stats(root)
            if stats["total_files"] > 0:
                grand_files += stats["total_files"]
                grand_dirs += stats["total_dirs"]
                display = _display_data_root_label(env_name)
                lines.append(f"{display}: {stats['total_files']} files, {stats['total_dirs']} folders")
                lines.append("")

    # --- 2. Biomni-AD Datalake ----------------------------------------------
    builtin_root = _resolve_builtin_data_lake_root()
    biomni_ad_root = os.path.join(builtin_root, "biomniAD")
    if os.path.isdir(biomni_ad_root):
        stats = _collect_path_stats(biomni_ad_root)
        if stats["total_files"] > 0:
            grand_files += stats["total_files"]
            grand_dirs += stats["total_dirs"]
            lines.append(f"Biomni-AD Datalake: {stats['total_files']} files, {stats['total_dirs']} folders")
            lines.append("")

    # --- 3. Biomni Datalake (root of data_lake, excluding biomniAD) ----------
    if os.path.isdir(builtin_root):
        stats = _collect_path_stats(builtin_root, exclude_top_subdirs={"biomniAD"})
        if stats["total_files"] > 0:
            grand_files += stats["total_files"]
            grand_dirs += stats["total_dirs"]
            lines.append(f"Biomni Datalake: {stats['total_files']} files, {stats['total_dirs']} folders")
            lines.append("")

    if grand_files == 0 and grand_dirs == 0:
        return "No local data roots found."

    lines.extend([
        "Combined totals",
        f"Files: {grand_files}",
        f"Folders: {grand_dirs}",
    ])
    return "\n".join(lines)


def _resolve_builtin_data_lake_root() -> str:
    """Return the default repo-local data lake directory path."""
    repo_root = Path(__file__).resolve().parent
    return str((repo_root / "data" / "biomni_data" / "data_lake").resolve())


def _list_local_data_lake_files(base_path: str, max_items: int = 30) -> list[str]:
    """List local data lake files (relative paths) from common Biomni folders."""
    candidates = [
        Path(base_path) / "biomni_data" / "data_lake",
        Path(base_path) / "data_lake",
    ]

    data_lake_dir = next((c for c in candidates if c.is_dir()), None)
    if data_lake_dir is None:
        return []

    items: list[str] = []
    for root, _dirs, files in os.walk(data_lake_dir):
        for file_name in files:
            if file_name.startswith("."):
                continue
            full_path = Path(root) / file_name
            rel = full_path.relative_to(data_lake_dir).as_posix()
            if rel == "_custom_data_index.json":
                continue
            items.append(rel)

    items = sorted(set(items))
    return items[:max_items]


def _build_welcome_local_dataset_section() -> str:
    """Build a collapsed markdown block shown at the bottom of the Chainlit welcome page."""
    # Data lake always sourced from the repo-local built-in location
    _repo_root = Path(__file__).resolve().parent
    builtin_data_lake = _repo_root / "data" / "biomni_data" / "data_lake"
    data_lake_files = _list_local_data_lake_files(str(_repo_root / "data"), max_items=200)

    # User data can come from BIOMNI_USER_DATA_PATH and legacy BIOMNI_DATA_PATH/BIOMNI_PATH.
    user_roots = _resolve_user_data_roots()
    user_total_files = 0
    user_preview: list[str] = []
    if user_roots:
        first_root = user_roots[0][1]
        user_preview, user_total_files = _list_path_entries_recursive(first_root, max_items=20)

    # One-line summary for the collapsed header
    summary_parts = [f"{len(data_lake_files)} data lake files"]
    if user_total_files:
        summary_parts.append(f"{user_total_files} user data files")

    lines: list[str] = []
    lines.append(f"<details><summary>📊 {' · '.join(summary_parts)} available — click to expand</summary>")
    lines.append("")
    lines.append(f"**Built-in Data Lake** — `{builtin_data_lake}`")
    lines.append("")
    if data_lake_files:
        for name in data_lake_files:
            lines.append(f"- `{name}`")
    else:
        lines.append("- *(none found)*")

    if user_roots:
        lines.append("")
        lines.append("**User Data** — from `BIOMNI_USER_DATA_HOST_PATH` / `BIOMNI_USER_DATA_PATH` / `BIOMNI_DATA_PATH` / `BIOMNI_PATH`")
        lines.append("")
        primary_label = _display_data_root_label(user_roots[0][0])
        lines.append(f"Primary path ({primary_label}): `{user_roots[0][1]}`")
        if len(user_roots) > 1:
            lines.append("Additional configured paths:")
            for env_name, root in user_roots[1:]:
                lines.append(f"- `{_display_data_root_label(env_name)}`: `{root}`")
        lines.append("")
        if user_preview:
            lines.append(f"Detected files: {user_total_files}")
            for name in user_preview:
                lines.append(f"- `{name}`")
            if user_total_files > len(user_preview):
                lines.append(f"- `... and {user_total_files - len(user_preview)} more`")
        else:
            lines.append("- *(none found)*")

    lines.append("")
    lines.append("</details>")

    return "\n".join(lines)


def _refresh_chainlit_welcome_markdown() -> None:
    """Append or replace managed sections (suggested prompts + dataset list) in chainlit.md."""
    try:
        if CHAINLIT_MD_PATH.exists():
            original = CHAINLIT_MD_PATH.read_text(encoding="utf-8")
        else:
            original = (
                "## Hi, I'm Biomni-AD 🧠\n"
                "#### Your AI co-scientist on the journey to conquer Alzheimer's disease.\n\n"
                "Tell me a research question to get started.\n"
            )

        # Strip both managed blocks
        for start, end in [
            (_SUGGESTED_PROMPTS_BLOCK_START, _SUGGESTED_PROMPTS_BLOCK_END),
            (_WELCOME_DATASET_BLOCK_START, _WELCOME_DATASET_BLOCK_END),
        ]:
            original = re.sub(
                rf"\n?{re.escape(start)}.*?{re.escape(end)}\n?",
                "\n",
                original,
                flags=re.DOTALL,
            )
        base = original.rstrip()

        # Build suggested prompts block (only shown when local datasets are present)
        suggested = _build_ad_suggested_prompts()
        prompts_block = (
            f"\n\n{_SUGGESTED_PROMPTS_BLOCK_START}\n"
            f"{suggested}\n"
            f"{_SUGGESTED_PROMPTS_BLOCK_END}\n"
        ) if suggested else ""

        # Build dataset inventory block
        dataset_section = _build_welcome_local_dataset_section()
        dataset_block = (
            f"\n\n{_WELCOME_DATASET_BLOCK_START}\n"
            f"{dataset_section}\n"
            f"{_WELCOME_DATASET_BLOCK_END}\n"
        )

        CHAINLIT_MD_PATH.write_text(base + prompts_block + dataset_block, encoding="utf-8")
    except Exception as exc:
        print(f"Warning: Could not refresh chainlit welcome markdown: {exc}")


_refresh_chainlit_welcome_markdown()

# ---------------------------------------------------------------------------
# Dataset listing helper
# ---------------------------------------------------------------------------

_DATALAKE_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Protein Interactions", [
        "affinity_capture-ms", "affinity_capture-rna", "co-fractionation",
        "proximity_label-ms", "reconstituted_complex", "two-hybrid",
        "Virus-Host_PPI_P-HIPSTER_2020",
    ]),
    ("Drug & Compound Data", [
        "BindingDB_All_202409", "broad_repurposing_hub_molecule_with_smiles",
        "broad_repurposing_hub_phase_moa_target_info", "enamine_cloud_library_smiles",
        "ddinter_alimentary_tract_metabolism", "ddinter_antineoplastic",
        "ddinter_antiparasitic", "ddinter_blood_organs", "ddinter_dermatological",
        "ddinter_hormonal", "ddinter_respiratory", "ddinter_various",
    ]),
    ("Gene Expression & Cancer", [
        "DepMap_CRISPRGeneDependency", "DepMap_CRISPRGeneEffect", "DepMap_Model",
        "DepMap_OmicsExpressionProteinCodingGenesTPMLogp1",
        "gtex_tissue_gene_tpm", "proteinatlas",
    ]),
    ("Genomics & Genetic Variants", [
        "genebass_missense_LC_filtered", "genebass_pLoF_filtered",
        "genebass_synonymous_filtered", "gwas_catalog", "variant_table",
        "sgRNA_KO_SP_human", "sgRNA_KO_SP_mouse",
    ]),
    ("Gene Sets & Functional Annotations", [
        "msigdb_human_c1_positional_geneset", "msigdb_human_c2_curated_geneset",
        "msigdb_human_c3_regulatory_target_geneset",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD",
        "msigdb_human_c4_computational_geneset", "msigdb_human_c5_ontology_geneset",
        "msigdb_human_c6_oncogenic_signature_geneset",
        "msigdb_human_c7_immunologic_signature_geneset",
        "msigdb_human_c8_celltype_signature_geneset", "msigdb_human_h_hallmark_geneset",
        "mousemine_m1_positional_geneset", "mousemine_m2_curated_geneset",
        "mousemine_m3_regulatory_target_geneset", "mousemine_m5_ontology_geneset",
        "mousemine_m8_celltype_signature_geneset", "mousemine_mh_hallmark_geneset",
        "go-plus", "gene_info",
    ]),
    ("Disease & Phenotype", [
        "DisGeNET", "omim", "hp", "kg",
    ]),
    ("Cell Biology", [
        "czi_census_datasets_v4", "marker_celltype",
    ]),
    ("RNA Biology", [
        "miRDB_v6.0_results", "miRTarBase_microRNA_target_interaction",
        "miRTarBase_microRNA_target_interaction_pubmed_abtract",
        "miRTarBase_MicroRNA_Target_Sites",
    ]),
    ("Genetic Interactions", [
        "dosage_growth_defect", "genetic_interaction",
        "synthetic_growth_defect", "synthetic_lethality", "synthetic_rescue",
    ]),
    ("Immunology & Other", [
        "McPAS-TCR", "txgnn_name_mapping", "txgnn_prediction",
    ]),
]


def _build_dataset_listing(agent) -> str:
    """Return a compact sidebar summary (tree details are in separate elements)."""
    _ = agent
    user_roots = _resolve_user_data_roots()
    builtin_root = _resolve_builtin_data_lake_root()
    if not user_roots and not os.path.isdir(builtin_root):
        return "No local data tree available"
    return "Open a Tree item below"


def _build_full_user_data_inventory() -> str:
    """Build a comprehensive file listing of all user-data roots for agent injection.

    This produces the SAME content the sidebar tree shows but as a single text
    block that can be set on the agent so its system prompt has full visibility
    into mounted VM datasets.
    """
    sections: list[str] = []

    user_data_host_path = os.getenv("BIOMNI_USER_DATA_HOST_PATH", "").strip()
    user_data_path = os.getenv("BIOMNI_USER_DATA_PATH", "").strip()

    # --- User / AD Workbench data ---
    if user_data_host_path:
        ad_path = ""
        if user_data_path and os.path.isdir(user_data_path):
            ad_path = user_data_path
        elif os.path.isdir(user_data_host_path):
            ad_path = user_data_host_path
        if ad_path:
            preview, total = _list_path_entries_recursive(ad_path, max_items=500, max_depth=10)
            if total > 0:
                tree_lines = _build_tree_preview_lines(preview, max_lines=300, max_depth=6)
                sections.append(
                    f"AD Workbench / User Data ({ad_path}) — {total} files:\n"
                    + "\n".join(tree_lines)
                )
                if total > len(preview):
                    sections[-1] += f"\n  ... and {total - len(preview)} more files"
    else:
        for env_name, root in _resolve_user_data_roots():
            preview, total = _list_path_entries_recursive(root, max_items=500, max_depth=10)
            if total > 0:
                label = _display_data_root_label(env_name)
                tree_lines = _build_tree_preview_lines(preview, max_lines=300, max_depth=6)
                sections.append(
                    f"{label} ({root}) — {total} files:\n"
                    + "\n".join(tree_lines)
                )
                if total > len(preview):
                    sections[-1] += f"\n  ... and {total - len(preview)} more files"

    return "\n\n".join(sections) if sections else ""


def _build_user_data_sidebar_elements() -> list[cl.Text]:
    """Build tree elements for built-in data lake and configured user data roots.

    Tree order mirrors the overview:
    1. AD Workbench Datasets (when BIOMNI_USER_DATA_HOST_PATH is defined)
       or regular user-data roots otherwise.
    2. Biomni-AD Datalake  (data_lake/biomniAD/)
    3. Biomni Datalake     (data_lake/ root, excluding biomniAD)
    Only entries with files are included.
    """
    elements: list[cl.Text] = [
        cl.Text(name="Summary", content=_build_sidebar_overview_content(), display="page")
    ]

    user_data_host_path = os.getenv("BIOMNI_USER_DATA_HOST_PATH", "").strip()
    user_data_path = os.getenv("BIOMNI_USER_DATA_PATH", "").strip()

    # --- User data / AD Workbench tree ---------------------------------------
    if user_data_host_path:
        # Use container-side mount for file counts; host path is the display label.
        ad_path = ""
        if user_data_path and os.path.isdir(user_data_path):
            ad_path = user_data_path
        elif os.path.isdir(user_data_host_path):
            ad_path = user_data_host_path
        if ad_path:
            tree_content, total_files = _build_user_data_tree_content(ad_path)
            if total_files > 0:
                content = f"Path: {ad_path}\n\n{tree_content}"
                elements.append(cl.Text(name=f"Tree [AD Workbench Datasets] ({total_files})", content=content, display="page"))
    else:
        for env_name, root in _resolve_user_data_roots():
            tree_content, total_files = _build_user_data_tree_content(root)
            if total_files > 0:
                label = f"Tree [{_display_data_root_label(env_name)}] ({total_files})"
                content = f"Path: {root}\n\n{tree_content}"
                elements.append(cl.Text(name=label, content=content, display="page"))

    # --- Built-in datalake trees ---------------------------------------------
    builtin_root = _resolve_builtin_data_lake_root()

    # Biomni-AD Datalake
    biomni_ad_root = os.path.join(builtin_root, "biomniAD")
    if os.path.isdir(biomni_ad_root):
        tree_content, total_files = _build_user_data_tree_content(biomni_ad_root)
        if total_files > 0:
            content = f"Path: {biomni_ad_root}\n\n{tree_content}"
            elements.append(cl.Text(name=f"Tree [Biomni-AD Datalake] ({total_files})", content=content, display="page"))

    # Biomni Datalake (root, excluding biomniAD)
    if os.path.isdir(builtin_root):
        preview, total_files = _list_path_entries_recursive(builtin_root, max_items=300, exclude_top_subdirs={"biomniAD"})
        if total_files > 0:
            tree_lines = _build_tree_preview_lines(preview, max_lines=70, max_depth=3)
            lake_lines: list[str] = [
                f"Directory structure preview (showing first {len(preview)} files)",
                "",
                *tree_lines,
            ]
            if total_files > len(preview):
                lake_lines.append(f"... and {total_files - len(preview)} more files")
            tree_content = "\n".join(lake_lines)
            content = f"Path: {builtin_root}\n\n{tree_content}"
            elements.append(cl.Text(name=f"Tree [Biomni Datalake] ({total_files})", content=content, display="page"))

    return elements


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def run_in_executor(fn, *args):
    """Run a synchronous function in a thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def stream_langgraph(agent_app, inputs, config):
    """Yield LangGraph state dicts asynchronously from a sync stream."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _producer():
        try:
            for state in agent_app.stream(inputs, stream_mode="values", config=config):
                asyncio.run_coroutine_threadsafe(queue.put(state), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(_producer)

    while True:
        state = await queue.get()
        if state is None:
            break
        yield state


# ---------------------------------------------------------------------------
# Chat profiles (A1 vs AD1)
# ---------------------------------------------------------------------------

@cl.set_chat_profiles
async def set_chat_profiles():
    if FORCE_AGENT:
        # CLI override: skip the selector entirely
        return None
    return [
        cl.ChatProfile(
            name="AD1",
            markdown_description=(
                "**Hi, I'm Biomni-AD — Your Alzheimer's Disease Co-Scientist**\n\n"
                "What would you like to discover about Alzheimer's today?"
            ),
            icon="/public/avatars/ad1.png",
        ),
        cl.ChatProfile(
            name="A1",
            markdown_description=(
                "**A1 — General-Purpose Biomedical Agent**\n\n"
                "Broad biomedical research across genomics, proteomics, "
                "single-cell, clinical data, and more."
            ),
            icon="/public/avatars/ad1.png",
        ),
    ]


# ---------------------------------------------------------------------------
# AD1 example starters
# ---------------------------------------------------------------------------

AD1_STARTERS = [
    cl.Starter(
        label="Multi-omics AD risk gene portrait",
        message=(
            "Build a multi-omics portrait of the top 5 AD risk genes (APOE, TREM2, BIN1, CLU, SORL1). "
            "For each gene: (1) pull GWAS significance from any local summary stats "
            "(GCST90027158, NG00075, or NG00052), (2) check brain eQTL evidence in NG00105 or "
            "SingleBrain, (3) look up proteomic levels in NG00102 if available, and "
            "(4) find CRISPR dependency scores from CRISPRbrain relevant screen from the biomni-AD datalake."
            "Compile everything into a single comparison table and a radar chart per gene."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="Microglia enhancer–GWAS overlap",
        message=(
            "Using the AD Workbench ATAC-seq / enhancer-promoter interactome datasets and the "
            "Bellenguez 2022 GWAS (GCST90027158), identify AD risk SNPs that fall within "
            "microglia-specific enhancer regions. For each hit, report the target gene linked by "
            "the enhancer-promoter map and whether it also appears as a microglia eQTL in "
            "SingleBrain or isoMiGA. Produce a Venn diagram of the overlaps and a genome-browser "
            "style track plot for the top 3 loci."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="AD plasma biomarker × genetic risk",
        message=(
            "Cross-reference AD Workbench plasma/CSF biomarker datasets (e.g. Bio-Hermes, "
            "BCG-PANDA, NG00133) with genetic risk data: (1) load the biomarker tables and "
            "identify analytes that differ most between AD cases and controls, (2) check whether "
            "the genes encoding those top analytes carry genome-wide significant variants in any "
            "local GWAS files (NG00075, GCST90027158), and (3) look up their rare-variant burden "
            "in RADR. Produce a dot plot of effect size vs. GWAS p-value and a summary table."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="AD drug target prioritization",
        message=(
            "Prioritize druggable AD targets by integrating local data: (1) extract genes reaching "
            "genome-wide significance from available GWAS summary stats, (2) filter those with "
            "brain eQTL support (NG00105 / SingleBrain), (3) check which have known drug "
            "interactions in the BindingDB and Broad Repurposing Hub files from the general data "
            "lake, and (4) cross-reference CRISPR dependency in DepMap. "
            "Rank the final targets by a composite score and produce a waterfall plot with a table "
            "listing each gene, its top drug candidates, and supporting evidence."
        ),
        icon="/public/avatars/ad1.png",
    ),
]


@cl.set_starters
async def set_starters():
    return AD1_STARTERS


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialize the selected agent and greet the user."""
    # Determine agent type: CLI env var > chat profile selection > default AD1
    if FORCE_AGENT in ("a1", "ad1"):
        agent_type = FORCE_AGENT
    else:
        profile = cl.user_session.get("chat_profile", "AD1")
        agent_type = "a1" if str(profile).upper() == "A1" else "ad1"

    label = "AD1" if agent_type == "ad1" else "A1"
    try:
        if agent_type == "ad1":
            from biomni.agent.ad1 import AD1
            agent = await run_in_executor(lambda: AD1(llm=DEFAULT_LLM))
        else:
            from biomni.agent.a1 import A1
            agent = await run_in_executor(lambda: A1(llm=DEFAULT_LLM))
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_type", agent_type)
        cl.user_session.set("history", [])
        cl.user_session.set("thread_id", str(uuid.uuid4()))
        cl.user_session.set("dataset_listing", _build_dataset_listing(agent))
        cl.user_session.set("dataset_panel_shown", False)

        # Build full user-data inventory and inject it into the agent so its
        # system prompt has the same visibility as the sidebar tree.
        inventory_text = await run_in_executor(_build_full_user_data_inventory)
        if inventory_text:
            agent.user_data_inventory = inventory_text
    except Exception as exc:
        await cl.Message(content=f"Failed to initialize {label}: {exc}").send()
        return

    # Render local-data panel in the native sidebar at startup,
    # keeping the center welcome/search screen unchanged.
    sidebar_content = cl.user_session.get("dataset_listing") or _build_dataset_listing(agent)
    sidebar_elements = _build_user_data_sidebar_elements()
    if not sidebar_elements:
        sidebar_elements = [cl.Text(name="Local Datasets", content=sidebar_content)]
    await cl.ElementSidebar.set_title("Local Datasets")
    await cl.ElementSidebar.set_elements(sidebar_elements)
    cl.user_session.set("dataset_panel_shown", True)


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type", "a1")

    if agent is None:
        await cl.Message(
            content="Agent not initialized. Please refresh the page."
        ).send()
        return

    prompt = message.content
    if message.elements:
        file_paths = [e.path for e in message.elements if hasattr(e, "path") and e.path]
        if file_paths:
            prompt += "\n\nUser uploaded these files:\n" + "\n".join(f"- {p}" for p in file_paths)

    # ------------------------------------------------------------------
    # Phase 1: AD context injection (AD1 only)
    # ------------------------------------------------------------------
    if agent_type == "ad1":
        ad_keywords = getattr(agent, "ad_keywords", [])
        if any(kw.lower() in prompt.lower() for kw in ad_keywords):
            async with cl.Step(name="🧠 AD Context Detected", type="tool", show_input=False) as step:
                await run_in_executor(agent._inject_ad_context)
                step.output = (
                    "Specialized AD/dementia data sourcing protocols injected into context."
                )

    # ------------------------------------------------------------------
    # Phase 2: Tool retrieval
    # ------------------------------------------------------------------
    if getattr(agent, "use_tool_retriever", False):
        async with cl.Step(name="🔍 Selecting Resources", type="retrieval", show_input=False) as step:
            try:
                resources = await run_in_executor(
                    agent._prepare_resources_for_retrieval, prompt
                )
                if resources:
                    await run_in_executor(
                        agent.update_system_prompt_with_selected_resources, resources
                    )
                    tools_n = len(resources.get('tools', []))
                    data_n = len(resources.get('data_lake', []))
                    libs_n = len(resources.get('libraries', []))
                    knowhow_n = len(resources.get('know_how', []))
                    total = tools_n + data_n + libs_n + knowhow_n
                    step.output = (
                        f"Selected {total} resources: "
                        f"🔧 {tools_n} tools, "
                        f"📊 {data_n} datasets, "
                        f"⚙️ {libs_n} libraries, "
                        f"📚 {knowhow_n} know-how documents."
                    )
                else:
                    step.output = "No resources selected; proceeding with full tool set."
            except Exception as exc:
                step.output = f"⚠️ Tool retrieval failed ({exc}); proceeding with all tools."

    # ------------------------------------------------------------------
    # Phase 3: Interactive planning
    # ------------------------------------------------------------------
    prompt = await _interactive_planning(agent, prompt, agent_type=agent_type)
    if prompt is None:
        # User cancelled
        await cl.Message(content="Execution cancelled.").send()
        return

    # ------------------------------------------------------------------
    # Phase 4: Stream agent execution
    # ------------------------------------------------------------------
    history = cl.user_session.get("history", [])
    thread_id = cl.user_session.get("thread_id", "42")

    # Pre-create run directory so OUTPUT_DIR is available during code execution.
    try:
        _run_id = _build_run_id(prompt)
        _runs_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
        os.makedirs(_runs_root, exist_ok=True)
        _current_run_dir = os.path.join(_runs_root, _run_id)
        os.makedirs(_current_run_dir, exist_ok=True)
        agent._current_run_dir = _current_run_dir
        os.environ["BIOMNI_OUTPUT_PATH"] = _current_run_dir
    except Exception as _e:
        print(f"Warning: Could not pre-create run directory: {_e}")
        _current_run_dir = None

    # Snapshot files before execution (cwd + data root) to detect new outputs.
    initial_files = _get_all_files(os.getcwd())
    _data_root = getattr(agent, "data_root_dir", None)
    if _data_root and os.path.isdir(_data_root):
        initial_files |= _get_all_files(_data_root)

    final_state = await _stream_execution(agent, prompt, history, thread_id)

    # Update conversation history with the user prompt and agent response
    if final_state:
        answer = _extract_final_answer(final_state)
        history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        cl.user_session.set("history", history)

    # ------------------------------------------------------------------
    # Phase 5: Artifact saving (all agents)
    # ------------------------------------------------------------------
    if final_state and hasattr(agent, "_save_run_artifacts"):
        await _save_run_artifacts_for_agent(agent, final_state, initial_files, topic=prompt)

# ---------------------------------------------------------------------------
# Interactive planning helpers
# ---------------------------------------------------------------------------

async def _interactive_planning(agent, prompt: str, agent_type: str = "a1") -> str | None:
    """
    Generate a research plan, show it for user approval, and handle revisions.
    Returns the (possibly modified) prompt on approval, or None if cancelled.
    """
    base_prompt = AD1_PLANNING_SYSTEM_PROMPT if agent_type == "ad1" else PLANNING_SYSTEM_PROMPT

    # Dynamically append the user-data directory listing so the planner
    # knows exactly which datasets are available locally.
    inventory = getattr(agent, "user_data_inventory", None)
    if inventory:
        data_root = getattr(agent, "data_root_dir", None) or ""
        base_prompt += (
            f"\n\nThe following datasets are available in the local user data directory "
            f"({data_root}). Reference specific datasets from this listing when relevant "
            f"to the user's question:\n{inventory}"
        )

    modification_context = ""

    while True:
        # Build planning messages
        full_system = base_prompt
        if modification_context:
            full_system += f"\n\nUser requested these revisions to the previous plan:\n{modification_context}"

        planning_messages = [
            SystemMessage(content=full_system),
            HumanMessage(content=prompt),
        ]

        async with cl.Step(name="📋 Generating Research Plan", type="llm", show_input=False) as step:
            try:
                response = await run_in_executor(agent.llm.invoke, planning_messages)
                plan_text = response.content if hasattr(response, "content") else str(response)
                step.output = plan_text
            except Exception as exc:
                step.output = f"⚠️ Could not generate plan ({exc}). Proceeding without a plan."
                # Fall through to execution without approval gate
                return prompt

        # Ask the user to approve or revise
        res = await cl.AskActionMessage(
            content="Here is the research plan. Would you like to proceed?",
            actions=[
                cl.Action(name="approve", label="✅ Approve & Execute", payload={"value": "approve"}),
                cl.Action(name="revise", label="✏️ Revise Plan", payload={"value": "revise"}),
                cl.Action(name="cancel", label="🚫 Cancel", payload={"value": "cancel"}),
            ],
            timeout=300,
        ).send()

        action_value = (res.get("payload") or {}).get("value") if res else None

        if res is None or action_value == "cancel":
            return None

        if action_value == "approve":
            return prompt

        # User wants to revise — collect modification request
        mod_res = await cl.AskUserMessage(
            content="Describe the changes you'd like in the plan:",
            timeout=300,
        ).send()

        if mod_res:
            modification_context = mod_res.get("output", "")
        # Loop back to re-generate the plan


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------

async def _stream_execution(
    agent, prompt: str, history: list[dict] | None = None, thread_id: str = "42"
) -> dict | None:
    """Stream the LangGraph ReAct loop and display steps in Chainlit."""
    # Build full message list from conversation history so the agent has context
    messages = []
    for msg in (history or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg.get("content"):
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=prompt))

    inputs = {"messages": messages, "next_step": None}
    config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

    final_state = None
    solution_found = False
    code_steps: list[cl.Step] = []

    async for state in stream_langgraph(agent.app, inputs, config):
        final_state = state
        message = state["messages"][-1]
        content = message.content if isinstance(message.content, str) else ""

        if not content or content == prompt:
            continue

        # ------------------------------------------------------------------
        # Parse XML tags from the agent's raw output
        # ------------------------------------------------------------------

        # Locate first structural tag to separate reasoning prefix
        tag_positions = []
        for tag in ["<execute>", "<solution>", "<observation>"]:
            pos = content.find(tag)
            if pos != -1:
                tag_positions.append(pos)

        # 1. Reasoning / thinking (text before the first tag)
        if tag_positions:
            first_tag = min(tag_positions)
            thinking = content[:first_tag].strip()
            if thinking:
                async with cl.Step(name="🤔 Thinking", type="llm", show_input=False) as step:
                    step.output = thinking

        # 2. Solution (final answer)
        solution_match = re.search(r"<solution>(.*?)</solution>", content, re.DOTALL)
        if solution_match and not solution_found:
            solution_found = True
            solution_text = solution_match.group(1).strip()
            await cl.Message(content=solution_text).send()

        # 3. Code execution block
        execute_match = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL)
        if execute_match:
            code = execute_match.group(1).strip()
            language = "python"
            if code.startswith("#!R"):
                language = "r"
                code = re.sub(r"^#!R\s*", "", code, count=1)
            elif code.startswith("#!BASH") or code.startswith("#!CLI"):
                language = "bash"
                code = re.sub(r"^#!(BASH|CLI)\s*", "", code, count=1)

            code_step = cl.Step(name=f"⚡ Executing {language.upper()}", type="run", show_input=False)
            await code_step.__aenter__()
            code_step.output = f"```{language}\n{code}\n```"
            code_steps.append(code_step)
            # Do NOT exit the step yet; we close it when the observation arrives

        # 4. Observation (result of code execution)
        obs_match = re.search(r"<observation>(.*?)</observation>", content, re.DOTALL)
        if obs_match:
            observation = obs_match.group(1).strip()

            # Close the pending code step now that we have a result
            if code_steps:
                finished_step = code_steps.pop()
                await finished_step.__aexit__(None, None, None)

            async with cl.Step(name="👁 Observation", type="tool", show_input=False) as obs_step:
                # Truncate very long output for display
                display_obs = observation[:3000] + "\n...[truncated]" if len(observation) > 3000 else observation
                obs_step.output = display_obs

                # Display any generated images mentioned in the observation
                await _display_images(observation)

    # Close any code steps that never received an observation (edge case)
    for step in code_steps:
        await step.__aexit__(None, None, None)

    # If no <solution> tag was found, surface the last message content
    if not solution_found and final_state:
        last_content = final_state["messages"][-1].content
        if isinstance(last_content, str):
            cleaned = re.sub(r"<execute>.*?</execute>", "", last_content, flags=re.DOTALL)
            cleaned = re.sub(r"<observation>.*?</observation>", "", cleaned, flags=re.DOTALL)
            cleaned = cleaned.strip()
            if cleaned:
                await cl.Message(content=cleaned).send()

    return final_state


# ---------------------------------------------------------------------------
# Image display helper
# ---------------------------------------------------------------------------

async def _display_images(observation: str):
    """Scan observation text for image file paths and display them."""
    pattern = r"(\S+?(?:" + "|".join(re.escape(e) for e in SUPPORTED_IMAGE_EXTENSIONS) + r"))"
    for match in re.findall(pattern, observation, re.IGNORECASE):
        fp = match.strip("\"'")
        # Resolve relative or absolute path
        candidates = [fp, os.path.join(os.getcwd(), fp)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                try:
                    image = cl.Image(path=candidate, name=os.path.basename(candidate), display="inline")
                    await cl.Message(content="", elements=[image]).send()
                except Exception:
                    pass
                break


# ---------------------------------------------------------------------------
# AD1 artifact saving
# ---------------------------------------------------------------------------

async def _save_ad1_artifacts(agent, final_state: dict, initial_files: set):
    """Sync AD1 run artifacts to ./runs/ and notify the user."""
    await _save_run_artifacts_for_agent(agent, final_state, initial_files)


async def _save_run_artifacts_for_agent(
    agent,
    final_state: dict,
    initial_files: set,
    topic: str | None = None,
):
    """Save run artifacts to ./runs/ for any agent and notify the user."""
    # Reuse the pre-created run directory if the agent already has one set
    # (created before streaming so OUTPUT_DIR was available during execution).
    if hasattr(agent, "_current_run_dir") and agent._current_run_dir:
        current_run_dir = agent._current_run_dir
        run_id = os.path.basename(current_run_dir)
    else:
        run_id = _build_run_id(topic)
        runs_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
        os.makedirs(runs_root, exist_ok=True)
        current_run_dir = os.path.join(runs_root, run_id)
        os.makedirs(current_run_dir, exist_ok=True)

    # Sync internal state so artifact methods work correctly
    if "messages" in final_state:
        agent.raw_log = list(final_state["messages"])
    agent._conversation_state = final_state

    async with cl.Step(name="📦 Saving Artifacts", type="tool", show_input=False) as step:
        try:
            await run_in_executor(
                agent._save_run_artifacts, run_id, current_run_dir, initial_files
            )
            step.output = f"Artifacts saved to `{current_run_dir}`"
        except Exception as exc:
            step.output = f"⚠️ Artifact saving failed: {exc}"

    await cl.Message(
        content=f"**Run complete.** Artifacts saved to:\n`{current_run_dir}`"
    ).send()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_all_files(directory: str) -> set:
    """Recursively collect all non-hidden file paths in a directory."""
    result = set()
    for root, _, files in os.walk(directory):
        if "/." in root or root.startswith("."):
            continue
        for fname in files:
            if not fname.startswith("."):
                result.add(os.path.join(root, fname))
    return result


def _build_run_id(topic: str | None = None) -> str:
    """Build run directory ID as run_YYYYMMDD_HHMMSS_topic1_topic2_topic3."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not topic:
        return f"run_{timestamp}"

    topic_slug = _summarize_topic_for_run_id(topic)
    if not topic_slug:
        return f"run_{timestamp}"

    return f"run_{timestamp}_{topic_slug}"


def _summarize_topic_for_run_id(topic: str) -> str:
    """Extract a compact 1-3 word filesystem-safe summary from a prompt."""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "into", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
        "using", "use", "please", "can", "could", "would", "should", "do", "does",
        "analyze", "analysis", "show", "find", "run", "task", "generate", "get",
    }

    raw_tokens = re.findall(r"[A-Za-z0-9]+", topic)
    if not raw_tokens:
        return ""

    selected: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        if lower in stopwords:
            continue
        if len(lower) <= 2 and not lower.isdigit():
            continue
        selected.append(lower)
        if len(selected) == 3:
            break

    if not selected:
        selected = [t.lower() for t in raw_tokens[:3]]

    summary = "_".join(selected)
    summary = re.sub(r"[^0-9a-z_]+", "", summary)
    summary = re.sub(r"_+", "_", summary).strip("_")
    return summary[:40]
