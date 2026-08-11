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
import hashlib
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

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
from biomni.artifact import build_run_id, get_all_files
from biomni.config import default_config, resolve_default_llm
from biomni.fs_scan import scan_directory
from biomni.health import register_health_routes
from biomni.identity import UserIdentity, resolve_identity
from biomni.observability import (
    RunHeartbeat,
    bind_run,
    capture_context,
    diff_usage_summary,
    emit_event,
    log_llm_usage,
    set_session_id,
    setup_logging,
)
from biomni.run_registry import RunRecord, RunRegistry, build_run_registry
from biomni.workspace_prefs import (
    NullPrefsStore,
    OutputTarget,
    PrefsStore,
    ScopeResolution,
    WorkspacePrefs,
    build_prefs_store,
    env_flag,
    load_prefs,
    normalize_scope_entries,
    resolve_output_dir,
    resolve_scope,
)
from chainlit.input_widget import InputWidget, MultiSelect, Switch, Tags, TextInput
from langchain_core.messages import AIMessage, HumanMessage

# Configure structured (JSON) logging to stdout before anything else logs, so
# every app/agent/library line is one queryable record in the container log
# pipeline (Azure Container Insights / Log Analytics). Honors $LOG_LEVEL and
# $BIOMNI_LOG_FORMAT (json|text).
setup_logging()

# Register Kubernetes liveness (/healthz) and readiness (/readyz) probes on
# Chainlit's FastAPI app. Done at import so the routes exist before uvicorn
# starts serving. Wiring failure must never block app startup.
try:
    from chainlit.server import app as _fastapi_app

    register_health_routes(_fastapi_app)
except Exception:  # pragma: no cover - defensive: probes are non-critical to boot
    logger.warning("Could not register health endpoints", exc_info=True)

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

# Sidebar / planning helpers live in chainlit_ui/. `interactive_planning` is
# aliased to the original private name so the existing call site at the bottom
# of this file keeps working unchanged. The module-scope prompt constants are
# intentionally not re-exported — anything that needs them should
# `from chainlit_ui.planning import PLANNING_SYSTEM_PROMPT, AD1_PLANNING_SYSTEM_PROMPT`.
from chainlit_ui.datasets import build_suggested_prompts_markdown
from chainlit_ui.planning import (
    interactive_planning as _interactive_planning,
)
from chainlit_ui.workspace_panel import (
    build_previous_runs_notice,
    build_runs_panel,
    build_scope_inventory,
    build_scope_panel,
    list_top_level_dirs,
    scope_choice_items,
    summarize_scope,
)
from chainlit_ui.workspace_panel import (
    build_tree_preview_lines as _build_tree_preview_lines,
)

DEFAULT_LLM = resolve_default_llm()
DEFAULT_PATH = os.getenv("BIOMNI_PATH", "./data")
# Set BIOMNI_AGENT=a1 to force the A1 agent on startup (skips the profile selector)
FORCE_AGENT = os.getenv("BIOMNI_AGENT", "").lower()  # "a1" | "ad1" | ""

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

CHAINLIT_MD_PATH = Path(__file__).with_name("chainlit.md")
# ``chainlit.md`` itself is gitignored — it's rewritten on every launch with
# the local data inventory, producing a spurious diff on every dev machine.
# ``chainlit.md.template`` is the source of truth in git: same content with
# the managed marker blocks empty.
CHAINLIT_MD_TEMPLATE_PATH = Path(__file__).with_name("chainlit.md.template")
_WELCOME_DATASET_BLOCK_START = "<!-- BIOMNI_LOCAL_DATASET_SECTION_START -->"
_WELCOME_DATASET_BLOCK_END = "<!-- BIOMNI_LOCAL_DATASET_SECTION_END -->"
_SUGGESTED_PROMPTS_BLOCK_START = "<!-- BIOMNI_SUGGESTED_PROMPTS_START -->"
_SUGGESTED_PROMPTS_BLOCK_END = "<!-- BIOMNI_SUGGESTED_PROMPTS_END -->"


def _build_ad_suggested_prompts() -> str:
    """Thin shim over chainlit_ui.datasets — resolves the repo-local AD lake path."""
    ad_lake = Path(__file__).resolve().parent / "data" / "biomni_data" / "data_lake" / "biomniAD"
    return build_suggested_prompts_markdown(ad_lake)


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


def _list_path_entries_recursive(
    path: str, max_items: int = 80, max_depth: int = 10, exclude_top_subdirs: set[str] | None = None
) -> tuple[list[str], int]:
    """Recursively list non-hidden files under a directory.

    Returns a (preview_items, total_file_count) tuple. Preview items are
    relative POSIX-style paths suitable for UI display.

    Thin adapter over :func:`biomni.fs_scan.scan_directory` — the scan is
    bounded (file cap + wall-clock deadline) and cached, so this never hangs on
    a large or network-backed workspace. ``total_file_count`` is a lower bound
    when the underlying scan was truncated; callers that display it should treat
    it as approximate (see ``_build_user_data_tree_content``).
    """
    result = scan_directory(path, max_depth=max_depth, exclude_top_subdirs=exclude_top_subdirs)
    return result.files[:max_items], result.file_count


def _collect_path_stats(path: str, max_depth: int = 10, exclude_top_subdirs: set[str] | None = None) -> dict:
    """Collect compact stats for a directory tree for sidebar summaries.

    ``exclude_top_subdirs`` names top-level subdirectories to skip entirely
    (useful for counting the datalake root without the biomniAD subfolder).

    Backed by the bounded/cached scanner; ``truncated`` is True when the counts
    are a floor (workspace larger than the scan budget).
    """
    result = scan_directory(path, max_depth=max_depth, exclude_top_subdirs=exclude_top_subdirs)
    return {
        "total_files": result.file_count,
        "total_dirs": result.dir_count,
        "top_level_counts": dict(result.top_level_counts),
        "extension_counts": dict(result.extension_counts),
        "truncated": result.bounded,
    }


def _format_compact_counts(counts: dict[str, int], max_items: int = 8) -> str:
    """Format a frequency map as a compact markdown bullet list."""
    if not counts:
        return "- *(none)*"
    top_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_items]
    return "\n".join(f"- `{name}`: {value}" for name, value in top_items)


def _build_user_data_tree_content(root_path: str, preview_files: int = 300) -> tuple[str, int]:
    """Build concise per-root content for sidebar readability."""
    result = scan_directory(root_path, max_depth=10)
    if result.file_count == 0:
        # Distinguish a genuinely empty root from a scan that hit the
        # time/size budget before reading any file (slow network mount) — the
        # latter must not masquerade as "no files".
        if result.bounded:
            return (
                "(listing unavailable — the workspace scan hit its time/size budget before any file "
                "was read; the folder is likely very large or on a slow mount. Open it directly to browse.)",
                0,
            )
        return "(no files found)", 0

    preview = result.files[:preview_files]
    tree_lines = _build_tree_preview_lines(preview, max_lines=70, max_depth=3)
    lines: list[str] = [
        f"Directory structure preview (showing first {len(preview)} files)",
        "",
        *tree_lines,
    ]
    if result.bounded:
        # Workspace exceeded the scan budget: the tree is a partial sample.
        lines.append(
            f"... workspace is large — listing capped at {result.count_label()} files "
            "(not fully indexed; scan bounded for responsiveness)"
        )
    elif result.file_count > len(preview):
        lines.append(f"... and {result.file_count - len(preview)} more files")

    return "\n".join(lines), result.file_count


def _root_count_label(root_path: str, exclude_top_subdirs: set[str] | None = None) -> str:
    """Cached file count for a root, with a trailing ``+`` when the scan was
    bounded (a floor, not an exact total). Reuses the cached scan so calling it
    for a tab label right after building the tree is free."""
    return scan_directory(root_path, max_depth=10, exclude_top_subdirs=exclude_top_subdirs).count_label()


def _build_sidebar_overview_content() -> str:
    """At-a-glance counts for the built-in data lakes. Empty string when absent.

    Sections (only shown when they have files):
    1. Biomni-AD Datalake - data_lake/biomniAD/ subfolder.
    2. Biomni Datalake    - root of data_lake/ excluding biomniAD.
    """
    lines: list[str] = ["At-a-glance overview", ""]
    grand_files = 0
    grand_dirs = 0
    grand_truncated = False

    def _add(label: str, stats: dict) -> None:
        # Render one section line and fold its counts into the grand totals. A
        # trailing "+" signals the scan was bounded (workspace larger than the
        # scan budget), so the number is a floor, not an exact count.
        nonlocal grand_files, grand_dirs, grand_truncated
        if stats["total_files"] <= 0:
            return
        grand_files += stats["total_files"]
        grand_dirs += stats["total_dirs"]
        grand_truncated = grand_truncated or stats.get("truncated", False)
        plus = "+" if stats.get("truncated") else ""
        lines.append(f"{label}: {stats['total_files']}{plus} files, {stats['total_dirs']}{plus} folders")
        lines.append("")

    # The user's own workspace is deliberately absent from this overview. Its
    # totals used to be computed with a full recursive walk on every session
    # start, which is the cost this change removes; the Workspace panel reports
    # counts for the folders the user actually selected instead.

    # --- 1. Biomni-AD Datalake ----------------------------------------------
    builtin_root = _resolve_builtin_data_lake_root()
    biomni_ad_root = os.path.join(builtin_root, "biomniAD")
    if os.path.isdir(biomni_ad_root):
        _add("Biomni-AD Datalake", _collect_path_stats(biomni_ad_root))

    # --- 2. Biomni Datalake (root of data_lake, excluding biomniAD) ----------
    if os.path.isdir(builtin_root):
        _add("Biomni Datalake", _collect_path_stats(builtin_root, exclude_top_subdirs={"biomniAD"}))

    if grand_files == 0 and grand_dirs == 0:
        return ""

    plus = "+" if grand_truncated else ""
    lines.extend(
        [
            "Combined totals",
            f"Files: {grand_files}{plus}",
            f"Folders: {grand_dirs}{plus}",
        ]
    )
    if grand_truncated:
        lines.append("(partial — workspace exceeded the scan budget; counts are a lower bound)")
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

    # Bounded/cached scan so this can't stall process startup even if the
    # built-in data lake is pointed at a large volume.
    result = scan_directory(str(data_lake_dir))
    items = sorted(f for f in result.files if f != "_custom_data_index.json")
    return items[:max_items]


def _build_welcome_local_dataset_section() -> str:
    """Build a collapsed markdown block shown at the bottom of the Chainlit welcome page."""
    # Data lake always sourced from the repo-local built-in location
    _repo_root = Path(__file__).resolve().parent
    builtin_data_lake = _repo_root / "data" / "biomni_data" / "data_lake"
    data_lake_files = _list_local_data_lake_files(str(_repo_root / "data"), max_items=200)

    # User data can come from BIOMNI_USER_DATA_PATH and legacy BIOMNI_DATA_PATH/BIOMNI_PATH.
    #
    # Only the top-level folder NAMES are read here, with a single directory
    # listing. This function runs at import time, before any session exists, so
    # a recursive scan of the workspace would put a large-mount traversal on the
    # process-startup path where nothing can bound its effect on first paint.
    # The per-session workspace panel is where the user picks a scope and sees
    # real file counts.
    user_roots = _resolve_user_data_roots()
    user_folders: list[str] = []
    if user_roots:
        user_folders = list_top_level_dirs(user_roots[0][1])

    # One-line summary for the collapsed header
    summary_parts = [f"{len(data_lake_files)} data lake files"]
    if user_folders:
        summary_parts.append(f"{len(user_folders)} user data folders")

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
        lines.append(
            "**User Data** — from `BIOMNI_USER_DATA_HOST_PATH` / `BIOMNI_USER_DATA_PATH` / `BIOMNI_DATA_PATH` / `BIOMNI_PATH`"
        )
        lines.append("")
        primary_label = _display_data_root_label(user_roots[0][0])
        lines.append(f"Primary path ({primary_label}): `{user_roots[0][1]}`")
        if len(user_roots) > 1:
            lines.append("Additional configured paths:")
            for env_name, root in user_roots[1:]:
                lines.append(f"- `{_display_data_root_label(env_name)}`: `{root}`")
        lines.append("")
        if user_folders:
            lines.append("Top-level folders:")
            for name in user_folders[:40]:
                lines.append(f"- `{name}/`")
            if len(user_folders) > 40:
                lines.append(f"- `... and {len(user_folders) - 40} more`")
            lines.append("")
            lines.append("_Choose which of these the agent should use in ⚙️ Settings._")
        else:
            lines.append("- *(none found)*")

    lines.append("")
    lines.append("</details>")

    return "\n".join(lines)


def _refresh_chainlit_welcome_markdown() -> None:
    """Append or replace managed sections (suggested prompts + dataset list) in chainlit.md.

    Reads from ``chainlit.md`` if it already exists (preserves any operator
    edits between launches), else from ``chainlit.md.template`` (the tracked
    source of truth), else from a hardcoded minimal fallback.
    """
    try:
        if CHAINLIT_MD_PATH.exists():
            original = CHAINLIT_MD_PATH.read_text(encoding="utf-8")
        elif CHAINLIT_MD_TEMPLATE_PATH.exists():
            original = CHAINLIT_MD_TEMPLATE_PATH.read_text(encoding="utf-8")
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
            (f"\n\n{_SUGGESTED_PROMPTS_BLOCK_START}\n{suggested}\n{_SUGGESTED_PROMPTS_BLOCK_END}\n")
            if suggested
            else ""
        )

        # Build dataset inventory block
        dataset_section = _build_welcome_local_dataset_section()
        dataset_block = f"\n\n{_WELCOME_DATASET_BLOCK_START}\n{dataset_section}\n{_WELCOME_DATASET_BLOCK_END}\n"

        CHAINLIT_MD_PATH.write_text(base + prompts_block + dataset_block, encoding="utf-8")
    except Exception:
        logger.warning("Could not refresh chainlit welcome markdown", exc_info=True)


_refresh_chainlit_welcome_markdown()

# ---------------------------------------------------------------------------
# Dataset listing helper
# ---------------------------------------------------------------------------

_DATALAKE_CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "Protein Interactions",
        [
            "affinity_capture-ms",
            "affinity_capture-rna",
            "co-fractionation",
            "proximity_label-ms",
            "reconstituted_complex",
            "two-hybrid",
            "Virus-Host_PPI_P-HIPSTER_2020",
        ],
    ),
    (
        "Drug & Compound Data",
        [
            "BindingDB_All_202409",
            "broad_repurposing_hub_molecule_with_smiles",
            "broad_repurposing_hub_phase_moa_target_info",
            "enamine_cloud_library_smiles",
            "ddinter_alimentary_tract_metabolism",
            "ddinter_antineoplastic",
            "ddinter_antiparasitic",
            "ddinter_blood_organs",
            "ddinter_dermatological",
            "ddinter_hormonal",
            "ddinter_respiratory",
            "ddinter_various",
        ],
    ),
    (
        "Gene Expression & Cancer",
        [
            "DepMap_CRISPRGeneDependency",
            "DepMap_CRISPRGeneEffect",
            "DepMap_Model",
            "DepMap_OmicsExpressionProteinCodingGenesTPMLogp1",
            "gtex_tissue_gene_tpm",
            "proteinatlas",
        ],
    ),
    (
        "Genomics & Genetic Variants",
        [
            "genebass_missense_LC_filtered",
            "genebass_pLoF_filtered",
            "genebass_synonymous_filtered",
            "gwas_catalog",
            "variant_table",
            "sgRNA_KO_SP_human",
            "sgRNA_KO_SP_mouse",
        ],
    ),
    (
        "Gene Sets & Functional Annotations",
        [
            "msigdb_human_c1_positional_geneset",
            "msigdb_human_c2_curated_geneset",
            "msigdb_human_c3_regulatory_target_geneset",
            "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD",
            "msigdb_human_c4_computational_geneset",
            "msigdb_human_c5_ontology_geneset",
            "msigdb_human_c6_oncogenic_signature_geneset",
            "msigdb_human_c7_immunologic_signature_geneset",
            "msigdb_human_c8_celltype_signature_geneset",
            "msigdb_human_h_hallmark_geneset",
            "mousemine_m1_positional_geneset",
            "mousemine_m2_curated_geneset",
            "mousemine_m3_regulatory_target_geneset",
            "mousemine_m5_ontology_geneset",
            "mousemine_m8_celltype_signature_geneset",
            "mousemine_mh_hallmark_geneset",
            "go-plus",
            "gene_info",
        ],
    ),
    (
        "Disease & Phenotype",
        [
            "DisGeNET",
            "omim",
            "hp",
            "kg",
        ],
    ),
    (
        "Cell Biology",
        [
            "czi_census_datasets_v4",
            "marker_celltype",
        ],
    ),
    (
        "RNA Biology",
        [
            "miRDB_v6.0_results",
            "miRTarBase_microRNA_target_interaction",
            "miRTarBase_microRNA_target_interaction_pubmed_abtract",
            "miRTarBase_MicroRNA_Target_Sites",
        ],
    ),
    (
        "Genetic Interactions",
        [
            "dosage_growth_defect",
            "genetic_interaction",
            "synthetic_growth_defect",
            "synthetic_lethality",
            "synthetic_rescue",
        ],
    ),
    (
        "Immunology & Other",
        [
            "McPAS-TCR",
            "txgnn_name_mapping",
            "txgnn_prediction",
        ],
    ),
]


def _build_dataset_listing(agent) -> str:
    """Return a compact sidebar summary (tree details are in separate elements)."""
    _ = agent
    user_roots = _resolve_user_data_roots()
    builtin_root = _resolve_builtin_data_lake_root()
    if not user_roots and not os.path.isdir(builtin_root):
        return "No local data tree available"
    return "Open a Tree item below"


# ---------------------------------------------------------------------------
# Workspace session: identity, preferences, scope, outputs, run history
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceSession:
    """Everything one chat session needs to know about the user's workspace.

    Built once per session (off the event loop - it touches the filesystem) and
    recomputed whenever the user changes their settings. Holding it in one
    object keeps the sidebar, the agent's system prompt, the output directory
    and the run registry from drifting apart, which is how the previous code
    ended up describing one set of files to the user and another to the agent.
    """

    identity: UserIdentity
    prefs: WorkspacePrefs
    prefs_key: str
    store: PrefsStore
    registry: RunRegistry
    workspace_root: str | None
    scope: ScopeResolution
    output: OutputTarget
    top_level_dirs: list[str] = field(default_factory=list)
    inventory: str = ""
    runs: list[RunRecord] = field(default_factory=list)

    @property
    def persistence_label(self) -> str:
        return self.store.describe


def _resolve_workspace_root() -> str | None:
    """The user's workspace: the first configured user-data root that exists.

    Preferences, outputs and run records all hang off this one path, so it is
    resolved through the same env precedence the rest of the UI already uses
    rather than a second, subtly different rule.
    """
    for _env_name, root in _resolve_user_data_roots():
        if os.path.isdir(root):
            return root
    return None


def _refresh_workspace_view(ws: WorkspaceSession) -> WorkspaceSession:
    """Recompute everything derived from preferences. Blocking; run off-loop.

    Scanning happens here and only here, and only over the folders the user
    selected. With no selection this costs a single directory listing.
    """
    ws.scope = resolve_scope(ws.prefs, ws.workspace_root)
    ws.output = resolve_output_dir(ws.prefs, workspace_root=ws.workspace_root)
    # The output directory usually sits inside the workspace and gains a run
    # folder per query; it is a destination, never an input, so keep it out of
    # the picker and out of what the agent is told it can read.
    ws.top_level_dirs = list_top_level_dirs(ws.workspace_root, exclude_paths=[ws.output.path])
    ws.inventory = build_scope_inventory(ws.scope, ws.workspace_root, top_level_dirs=ws.top_level_dirs)
    return ws


def _build_workspace_session(session_id: str, header_source: object) -> WorkspaceSession:
    """Resolve identity, load stored preferences and derive the session view.

    Blocking (filesystem + preference load), so callers must run it in an
    executor. Never raises: any failure degrades to defaults, because a user
    who cannot load their settings should still get a working session.
    """
    identity = resolve_identity(header_source, session_id=session_id)
    workspace_root = _resolve_workspace_root()
    prefs_key = identity.scoped_key()

    # An anonymous key is per-connection, so anything stored under it can never
    # be read back - it would just accumulate orphaned directories on the state
    # volume, one per page load. Persist only for an identified user, unless a
    # deployment opts in (useful for local development, where there is no
    # gateway but you still want to exercise the feature).
    if identity.is_authenticated or env_flag("BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE"):
        store = build_prefs_store(workspace_root)
        registry = build_run_registry(workspace_root)
    else:
        store = NullPrefsStore("no authenticated user; set BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE=true to override")
        registry = RunRegistry(None)

    prefs = load_prefs(store, prefs_key)

    ws = WorkspaceSession(
        identity=identity,
        prefs=prefs,
        prefs_key=prefs_key,
        store=store,
        registry=registry,
        workspace_root=workspace_root,
        scope=resolve_scope(prefs, workspace_root),
        output=resolve_output_dir(prefs, workspace_root=workspace_root),
    )
    # Reconciling here (rather than at write time) is what turns a pod restart
    # into an honest "interrupted" on the user's next visit.
    ws.runs = registry.reconcile(prefs_key)
    return _refresh_workspace_view(ws)


def _workspace_settings_widgets(ws: WorkspaceSession) -> list[InputWidget]:
    """Chat-settings controls for scope and output directory.

    The folder multi-select is omitted entirely when the workspace has no
    subdirectories - Chainlit's MultiSelect rejects an empty item list, and an
    empty picker would be noise anyway.
    """
    widgets: list[InputWidget] = []
    known = set(ws.top_level_dirs)

    if ws.top_level_dirs:
        widgets.append(
            MultiSelect(
                id="scope_folders",
                label="Data folders the agent may use",
                items=scope_choice_items(ws.top_level_dirs),
                initial=[p for p in ws.prefs.scope_paths if p in known],
                description=(
                    "Only the selected folders are scanned and described to the agent. "
                    "Selecting large folders slows every session and dilutes the agent's attention, "
                    "so prefer the specific studies you are working on. "
                    "With nothing selected, the agent is told which folders exist and looks inside "
                    "them only when a task calls for it."
                ),
            )
        )

    widgets.append(
        Tags(
            id="scope_extra_paths",
            label="Additional paths (workspace-relative)",
            initial=[p for p in ws.prefs.scope_paths if p not in known],
            description="Sub-folders deeper than the top level, e.g. studyA/processed",
        )
    )

    widgets.append(
        TextInput(
            id="output_dir",
            label="Output directory",
            initial=ws.prefs.output_dir or ws.output.path,
            description=(
                f"Where run results are written. Currently resolved to {ws.output.path} "
                f"(source: {ws.output.source}). Leave as-is to keep the deployment default."
            ),
        )
    )

    widgets.append(
        Switch(
            id="remember_settings",
            label="Remember these settings for my next session",
            initial=ws.prefs.remember,
        )
    )
    return widgets


def _apply_workspace_settings(ws: WorkspaceSession, settings: dict) -> tuple[WorkspaceSession, bool]:
    """Fold submitted settings into preferences and recompute. Blocking.

    Returns the session and whether the preferences were persisted, so the
    caller can tell the user the truth about whether their choice will survive
    the session (it will not when no writable store exists).
    """
    folders = settings.get("scope_folders") or []
    extra = settings.get("scope_extra_paths") or []
    if isinstance(folders, str):
        folders = [folders]
    if isinstance(extra, str):
        extra = [extra]

    ws.prefs.scope_paths = normalize_scope_entries([*folders, *extra], ws.workspace_root)

    submitted_output = (settings.get("output_dir") or "").strip()
    # The field is pre-filled with the resolved path, so an untouched form must
    # not turn today's default into a pinned preference that would survive a
    # deployment moving its volumes. Compare against what the deployment would
    # resolve to with NO preference set - comparing against the currently
    # resolved path would also match an already-pinned directory, silently
    # unpinning it every time the user re-saved any other setting.
    deployment_default = resolve_output_dir(replace(ws.prefs, output_dir=None), workspace_root=ws.workspace_root).path
    if not submitted_output or submitted_output == deployment_default:
        ws.prefs.output_dir = None
    else:
        ws.prefs.output_dir = submitted_output

    ws.prefs.remember = bool(settings.get("remember_settings", True))
    ws.prefs.email = ws.identity.email

    persisted = False
    if ws.prefs.remember:
        persisted = ws.store.save(ws.prefs_key, ws.prefs)
    else:
        # Opting out must forget what was already stored, otherwise the old
        # scope silently returns on the next visit and the toggle looks broken.
        ws.store.delete(ws.prefs_key)

    _refresh_workspace_view(ws)
    return ws, persisted


def _build_workspace_sidebar_elements(ws: WorkspaceSession) -> list[cl.Text]:
    """Sidebar pages: the active scope, run history, then the built-in lakes.

    Replaces the old flat dump of every workspace file, which reviewers
    correctly called out as unusable: it could not be acted on, and producing
    it required the full traversal this change exists to avoid.
    """
    elements: list[cl.Text] = [
        cl.Text(
            name="Workspace",
            content=build_scope_panel(
                ws.scope,
                ws.workspace_root,
                ws.output,
                summaries=summarize_scope(ws.scope, ws.workspace_root),
                top_level_dirs=ws.top_level_dirs,
                persistence=ws.persistence_label,
            ),
            display="page",
        ),
        cl.Text(name="Recent runs", content=build_runs_panel(ws.runs), display="page"),
    ]
    elements.extend(_build_builtin_datalake_elements())
    return elements


async def _render_workspace_sidebar(ws: WorkspaceSession) -> None:
    """Rebuild and push the sidebar. Element building is blocking, so off-loop."""
    elements = await run_in_executor(_build_workspace_sidebar_elements, ws)
    await cl.ElementSidebar.set_elements(elements, key=_sidebar_key(elements))


def _sidebar_key(elements: list[cl.Text]) -> str:
    """Content-derived key for ``ElementSidebar.set_elements``.

    Chainlit skips replacing the sidebar when it is already open with the same
    key, and the default key is ``None`` - so a second call with fresh content
    is silently ignored and the panel keeps showing the old scope. Keying on a
    digest of the rendered content makes an update land exactly when something
    actually changed, and costs nothing when it has not.
    """
    payload = "\x00".join(f"{el.name}:{getattr(el, 'content', '')}" for el in elements)
    return "workspace-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_builtin_datalake_elements() -> list[cl.Text]:
    """Sidebar trees for the repo-local data lakes.

    These stay a full listing: they are curated, bounded and identical for every
    user, so none of the large-workspace concerns apply.
    """
    elements: list[cl.Text] = []
    summary = _build_sidebar_overview_content()
    if summary:
        elements.append(cl.Text(name="Data lake summary", content=summary, display="page"))

    # --- Built-in datalake trees ---------------------------------------------
    builtin_root = _resolve_builtin_data_lake_root()

    # Biomni-AD Datalake
    biomni_ad_root = os.path.join(builtin_root, "biomniAD")
    if os.path.isdir(biomni_ad_root):
        tree_content, total_files = _build_user_data_tree_content(biomni_ad_root)
        if total_files > 0:
            content = f"Path: {biomni_ad_root}\n\n{tree_content}"
            elements.append(
                cl.Text(
                    name=f"Tree [Biomni-AD Datalake] ({_root_count_label(biomni_ad_root)})",
                    content=content,
                    display="page",
                )
            )

    # Biomni Datalake (root, excluding biomniAD)
    if os.path.isdir(builtin_root):
        lake = scan_directory(builtin_root, max_depth=10, exclude_top_subdirs={"biomniAD"})
        if lake.file_count > 0:
            preview = lake.files[:300]
            tree_lines = _build_tree_preview_lines(preview, max_lines=70, max_depth=3)
            lake_lines: list[str] = [
                f"Directory structure preview (showing first {len(preview)} files)",
                "",
                *tree_lines,
            ]
            if lake.bounded:
                lake_lines.append(f"... listing capped at {lake.count_label()} files (scan bounded for responsiveness)")
            elif lake.file_count > len(preview):
                lake_lines.append(f"... and {lake.file_count - len(preview)} more files")
            content = f"Path: {builtin_root}\n\n" + "\n".join(lake_lines)
            elements.append(
                cl.Text(name=f"Tree [Biomni Datalake] ({lake.count_label()})", content=content, display="page")
            )

    return elements


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def run_in_executor(fn, *args):
    """Run a synchronous function in a thread-pool executor.

    The caller's context is captured *here* (in the event-loop thread) and
    replayed inside the worker, so correlation ids (session_id / run_id)
    propagate into agent-side logs — executors do not copy contextvars on their
    own. Capturing inside the worker would snapshot its empty context instead.
    """
    ctx = capture_context()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: ctx.run(fn, *args))


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
    # Capture the active context here (caller thread) and replay it in the
    # producer thread so the agent graph's logs (LLM calls, code-execution
    # audit) carry the session_id/run_id.
    ctx = capture_context()
    executor.submit(ctx.run, _producer)

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
    # One correlation id per chat session, bound for the lifetime of this
    # handler so agent-init logs carry it. on_message rebinds it per message.
    thread_id = uuid.uuid4().hex
    set_session_id(thread_id)

    # Determine agent type: CLI env var > chat profile selection > default AD1
    if FORCE_AGENT in ("a1", "ad1"):
        agent_type = FORCE_AGENT
    else:
        profile = cl.user_session.get("chat_profile", "AD1")
        agent_type = "a1" if str(profile).upper() == "A1" else "ad1"

    label = "AD1" if agent_type == "ad1" else "A1"
    emit_event("chat_start", agent_type=agent_type, llm=DEFAULT_LLM)
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
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("dataset_listing", _build_dataset_listing(agent))
        cl.user_session.set("dataset_panel_shown", False)
    except Exception as exc:
        logger.exception("Failed to initialize %s agent", label)
        await cl.Message(content=f"Failed to initialize {label}: {exc}").send()
        return

    # Resolve identity, preferences, data scope and output directory, then tell
    # the agent what it may look at. All of it touches the filesystem, so it
    # MUST run in a worker thread: doing it inline would block the asyncio event
    # loop, starving the /healthz liveness probe and getting the pod restarted.
    # Guarded so a settings/scan hiccup degrades to a default session rather
    # than failing an otherwise usable one.
    ws = None
    try:
        ws = await run_in_executor(_build_workspace_session, thread_id, _session_header_source())
        cl.user_session.set("workspace", ws)
        _apply_scope_to_agent(agent, ws)
        emit_event(
            "workspace_session",
            identity_source=ws.identity.source,
            has_workspace=bool(ws.workspace_root),
            scope_folders=len(ws.scope.roots),
            scope_is_default=ws.scope.is_default,
            output_source=ws.output.source,
            output_ephemeral=ws.output.is_ephemeral,
            prefs_persisted=not isinstance(ws.store, NullPrefsStore),
        )
    except Exception:
        logger.exception("Failed to resolve workspace session (continuing with defaults)")

    # Expose the scope / output controls.
    if ws is not None:
        try:
            await cl.ChatSettings(_workspace_settings_widgets(ws)).send()
        except Exception:
            logger.exception("Failed to render workspace settings (session remains usable)")

    # Render the sidebar: active scope, run history, then the built-in lakes.
    try:
        if ws is not None:
            sidebar_elements = await run_in_executor(_build_workspace_sidebar_elements, ws)
        else:
            sidebar_elements = await run_in_executor(_build_builtin_datalake_elements)
        if not sidebar_elements:
            sidebar_content = cl.user_session.get("dataset_listing") or _build_dataset_listing(agent)
            sidebar_elements = [cl.Text(name="Local Datasets", content=sidebar_content)]
        await cl.ElementSidebar.set_title("Workspace")
        await cl.ElementSidebar.set_elements(sidebar_elements, key=_sidebar_key(sidebar_elements))
        cl.user_session.set("dataset_panel_shown", True)
    except Exception:
        logger.exception("Failed to render local-data sidebar (session remains usable)")

    # Tell the user about work that did not finish while they were away. Only
    # unfinished runs interrupt them; completed ones wait in the sidebar.
    if ws is not None:
        try:
            notice = build_previous_runs_notice(ws.runs)
            if notice:
                await cl.Message(content=notice).send()
        except Exception:
            logger.exception("Failed to render previous-runs notice")


def _session_header_source() -> object:
    """The gateway headers for this session, or None outside a request context.

    Chainlit keeps the connection's WSGI/ASGI environ on the session; that is
    where an authentication gateway's forwarded headers arrive. Best-effort by
    design - with no gateway (local dev, today's deployment) this yields None
    and identity falls back to an anonymous, session-scoped key.
    """
    try:
        session = cl.context.session
    except Exception:
        return None
    return getattr(session, "environ", None) or getattr(session, "http_headers", None)


def _apply_scope_to_agent(agent, ws: WorkspaceSession) -> None:
    """Point the agent at the selected scope and output directory.

    ``user_data_inventory`` is what the system prompt renders for the data root
    (see A1._generate_system_prompt), ``scope_roots`` narrows the retriever's
    user-data index so it stops surfacing files the user excluded, and
    ``runs_root`` makes the agent's own run directories land where the session
    resolved outputs to (A1._resolve_runs_root) instead of the process cwd.
    """
    agent.user_data_inventory = ws.inventory or ""
    agent.scope_roots = list(ws.scope.roots)
    agent.runs_root = ws.output.path


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Apply a scope / output-directory change and re-render everything from it."""
    ws = cl.user_session.get("workspace")
    agent = cl.user_session.get("agent")
    if ws is None:
        return

    try:
        ws, persisted = await run_in_executor(_apply_workspace_settings, ws, settings)
    except Exception:
        logger.exception("Failed to apply workspace settings")
        await cl.Message(content="⚠️ Could not apply those settings; the previous scope is still active.").send()
        return

    cl.user_session.set("workspace", ws)
    if agent is not None:
        _apply_scope_to_agent(agent, ws)

    try:
        await _render_workspace_sidebar(ws)
    except Exception:
        logger.exception("Failed to refresh sidebar after settings update")

    emit_event(
        "workspace_settings_updated",
        scope_folders=len(ws.scope.roots),
        scope_is_default=ws.scope.is_default,
        output_source=ws.output.source,
        persisted=persisted,
    )

    if ws.scope.roots:
        scope_text = ", ".join(f"`{label}`" for label in ws.prefs.scope_paths[:8])
        if len(ws.prefs.scope_paths) > 8:
            scope_text += f" and {len(ws.prefs.scope_paths) - 8} more"
        summary = f"**Data scope updated.** The agent will use {scope_text}."
    else:
        summary = "**Data scope cleared.** The agent will only look inside workspace folders a task points it to."

    summary += f"\n\nOutputs: `{ws.output.path}`"
    if not ws.output.writable:
        summary += f"\n\n⚠️ That directory is not writable. {ws.output.reason or ''}".rstrip()
    elif ws.output.is_ephemeral:
        summary += "\n\n⚠️ Container-local storage: results are lost when the application restarts."

    if ws.prefs.remember and not persisted:
        summary += "\n\n_Note: settings could not be saved, so they apply to this session only._"

    await cl.Message(content=summary).send()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


def _usage_snapshot(agent) -> dict | None:
    """Best-effort snapshot of an agent's cumulative LLM usage."""
    try:
        if agent is not None and hasattr(agent, "llm_usage_summary"):
            return agent.llm_usage_summary()
    except Exception:
        logger.debug("usage snapshot failed", exc_info=True)
    return None


def _emit_run_telemetry(agent, usage_before: dict | None, started: float) -> None:
    """Emit a per-run ``llm_usage`` event (token counts + cost + latency).

    Gated on ``enable_llm_telemetry`` (env ``BIOMNI_ENABLE_LLM_TELEMETRY``); the
    delta is computed against the snapshot taken before the run so the numbers
    are per-message, not cumulative across the chat. Never raises.
    """
    if not getattr(default_config, "enable_llm_telemetry", False):
        return
    try:
        usage_after = _usage_snapshot(agent)
        if usage_before is None or usage_after is None:
            return
        delta = diff_usage_summary(usage_before, usage_after)
        if delta.get("calls", 0) <= 0:
            return
        log_llm_usage(
            delta,
            model=DEFAULT_LLM,
            source=getattr(agent, "_llm_source", None),
            latency_ms=round((time.monotonic() - started) * 1000, 1),
            agent_type=cl.user_session.get("agent_type"),
        )
    except Exception:
        logger.debug("run telemetry emission failed", exc_info=True)


@cl.on_message
async def on_message(message: cl.Message):
    """Bind per-message correlation ids and emit run telemetry around the handler."""
    thread_id = cl.user_session.get("thread_id", "unknown")
    run_id = uuid.uuid4().hex
    agent = cl.user_session.get("agent")
    usage_before = _usage_snapshot(agent)
    started = time.monotonic()
    with bind_run(session_id=thread_id, run_id=run_id):
        try:
            await _process_message(message)
        finally:
            _emit_run_telemetry(agent, usage_before, started)


async def _process_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type", "a1")

    if agent is None:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
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
                step.output = "Specialized AD/dementia data sourcing protocols injected into context."

    # ------------------------------------------------------------------
    # Phase 2: Tool retrieval
    # ------------------------------------------------------------------
    if getattr(agent, "use_tool_retriever", False):
        async with cl.Step(name="🔍 Selecting Resources", type="retrieval", show_input=False) as step:
            try:
                resources = await run_in_executor(agent._prepare_resources_for_retrieval, prompt)
                if resources:
                    await run_in_executor(agent.update_system_prompt_with_selected_resources, resources)
                    tools_n = len(resources.get("tools", []))
                    data_n = len(resources.get("data_lake", []))
                    libs_n = len(resources.get("libraries", []))
                    knowhow_n = len(resources.get("know_how", []))
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
                logger.warning("Tool retrieval failed; falling back to full tool set", exc_info=True)
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

    # Pre-create the run directory so OUTPUT_DIR is available during code
    # execution. The root comes from the session's resolved output target
    # (user preference -> BIOMNI_OUTPUT_ROOT -> workspace -> cwd/runs) rather
    # than being hardcoded to the working directory, so results land somewhere
    # the user can reach after the pod restarts.
    ws = cl.user_session.get("workspace")
    _run_id = build_run_id(prompt)
    _current_run_dir = None
    try:
        _runs_root = ws.output.path if ws is not None else os.path.abspath(os.path.join(os.getcwd(), "runs"))
        os.makedirs(_runs_root, exist_ok=True)
        _current_run_dir = os.path.join(_runs_root, _run_id)
        os.makedirs(_current_run_dir, exist_ok=True)
        agent._current_run_dir = _current_run_dir
        os.environ["BIOMNI_OUTPUT_PATH"] = _current_run_dir
    except OSError:
        logger.warning("Could not pre-create run directory under %s", _runs_root, exc_info=True)
        await cl.Message(
            content=(
                f"⚠️ Could not write to the output directory `{_runs_root}`. "
                "Results may not be saved - set a writable path in ⚙️ Settings."
            )
        ).send()
    except Exception:
        logger.warning("Could not pre-create run directory", exc_info=True)

    # Record the run durably before it starts, so a user who closes the tab (or
    # a pod that restarts mid-run) still has a trace of it next session.
    record = None
    if ws is not None and ws.registry.enabled:
        try:
            record = ws.registry.start(
                ws.prefs_key,
                _run_id,
                prompt=prompt,
                agent_type=agent_type,
                output_dir=_current_run_dir,
                session_id=thread_id,
                workspace_id=ws.identity.workspace_id,
            )
        except Exception:
            logger.warning("Could not record run start", exc_info=True)

    # Snapshot files before execution (cwd + data root) to detect new outputs.
    initial_files = get_all_files(os.getcwd())
    _data_root = getattr(agent, "data_root_dir", None)
    if _data_root and os.path.isdir(_data_root):
        initial_files |= get_all_files(_data_root)

    # Default to the pessimistic outcome and upgrade only on success. Closing
    # the tab or a SIGTERM raises asyncio.CancelledError, which is NOT an
    # Exception subclass, so it bypasses the handler below and lands straight in
    # the finally block - starting from "completed" would durably record an
    # abandoned run as finished, the exact false positive the registry exists to
    # prevent, and reconcile() would never correct a terminal record.
    run_status, run_error = "interrupted", "the run was cancelled before it finished"
    try:
        final_state = await _stream_execution(
            agent,
            prompt,
            history,
            thread_id,
            on_heartbeat=(lambda: ws.registry.heartbeat(ws.prefs_key, record)) if record is not None else None,
        )
        run_status, run_error = "completed", None
    except Exception as exc:
        # Close the record out before the exception propagates, otherwise the
        # run stays "running" until a later session reconciles it as stale.
        run_status, run_error = "failed", str(exc)
        raise
    finally:
        if record is not None:
            try:
                ws.registry.finish(ws.prefs_key, record, status=run_status, error=run_error)
            except Exception:
                logger.warning("Could not record run completion", exc_info=True)

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

    # Refresh the run history so the run that just finished appears there. The
    # panel is otherwise only built at session start, which would leave it
    # claiming "no runs recorded yet" immediately after completing one.
    if ws is not None and ws.registry.enabled:
        try:
            ws.runs = await run_in_executor(ws.registry.list_for_user, ws.prefs_key)
            await _render_workspace_sidebar(ws)
        except Exception:
            logger.warning("Could not refresh run history panel", exc_info=True)


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------

# Cadence (seconds) for the in-UI "running… Ns" timer on an executing code step.
_STEP_TIMER_INTERVAL = 3.0
# Cadence (seconds) for the backend run-liveness heartbeat log event.
_HEARTBEAT_SECONDS = float(os.getenv("BIOMNI_RUN_HEARTBEAT_SECONDS", "15"))


async def _tick_step_timer(step: cl.Step, language: str, code: str, started: float) -> None:
    """Refresh a running code step with an elapsed-time line so it doesn't look frozen.

    A single code execution can block for up to ``timeout_seconds`` with no
    streamed output; without this the step appears hung. Cancelled when the
    observation arrives. Best-effort — any UI error just stops the timer.
    """
    try:
        while True:
            await asyncio.sleep(_STEP_TIMER_INTERVAL)
            elapsed = int(time.monotonic() - started)
            step.output = f"```{language}\n{code}\n```\n\n⏱ running… {elapsed}s"
            try:
                await step.update()
            except Exception:
                logger.debug("step timer update failed", exc_info=True)
                return
    except asyncio.CancelledError:
        pass


async def _stream_execution(
    agent,
    prompt: str,
    history: list[dict] | None = None,
    thread_id: str = "42",
    on_heartbeat=None,
) -> dict | None:
    """Stream the LangGraph ReAct loop and display steps in Chainlit."""
    # Build full message list from conversation history so the agent has context
    messages = []
    for msg in history or []:
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
    code_timers: list[asyncio.Task] = []

    # Reset per-run telemetry counters and arm the wall-clock budget for this query.
    if hasattr(agent, "_begin_run"):
        agent._begin_run()

    # Heartbeat thread: emit run-liveness events every few seconds so a slow-but-
    # healthy run is distinguishable from a wedged one in the log pipeline; the
    # per-step UI timer below gives the user the same signal.
    with RunHeartbeat(
        interval=_HEARTBEAT_SECONDS,
        logger=logger,
        status=lambda: {"step": getattr(agent, "_react_step", None)},
        # Advance the durable run record on the same cadence, so a later session
        # can tell a slow-but-alive run from one whose process died.
        on_tick=on_heartbeat,
    ):
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
                # Tick an elapsed-time line while the (blocking) code runs so the
                # step doesn't look frozen; cancelled when the observation lands.
                code_timers.append(asyncio.create_task(_tick_step_timer(code_step, language, code, time.monotonic())))
                # Do NOT exit the step yet; we close it when the observation arrives

            # 4. Observation (result of code execution)
            obs_match = re.search(r"<observation>(.*?)</observation>", content, re.DOTALL)
            if obs_match:
                observation = obs_match.group(1).strip()

                # Close the pending code step now that we have a result
                if code_timers:
                    await _cancel_task(code_timers.pop())
                if code_steps:
                    finished_step = code_steps.pop()
                    await finished_step.__aexit__(None, None, None)

                async with cl.Step(name="👁 Observation", type="tool", show_input=False) as obs_step:
                    # Truncate very long output for display
                    display_obs = observation[:3000] + "\n...[truncated]" if len(observation) > 3000 else observation
                    obs_step.output = display_obs

                    # Display any generated images mentioned in the observation
                    await _display_images(observation)

    # Close any code steps/timers that never received an observation (edge case)
    for timer in code_timers:
        await _cancel_task(timer)
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


async def _cancel_task(task: asyncio.Task) -> None:
    """Cancel a task and await its completion, swallowing the cancellation."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


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
                    logger.warning("Failed to render image %s", candidate, exc_info=True)
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
    """Save run artifacts to the session's output directory and notify the user."""
    # Reuse the pre-created run directory if the agent already has one set
    # (created before streaming so OUTPUT_DIR was available during execution).
    if hasattr(agent, "_current_run_dir") and agent._current_run_dir:
        current_run_dir = agent._current_run_dir
        run_id = os.path.basename(current_run_dir)
    else:
        ws = cl.user_session.get("workspace")
        run_id = build_run_id(topic)
        runs_root = ws.output.path if ws is not None else os.path.abspath(os.path.join(os.getcwd(), "runs"))
        os.makedirs(runs_root, exist_ok=True)
        current_run_dir = os.path.join(runs_root, run_id)
        os.makedirs(current_run_dir, exist_ok=True)

    # Sync internal state so artifact methods work correctly
    if "messages" in final_state:
        agent.raw_log = list(final_state["messages"])
    agent._conversation_state = final_state

    async with cl.Step(name="📦 Saving Artifacts", type="tool", show_input=False) as step:
        try:
            await run_in_executor(agent._save_run_artifacts, run_id, current_run_dir, initial_files)
            step.output = f"Artifacts saved to `{current_run_dir}`"
        except Exception as exc:
            logger.exception("Artifact saving failed for run %s", run_id)
            step.output = f"⚠️ Artifact saving failed: {exc}"

    await cl.Message(content=f"**Run complete.** Artifacts saved to:\n`{current_run_dir}`").send()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

# Run-id / file-snapshot helpers now live in biomni.artifact so that the agent
# and the UI use the same exclude list — see the imports at the top of the file.
