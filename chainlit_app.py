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
import inspect
import logging
import os
import re
import sys
import threading
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
from biomni.health import register_health_routes
from biomni.identity import UserIdentity, resolve_identity, single_user_mode, trust_auth_headers
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
from chainlit_ui.live_runs import LIVE_RUNS
from chainlit_ui.persistence import build_data_layer, ensure_auth_secret
from chainlit_ui.planning import PLAN_ACTIONS, STALE_PLAN_ACTION_NOTE
from chainlit_ui.planning import (
    interactive_planning as _interactive_planning,
)
from chainlit_ui.planning import (
    quick_answer as _quick_answer,
)
from chainlit_ui.uploads import store_uploads, uploads_dir
from chainlit_ui.workspace_panel import (
    build_scope_inventory,
    list_top_level_dirs,
    scope_choice_items,
    summarize_scope,
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


def _resolve_builtin_data_lake_root() -> str:
    """Return the default repo-local data lake directory path."""
    repo_root = Path(__file__).resolve().parent
    return str((repo_root / "data" / "biomni_data" / "data_lake").resolve())


def _refresh_chainlit_welcome_markdown() -> None:
    """Rebuild chainlit.md from its template plus the local-data examples.

    ``chainlit.md`` is generated and gitignored; ``chainlit.md.template`` is the
    tracked source, and the place to edit this page.

    Written whole rather than patched. The previous version preserved whatever
    was already in chainlit.md and spliced managed sections into it between HTML
    comment markers - but Chainlit's Readme renderer prints those markers as
    literal text ("<!-- BIOMNI_SUGGESTED_PROMPTS_START -->" in the middle of the
    page), so the mechanism that made the file editable also defaced it.

    The page also used to end with an inventory of the data lake and the
    workspace - 200 file names in a collapsed block. Gone: this is a page about
    how to use the app, and a file listing is neither instruction nor something
    the reader can act on from here.
    """
    try:
        if CHAINLIT_MD_TEMPLATE_PATH.exists():
            base = CHAINLIT_MD_TEMPLATE_PATH.read_text(encoding="utf-8")
        else:
            base = (
                "## How to use Biomni-AD\n\n"
                "Ask a quick question for a straight answer, or ask for an analysis and "
                "approve the plan before anything runs.\n"
            )

        # Tolerate a template still carrying the retired marker blocks.
        for start, end in [
            (_SUGGESTED_PROMPTS_BLOCK_START, _SUGGESTED_PROMPTS_BLOCK_END),
            (_WELCOME_DATASET_BLOCK_START, _WELCOME_DATASET_BLOCK_END),
        ]:
            base = re.sub(rf"\n?{re.escape(start)}.*?{re.escape(end)}\n?", "\n", base, flags=re.DOTALL)

        suggested = _build_ad_suggested_prompts()
        page = base.rstrip() + (f"\n\n{suggested}\n" if suggested else "\n")
        CHAINLIT_MD_PATH.write_text(page, encoding="utf-8")
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


def _output_dir_description(ws: WorkspaceSession) -> str:
    """Help text for the output-directory field, including its warnings.

    The two warnings used to live in the sidebar panel, and they are the only
    part of it that was load-bearing: a user whose results are going somewhere
    unwritable, or somewhere a restart will erase, has to be told before they
    start a long run rather than after.
    """
    text = (
        f"Where run results are written. Currently resolved to {ws.output.path} "
        f"(source: {ws.output.source}). Leave as-is to keep the deployment default."
    )
    if not ws.output.writable:
        text += f"\n\n⚠️ That directory is not writable. {ws.output.reason or ''}".rstrip()
    elif ws.output.is_ephemeral:
        text += (
            "\n\n⚠️ Container-local storage: results are lost when the application restarts. "
            "Set a path on a mounted volume, or ask an operator to configure BIOMNI_OUTPUT_ROOT."
        )
    if ws.persistence_label:
        text += f"\n\nSettings are stored in {ws.persistence_label}."
    return text


def _workspace_settings_widgets(ws: WorkspaceSession) -> list[InputWidget]:
    """Chat-settings controls for scope and output directory.

    This dialog is also where the workspace *state* is reported, now that there
    is no right-hand panel: the descriptions carry the active scope with its
    file counts, anything selected that has since disappeared, and where outputs
    resolve to. It is the one place a user opens when they want to know or
    change any of that.

    The folder multi-select is omitted entirely when the workspace has no
    subdirectories - Chainlit's MultiSelect rejects an empty item list, and an
    empty picker would be noise anyway.
    """
    widgets: list[InputWidget] = []
    known = set(ws.top_level_dirs)

    if ws.top_level_dirs:
        description = (
            "Only the selected folders are scanned and described to the agent. "
            "Selecting large folders slows every session and dilutes the agent's attention, "
            "so prefer the specific studies you are working on. "
            "With nothing selected, the agent is told which folders exist and looks inside "
            "them only when a task calls for it."
        )
        summaries = summarize_scope(ws.scope, ws.workspace_root)
        if summaries:
            active = ", ".join(f"{s.label} ({s.files_label()})" for s in summaries[:8])
            if len(summaries) > 8:
                active += f" and {len(summaries) - 8} more"
            description += f"\n\nCurrently active: {active}."
        if ws.scope.missing:
            description += (
                "\n\n⚠️ Selected but no longer present (deleted or renamed): " + ", ".join(ws.scope.missing) + "."
            )
        widgets.append(
            MultiSelect(
                id="scope_folders",
                label="Data folders the agent may use",
                items=scope_choice_items(ws.top_level_dirs),
                initial=[p for p in ws.prefs.scope_paths if p in known],
                description=description,
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
            description=_output_dir_description(ws),
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


# There is deliberately no right-hand panel.
#
# Chainlit's element sidebar cannot be filled without being opened -
# `ElementSidebar.set_elements` opens it on any non-empty list, and `set_title`
# says so in its own docstring - so "show the workspace state on the right"
# and "do not take a third of the screen on every session start" cannot both be
# true. Between the two, the screen wins: reviewers asked three times for the
# right side to stop appearing.
#
# Everything the panel carried now lives where the user is already looking when
# they care about it:
#
#   * active scope and resolved output directory -> the ⚙️ Settings dialog,
#     which is where they are changed anyway, plus the confirmation message
#     posted to the chat on save,
#   * where a finished run wrote its results -> the run's own completion
#     message,
#   * work that did not finish while the user was away -> a chat notice at
#     session start, and only when there is some,
#   * past conversations -> the thread list Chainlit renders on the left.


# ---------------------------------------------------------------------------
# Authentication and durable chat threads
# ---------------------------------------------------------------------------
#
# Chainlit shows the conversation list on the left, and lets a user reopen an
# old conversation, only when it has BOTH an authenticated user and a data
# layer. Registering `header_auth_callback` is what turns the first one on; it
# also makes Chainlit's notion of who is calling agree with the one the
# preferences and run registry already use, so all three key off the same
# string.
#
# Chainlit refuses to start with an auth callback and no CHAINLIT_AUTH_SECRET,
# so the secret is resolved (and persisted) at import, before uvicorn boots.
os.environ.setdefault("CHAINLIT_AUTH_SECRET", ensure_auth_secret(_resolve_workspace_root()))


@cl.header_auth_callback
def header_auth_callback(headers) -> cl.User | None:
    """Identify the caller from the authentication gateway's headers.

    Three outcomes, and the middle one is the reason this is not just a
    passthrough:

    * Gateway trusted and it named someone -> that person.
    * Gateway trusted and it named nobody -> refuse. A deployment that declared
      a gateway is in front and then received a request without identity headers
      has a routing hole, and serving it anyway would hand the un-authenticated
      caller whichever account the fallback picks.
    * No gateway configured -> a single shared local user, matching how the rest
      of the app already behaves when `BIOMNI_TRUST_AUTH_HEADERS` is off. This
      grants no access that was not already open: with no gateway the app has no
      front door at all. What it does mean is that everyone reaching it shares
      one history, which is why persistence stays off unless a deployment opts
      in with `BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE`.
    """
    identity = resolve_identity(headers)
    if not identity.is_authenticated and trust_auth_headers():
        logger.warning("rejecting a session with no gateway identity headers (BIOMNI_TRUST_AUTH_HEADERS is on)")
        return None
    return cl.User(
        identifier=identity.scoped_key(),
        display_name=identity.display_name,
        metadata={
            "source": identity.source,
            "email": identity.email or "",
            "workspace_id": identity.workspace_id or "",
        },
    )


@cl.data_layer
def data_layer():
    """Durable storage for conversations, or ``None`` to keep them in memory.

    Enabled only where the storage key is stable enough for a user to find their
    own threads again: behind a gateway, or in the explicitly opted-in
    single-user mode. Under a per-connection anonymous key every page load would
    start a fresh identity, so the sidebar would fill with orphaned threads that
    nobody could ever reopen.
    """
    if not (trust_auth_headers() or single_user_mode()):
        logger.info(
            "chat history is disabled: no authentication gateway is configured. "
            "Set BIOMNI_TRUST_AUTH_HEADERS (behind a gateway) or "
            "BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE (single-user deployments) to enable it."
        )
        return None
    return build_data_layer(_resolve_workspace_root())


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
    """Yield LangGraph state dicts asynchronously from a sync stream.

    The graph runs in a worker thread, which cancellation cannot reach: an
    asyncio task that is cancelled stops *reading*, and a thread that nobody
    reads goes right on calling the model. Stop therefore used to end the
    conversation while the run it stopped kept working - for as long as the
    whole plan took. The flag below is the cooperative half of the stop: the
    producer checks it between graph nodes, so the run ends after the step it
    was already in.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    stop = threading.Event()

    def _producer():
        try:
            for state in agent_app.stream(inputs, stream_mode="values", config=config):
                asyncio.run_coroutine_threadsafe(queue.put(state), loop)
                if stop.is_set():
                    break
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    executor = ThreadPoolExecutor(max_workers=1)
    # Capture the active context here (caller thread) and replay it in the
    # producer thread so the agent graph's logs (LLM calls, code-execution
    # audit) carry the session_id/run_id.
    ctx = capture_context()
    executor.submit(ctx.run, _producer)

    try:
        while True:
            state = await queue.get()
            if state is None:
                break
            yield state
    finally:
        stop.set()
        # Not waited on: the worker may be mid-node, and holding the event loop
        # for it would freeze the UI. It exits at the next check.
        executor.shutdown(wait=False)


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
    await _start_session()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Reopen a stored conversation and make it live again.

    Chainlit replays the persisted messages into the UI by itself; what it
    cannot do is rebuild the *server* side, so without this handler an old
    thread is read-only. Everything a session needs - the agent, the workspace
    scope, the settings panel - is constructed here exactly as on a fresh start,
    plus the conversation history so the agent knows what was already said.

    If the run that filled this conversation is still going - the user closed
    the tab and came back - it is redirected into this connection, so the rest
    of it streams in here instead of into a socket nobody is holding.
    """
    history = _history_from_thread(thread)
    reattached = await _follow_live_run(thread.get("id"))
    emit_event(
        "chat_resume",
        steps=len(thread.get("steps") or []),
        history_turns=len(history),
        reattached=reattached,
    )
    await _start_session(resumed_history=history)


async def _follow_live_run(thread_id: str | None) -> bool:
    """Point a still-running conversation at this connection. True if one was.

    Redirecting the emitter is enough for the work itself: the rest of the run
    arrives as ordinary streamed steps. Saying so is left to
    :func:`_announce_live_run`, which cannot run yet - see there.
    """
    try:
        if not LIVE_RUNS.reattach(thread_id, _websocket_session()):
            return False
        asyncio.create_task(_announce_live_run())
        return True
    except Exception:
        logger.warning("Could not follow the live run for thread %s", thread_id, exc_info=True)
        return False


# How long to let the browser finish restoring a conversation before telling it
# the conversation is still busy.
_RESUME_ANNOUNCE_DELAY_S = 1.0


async def _announce_live_run() -> None:
    """Put a reopened conversation back into its running state.

    Deferred, and that is the whole subtlety: Chainlit sends the stored
    transcript to the browser *after* the resume handler returns, and the page
    settles into an idle state as it renders it. Anything said from inside the
    handler is undone a moment later, which is exactly how this looked - live
    steps streaming in under an input box that believed nothing was happening,
    so Stop was not offered.

    ``task_start`` restores the running state and with it the Stop button, which
    :func:`on_stop` makes good on. Its ``task_end`` comes from the run's own
    handler when it finishes, through the emitter that was just redirected.
    """
    try:
        await asyncio.sleep(_RESUME_ANNOUNCE_DELAY_S)
        await cl.context.emitter.task_start()
        # send_toast is a coroutine on a websocket connection and a no-op stub
        # off one; only the first is awaitable.
        toast: object = cl.context.emitter.send_toast("This conversation is still running. Output will continue below.")
        if inspect.isawaitable(toast):
            await toast
    except Exception:
        logger.warning("Could not announce the live run", exc_info=True)


def _history_from_thread(thread: dict) -> list[dict]:
    """Reconstruct the agent-facing conversation from a persisted thread.

    Only the user's questions and the assistant's answers: the intermediate
    tool/code steps are in the transcript for the human to read, but replaying
    them into the model's context would cost a fortune and teach it nothing it
    cannot see in the answer.
    """
    role_by_type = {"user_message": "user", "assistant_message": "assistant"}
    history: list[dict] = []
    for step in thread.get("steps") or []:
        role = role_by_type.get(step.get("type") or "")
        content = step.get("output") or ""
        if role and content:
            history.append({"role": role, "content": content})
    return history


async def _start_session(resumed_history: list[dict] | None = None) -> None:
    """Build everything one chat session needs. Shared by start and resume."""
    # One correlation id per chat session, bound for the lifetime of this
    # handler so agent-init logs carry it. on_message rebinds it per message.
    # Chainlit's own thread id is preferred when threads are being persisted, so
    # a log line can be traced back to the conversation the user can reopen.
    thread_id = _chainlit_thread_id() or uuid.uuid4().hex
    set_session_id(thread_id)

    # Determine agent type: CLI env var > chat profile selection > default AD1
    if FORCE_AGENT in ("a1", "ad1"):
        agent_type = FORCE_AGENT
    else:
        profile = cl.user_session.get("chat_profile", "AD1")
        agent_type = "a1" if str(profile).upper() == "A1" else "ad1"

    label = "AD1" if agent_type == "ad1" else "A1"
    emit_event("chat_start", agent_type=agent_type, llm=DEFAULT_LLM, resumed=resumed_history is not None)
    try:
        if agent_type == "ad1":
            from biomni.agent.ad1 import AD1

            agent = await run_in_executor(lambda: AD1(llm=DEFAULT_LLM))
        else:
            from biomni.agent.a1 import A1

            agent = await run_in_executor(lambda: A1(llm=DEFAULT_LLM))
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_type", agent_type)
        cl.user_session.set("history", resumed_history or [])
        cl.user_session.set("thread_id", thread_id)
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
            # Earlier runs are reported here and nowhere else. Every run has a
            # conversation in the list on the left holding everything it
            # produced - including work that carried on after the browser was
            # closed - so opening a new chat with a summary of a different one
            # interrupted the user with what a click already shows them. The
            # count stays as an operational signal.
            unfinished_runs=len(ws.registry.unfinished(ws.runs)),
        )
    except Exception:
        logger.exception("Failed to resolve workspace session (continuing with defaults)")

    # Expose the scope / output controls.
    if ws is not None:
        try:
            await cl.ChatSettings(_workspace_settings_widgets(ws)).send()
        except Exception:
            logger.exception("Failed to render workspace settings (session remains usable)")


def _chainlit_thread_id() -> str | None:
    """Chainlit's id for the conversation, when there is a session to ask."""
    try:
        return getattr(cl.context.session, "thread_id", None)
    except Exception:
        return None


def _thread_key() -> str:
    """The conversation id used for run bookkeeping, from either lifecycle path.

    Always set by _start_session; the fallback only matters if a message
    somehow arrives before one ran, and it keeps register/release agreeing on a
    key either way.
    """
    return cl.user_session.get("thread_id") or "unknown"


def _websocket_session():
    """The connection this handler is running on, or None if there is none.

    The object itself is what a detached run needs: redirecting its ``emit`` is
    how a reopened tab starts receiving output again (see chainlit_ui.live_runs).
    """
    try:
        return cl.context.session
    except Exception:
        return None


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


async def _on_stale_plan_action(action: cl.Action) -> None:
    """Answer a click on a plan button that nothing is waiting for.

    Reached only once the ask behind those buttons is gone - a restart, a
    rollout, or the 300s timeout - because a live ask is resolved by the browser
    without consulting this registry. Without a handler Chainlit returns a bare
    "Not Found: No callback found for action approve", which looks like the app
    is broken at the exact moment the user is trying to authorise work.
    """
    logger.info("stale plan action clicked: %s", getattr(action, "name", "?"))
    await cl.Message(content=STALE_PLAN_ACTION_NOTE).send()


for _plan_action_name, _ in PLAN_ACTIONS:
    cl.action_callback(_plan_action_name)(_on_stale_plan_action)


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

    emit_event(
        "workspace_settings_updated",
        scope_folders=len(ws.scope.roots),
        scope_is_default=ws.scope.is_default,
        output_source=ws.output.source,
        persisted=persisted,
    )

    if ws.scope.roots:
        # With no panel to consult, this message is the only place the user is
        # told what their selection actually costs, so it carries the file
        # counts rather than just the folder names.
        summaries = await run_in_executor(summarize_scope, ws.scope, ws.workspace_root)
        shown = summaries[:8]
        scope_text = ", ".join(f"`{s.label}` ({s.files_label()})" for s in shown)
        if len(summaries) > len(shown):
            scope_text += f" and {len(summaries) - len(shown)} more"
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


async def _persist_message_uploads(message: cl.Message) -> list[str]:
    """Copy a message's attachments into the user's output directory.

    Returns the path to hand the agent for each one - the durable copy where
    that could be made, Chainlit's temporary one otherwise.

    Chainlit keeps uploads in a scratch tree it deletes when the session ends
    (and wipes entirely on shutdown), under a UUID filename. Left alone, an
    attachment is unreachable by the user's next visit and the agent never even
    learns what the file was called. Copying is blocking, so it runs off the
    event loop.
    """
    elements = [e for e in (message.elements or []) if getattr(e, "path", None)]
    if not elements:
        return []

    sources: list[tuple[str, str | None]] = [(str(e.path), getattr(e, "name", None)) for e in elements]
    ws = cl.user_session.get("workspace")
    if ws is None or not ws.output.writable:
        return [path for path, _ in sources]

    try:
        destination = uploads_dir(ws.output.path)
        stored = await run_in_executor(store_uploads, sources, destination)
    except Exception:
        logger.warning("Could not store uploads; using the temporary copies", exc_info=True)
        return [path for path, _ in sources]

    emit_event("uploads_stored", count=len(stored), durable=sum(1 for p in stored if p.startswith(destination)))
    return stored


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
            # Unconditional: the handler may have been cancelled or raised, and
            # a conversation left registered as live would have the next tab
            # that opens it wait on a run that is not there.
            LIVE_RUNS.release(_thread_key(), _websocket_session())
            _emit_run_telemetry(agent, usage_before, started)


@cl.on_stop
async def on_stop():
    """Make Stop work from a tab that did not start the run.

    Chainlit cancels the clicking session's own task before calling this. For a
    conversation reopened while its run continued in the background that task is
    not the run, so without this the button would report a stop that never
    happened. Cancelling the run makes its handler unwind normally: the record
    closes out and the page is told the task ended.
    """
    if LIVE_RUNS.cancel(_thread_key()):
        emit_event("run_stopped_by_user")


async def _process_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    agent_type = cl.user_session.get("agent_type", "a1")

    if agent is None:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    prompt = message.content
    file_paths = await _persist_message_uploads(message)
    if file_paths:
        prompt += "\n\nUser uploaded these files:\n" + "\n".join(f"- {p}" for p in file_paths)

    history = cl.user_session.get("history", [])

    # ------------------------------------------------------------------
    # Phase 0: Triage - can this be answered without the pipeline?
    # ------------------------------------------------------------------
    # "Which GWAS files do I have?" should not cost a resource-retrieval pass,
    # a generated plan and an approval click. Runs before the phases below
    # precisely so a direct answer skips them: they exist to set up an analysis
    # that is not going to happen. The model makes the call, biased towards
    # planning, and any failure falls through to the plan path.
    answer = await _quick_answer(agent, prompt, history)
    if answer:
        await cl.Message(content=answer).send()
        cl.user_session.set(
            "history",
            [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        )
        return

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
                    # Counts go to the log, not the transcript. This step used to
                    # announce things like "91 datasets", which reads as though
                    # the agent had loaded 91 datasets for the question. It had
                    # not: these are candidate references made available to the
                    # model, and the number is an artefact of how many entries
                    # the retriever considered relevant. The data that actually
                    # matters is the data the plan commits to, which the plan
                    # itself now states.
                    emit_event(
                        "resources_selected",
                        tools=len(resources.get("tools", [])),
                        datasets=len(resources.get("data_lake", [])),
                        libraries=len(resources.get("libraries", [])),
                        know_how=len(resources.get("know_how", [])),
                    )
                    step.output = "Matched the available tools and data references to your question."
                else:
                    step.output = "Using the full tool set."
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
    thread_id = cl.user_session.get("thread_id", "42")

    # From here on the work is detachable: Chainlit keeps the task alive when
    # the websocket drops, and every step is persisted as it happens, so
    # closing the browser costs nothing. Claiming the conversation lets a tab
    # that reopens it pick the output back up live (on_chat_resume). Deliberately
    # after the plan gate: an unanswered approval is not work in progress.
    LIVE_RUNS.register(_thread_key(), _websocket_session(), asyncio.current_task())

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
    """Scan observation text for image file paths and display them.

    One plot is usually named more than once in a single observation - the
    script prints the absolute path it saved to, and then the file turns up
    again in a listing or a summary line - so the same figure was posted twice,
    and twice again in the stored transcript. Matches are resolved and
    de-duplicated within the observation; a genuinely regenerated file in a
    later step is a later observation and still shown.
    """
    pattern = r"(\S+?(?:" + "|".join(re.escape(e) for e in SUPPORTED_IMAGE_EXTENSIONS) + r"))"
    shown: set[str] = set()
    for match in re.findall(pattern, observation, re.IGNORECASE):
        fp = match.strip("\"'")
        # Resolve relative or absolute path
        candidates = [fp, os.path.join(os.getcwd(), fp)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                resolved = os.path.realpath(candidate)
                if resolved in shown:
                    break
                shown.add(resolved)
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
