"""Triage and the plan-then-approve workflow for the Chainlit UI.

Two stages. First `quick_answer` decides whether the message needs the research
pipeline at all: "which GWAS files do I have?" is a question, not a project, and
making someone approve a numbered plan to be told is a waste of their time.
Anything that would require reading data, running code or searching falls
through to the second stage.

There, the agent proposes a numbered research plan; the user approves,
revises, or cancels before any code executes.

`build_planning_system_prompt` is a pure function and is unit-tested
independently. `interactive_planning` itself uses Chainlit's async UI
primitives (`async with cl.Step(...)`, `cl.AskActionMessage`) and is
exercised only through a running Chainlit session - its branching is
not currently covered by automated tests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING

import chainlit as cl
from biomni.observability import emit_event
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from biomni.agent.a1 import A1

logger = logging.getLogger(__name__)


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
    "DATA SOURCING - LOCAL FIRST: draw on the datasets listed below and in the AD data lake for as much "
    "of the analysis as possible. Do NOT download, fetch, or call external APIs when the data is already "
    "present locally. Do NOT simulate or fabricate data. "
    "Prefer built-in domain tools (database queries, tool functions) over custom code, and write custom "
    "code only to carry out analyses no tool covers. "
    # Every plan used to open with a step that walked the workspace and the data
    # lake and printed their absolute paths. It is plumbing, not research: the
    # user is being asked to approve a scientific approach, and a file-system
    # crawl tells them nothing about whether that approach is right. The
    # concrete files appear in the closing section instead, where they are the
    # point rather than a preamble.
    "PLAN CONTENT: every numbered step must be an analysis step that advances the science, named with the "
    "specific dataset, tool or method it uses. NEVER include a step whose purpose is to scan, list, "
    "enumerate, inventory or 'discover' files or directories - finding and loading a file is part of the "
    "step that uses it, not a step of its own. In the numbered steps refer to data by dataset or study "
    "name, not by filesystem path. "
    "Be specific and tailor the plan to user's question. Do not execute any code yet."
)


# --------------------------------------------------------------------------- #
# Triage: does this question need a plan at all?
# --------------------------------------------------------------------------- #

# Sentinel the triage call returns instead of an answer. Deliberately ugly and
# unlikely to appear in prose, since a false positive here silently swallows a
# real analysis request.
NEEDS_PLAN_SENTINEL = "NEEDS_PLAN"

TRIAGE_SYSTEM_PROMPT = (
    "You are the front desk of an Alzheimer's disease research agent. Decide whether the user's "
    "message needs the full research pipeline, or whether you can simply answer it.\n\n"
    f"Reply with exactly `{NEEDS_PLAN_SENTINEL}` and nothing else if answering would require ANY of: "
    "reading or parsing a data file, running code, computing or counting anything from data, querying a "
    "database, searching the literature or the web, or producing a figure or table from data.\n\n"
    "Otherwise answer the user directly, in a few sentences. That covers questions about what the data "
    "is (names, locations, formats, what a dataset contains), definitions and background knowledge, "
    "questions about what you can do, and follow-ups that are already answered by the conversation "
    "above.\n\n"
    "Never guess at a number, a result, or a file's contents in a direct answer - if you would have to "
    f"look, reply `{NEEDS_PLAN_SENTINEL}`. When you are unsure, reply `{NEEDS_PLAN_SENTINEL}`: an "
    "unnecessary plan costs the user one click, a fabricated answer costs them their trust."
)

# The workspace listing is injected so "which files do I have?" is answerable
# without a plan. Bounded because it can be hundreds of lines and this is a
# latency-sensitive call on every single message.
_TRIAGE_INVENTORY_CHARS = 4000


def build_triage_system_prompt(agent: A1) -> str:
    """Compose the triage prompt, with a bounded view of the workspace."""
    prompt = TRIAGE_SYSTEM_PROMPT
    inventory = getattr(agent, "user_data_inventory", None)
    if inventory:
        excerpt = inventory[:_TRIAGE_INVENTORY_CHARS]
        if len(inventory) > _TRIAGE_INVENTORY_CHARS:
            excerpt += "\n... (listing truncated)"
        prompt += (
            "\n\nThe user's selected data is listed below. You may answer questions about what is here "
            f"directly; anything about what is *inside* these files needs `{NEEDS_PLAN_SENTINEL}`.\n{excerpt}"
        )
    return prompt


def parse_triage_response(text: str) -> str | None:
    """The direct answer, or ``None`` when the question needs a plan.

    Treats a reply that merely *contains* the sentinel as a request for a plan.
    Models like to wrap a bare token in prose ("I think this is NEEDS_PLAN"),
    and reading that as an answer would show the user the sentinel and skip the
    analysis they asked for - the one failure here that is not self-correcting.
    """
    if not text:
        return None
    answer = text.strip()
    if not answer or NEEDS_PLAN_SENTINEL in answer:
        return None
    return answer


def direct_answers_enabled() -> bool:
    """Whether short questions may bypass the plan gate (default on).

    A deployment that needs every action to pass through an explicit approval -
    for audit, or because it does not trust the triage call - sets
    ``BIOMNI_ALWAYS_PLAN``.
    """
    return os.getenv("BIOMNI_ALWAYS_PLAN", "").strip().lower() not in {"1", "true", "yes", "on"}


async def quick_answer(agent: A1, prompt: str, history: list[dict] | None = None) -> str | None:
    """Answer a question that needs no analysis, or ``None`` to go on and plan.

    One extra LLM call on every message, which is the price of not making
    someone approve a five-step research plan to be told which file they are
    looking at. Any failure returns ``None``: the plan path is the safe default,
    so triage must never be able to break a request.
    """
    if not direct_answers_enabled():
        return None

    messages = [SystemMessage(content=build_triage_system_prompt(agent))]
    for turn in history or []:
        content = turn.get("content") or ""
        if not content:
            continue
        messages.append(
            AIMessage(content=content) if turn.get("role") == "assistant" else HumanMessage(content=content)
        )
    messages.append(HumanMessage(content=prompt))

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.llm.invoke, messages)
    except Exception:
        logger.warning("Triage call failed; falling back to the planning path", exc_info=True)
        return None

    text = response.content if hasattr(response, "content") else str(response)
    if not isinstance(text, str):
        return None
    answer = parse_triage_response(text)
    emit_event("query_triage", direct=answer is not None)
    return answer


# The plan-gate buttons. Named here rather than inline because the app also
# registers fallback handlers for these exact names - see STALE_PLAN_ACTION_NOTE.
PLAN_ACTIONS: tuple[tuple[str, str], ...] = (
    ("approve", "✅ Approve & Execute"),
    ("revise", "✏️ Revise Plan"),
    ("cancel", "🚫 Cancel"),
)

# Shown when a plan button is clicked but nothing is waiting for it any more.
#
# While a plan is pending, the browser resolves a click through the ask it was
# issued with; that path never consults the server's action-callback registry.
# The registry is only reached once the ask is gone - after a server restart or
# rollout, or after the 300s timeout - and with nothing registered Chainlit
# answers "Not Found: No callback found for action approve", which reads like a
# broken app rather than an expired question.
STALE_PLAN_ACTION_NOTE = (
    "That plan is no longer waiting for an answer - the server restarted, or the question timed out. "
    "**Nothing was run.** Send your question again to get a fresh plan."
)


# Heading the planner is told to emit, and that `extract_planned_data_files`
# parses back out. Kept as one constant so the two can never drift.
DATA_FILES_HEADING = "Data files this plan will read"

# Leading junk is anything that is not a letter: "**", "3.", "### ", "- ". Using
# \W* would not match a numbered heading, because digits are word characters.
_DATA_FILES_HEADING_RE = re.compile(rf"^[^A-Za-z]*{re.escape(DATA_FILES_HEADING)}\W*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")


# Wording that marks an entry as aspirational rather than a file the plan has
# actually committed to reading.
_PLACEHOLDER_MARKERS = (
    "to be discovered",
    "to be determined",
    "if available",
    "if present",
    "if found",
    "if it exists",
    "tbd",
    "unknown",
    "look for",
    "check ",
)


def _is_concrete_file(entry: str) -> bool:
    """Whether a listed entry names a real file rather than a place to look.

    The planner is instructed to list only concrete files, but instruction
    compliance is not a guarantee, and a list of directories annotated with
    "(files to be discovered)" is worse than no list: it reads as a commitment
    the plan has not made.
    """
    lowered = entry.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    if entry.endswith(("/", os.sep)):
        return False
    if "*" in entry or "?" in entry:  # a glob is not a file
        return False
    # A trailing parenthetical is where models put their hedging.
    if entry.endswith(")") and "(" in entry:
        return False
    # Require something that looks like a filename: a dotted suffix on the last
    # path segment. Extension-less data files exist, but accepting them here
    # would also accept every bare directory path, which is the failure mode.
    return "." in os.path.basename(entry.rstrip())


def extract_planned_data_files(plan_text: str) -> list[str]:
    """Pull the declared input files out of a generated plan.

    Returns an empty list when the planner declared none, omitted the section,
    or wrote the literal "none" - all of which mean "no files to confirm", so
    the caller shows nothing rather than an empty box.

    Tolerant of the formatting the model actually produces: the heading may be
    bolded or numbered, and entries may use any bullet character. Paths wrapped
    in backticks are unwrapped.
    """
    if not plan_text:
        return []

    lines = plan_text.splitlines()
    start = next((i for i, line in enumerate(lines) if _DATA_FILES_HEADING_RE.match(line.strip())), None)
    if start is None:
        return []

    files: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            # Blank lines inside the block are fine; stop only once something
            # has been collected, so a blank line right after the heading does
            # not truncate the section.
            if files:
                break
            continue
        match = _BULLET_RE.match(stripped)
        if match is None:
            break
        entry = match.group(1).strip().strip("`").strip()
        if entry.lower() in {"none", "n/a", "(none)"}:
            return []
        if entry and _is_concrete_file(entry):
            files.append(entry)

    # De-duplicate while preserving the planner's order.
    seen: set[str] = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def _format_data_lake_listing(agent: A1, names: list[str]) -> str:
    """Render the resource-retrieval step's selected data-lake items as a concrete listing.

    By the time planning runs, ``_prepare_resources_for_retrieval`` has already fetched
    every one of these from the catalog if it wasn't local (see ``_ensure_data_lake_files``
    in a1.py) - so the planner can name them outright rather than hedge with "if available".
    """
    data_lake_dict = getattr(agent, "data_lake_dict", {}) or {}
    lines = [f"- {name}: {data_lake_dict.get(name, 'local data lake file')}" for name in names]
    return "\n".join(lines)


def build_planning_system_prompt(agent: A1, agent_type: str, selected_data_lake: list[str] | None = None) -> str:
    """Compose the planning system prompt for `agent_type`, appending any
    locally-discovered user data inventory the agent has already snapshotted,
    plus the data-lake items the resource-retrieval step just matched to this query.

    Pure function - no I/O, no Chainlit calls - so it's directly testable.
    """
    base = AD1_PLANNING_SYSTEM_PROMPT if agent_type == "ad1" else PLANNING_SYSTEM_PROMPT

    inventory = getattr(agent, "user_data_inventory", None)
    if inventory:
        data_root = getattr(agent, "data_root_dir", None) or ""
        base += (
            f"\n\nThe following datasets are available in the local user data directory "
            f"({data_root}). Reference specific datasets from this listing when relevant "
            f"to the user's question:\n{inventory}"
        )

    # The retrieval step ("Selecting Resources") already matched data-lake items to this
    # query and fetched any that weren't local yet - but ran as a separate LLM call that
    # never told the planner what it found. Without this, the planner falls back on its
    # own training-data familiarity with dataset naming conventions: it confidently names
    # well-known files (GWAS summary stats, eQTL tables) and hedges on ones it's less sure
    # of ("if found", "if available") even when those are present in this app's catalog
    # right now. Grounding the plan in the actual retrieval result fixes both problems.
    if selected_data_lake:
        base += (
            "\n\nThe resource-retrieval step already matched the following data lake items to "
            "this question, and any that were not already local have been fetched. Treat these "
            "as available now - name the specific ones you will use in your plan rather than "
            "guessing at file names or hedging with 'if available' / 'if found':\n"
            f"{_format_data_lake_listing(agent, selected_data_lake)}"
        )

    # Reviewers asked that a plan commit to the data it will read, so the user
    # can correct the file choice before any code runs rather than discovering
    # the wrong input in the results.
    #
    # The section must contain real files or nothing. An earlier version invited
    # the model to list what it intended to look at, and it filled the section
    # with directories and "(files to be discovered in Step 1)" - which tells
    # the user nothing they did not already know and makes the plan look like it
    # has committed to data when it has not.
    base += (
        f"\n\n{DATA_FILES_HEADING} requirement: if - and only if - the listing above named concrete "
        f"data files that this plan will read, end your reply with a section titled exactly "
        f"'{DATA_FILES_HEADING}' listing them one per line as `- <path>`, with each name copied "
        "exactly from that listing and written relative to the folder it was listed under "
        "(`studyA/results.tsv`, never the full absolute path - the user knows where their own "
        "workspace is, and five absolute paths are a wall of text). "
        "Every entry must be a real file. Never list a directory, a glob, or a placeholder "
        "such as 'to be discovered' or 'if available'. If you have not been shown concrete files, omit "
        "the section entirely - do not replace it with a directory listing or a file-discovery step."
    )
    return base


async def interactive_planning(
    agent: A1, prompt: str, agent_type: str = "a1", selected_data_lake: list[str] | None = None
) -> str | None:
    """Generate a plan, prompt the user to approve / revise / cancel, loop on revise.

    `selected_data_lake` is the data-lake item names the resource-retrieval step ("Selecting
    Resources") already matched to this query, so the plan can name them concretely instead
    of guessing at file names from the model's own training data.

    Returns the (possibly modified) prompt on approval, or `None` if the user
    cancels. If plan generation itself raises, the approval gate is skipped
    and the original prompt is returned so execution still happens.
    """
    base_prompt = build_planning_system_prompt(agent, agent_type, selected_data_lake)
    modification_context = ""

    while True:
        full_system = base_prompt
        if modification_context:
            full_system += f"\n\nUser requested these revisions to the previous plan:\n{modification_context}"

        planning_messages = [
            SystemMessage(content=full_system),
            HumanMessage(content=prompt),
        ]

        # default_open: the plan is the thing the user is being asked to approve,
        # so it must be readable without first expanding a collapsed step.
        async with cl.Step(name="📋 Generating Research Plan", type="llm", show_input=False, default_open=True) as step:
            try:
                # NB: asyncio.to_thread would also work but copies the caller's
                # contextvars into the worker - the rest of chainlit_app.py
                # uses bare run_in_executor and we match that semantics so
                # LangChain callback/tracing contextvars don't silently change
                # which trace the LLM call attaches to.
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, agent.llm.invoke, planning_messages)
                plan_text = response.content if hasattr(response, "content") else str(response)
                step.output = plan_text
            except Exception as exc:
                logger.warning("Plan generation failed; proceeding without approval gate", exc_info=True)
                step.output = f"⚠️ Could not generate plan ({exc}). Proceeding without a plan."
                return prompt

        # The plan already carries its own data-files section and is shown
        # expanded, so repeating the list here only duplicated it on screen.
        # Logged rather than rendered: still useful for telemetry, invisible in
        # the transcript.
        planned_files = extract_planned_data_files(plan_text)
        emit_event("plan_data_files", count=len(planned_files))

        question = "Here is the research plan. Would you like to proceed?"
        if planned_files:
            question = (
                "If the data files listed in the plan are not the right ones, choose "
                "**Revise Plan** and say which to use.\n\n" + question
            )

        res = await cl.AskActionMessage(
            content=question,
            actions=[cl.Action(name=name, label=label, payload={"value": name}) for name, label in PLAN_ACTIONS],
            timeout=300,
        ).send()

        action_value = (res.get("payload") or {}).get("value") if res else None

        if res is None or action_value == "cancel":
            return None

        if action_value == "approve":
            return prompt

        mod_res = await cl.AskUserMessage(
            content="Describe the changes you'd like in the plan:",
            timeout=300,
        ).send()

        if mod_res:
            modification_context = mod_res.get("output", "")
