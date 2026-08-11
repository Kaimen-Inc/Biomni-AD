"""Plan-then-approve workflow for the Chainlit UI.

The agent first proposes a numbered research plan; the user approves,
revises, or cancels before any code executes.

`build_planning_system_prompt` is a pure function and is unit-tested
independently. `interactive_planning` itself uses Chainlit's async UI
primitives (`async with cl.Step(...)`, `cl.AskActionMessage`) and is
exercised only through a running Chainlit session — its branching is
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
from langchain_core.messages import HumanMessage, SystemMessage

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


def build_planning_system_prompt(agent: A1, agent_type: str) -> str:
    """Compose the planning system prompt for `agent_type`, appending any
    locally-discovered user data inventory the agent has already snapshotted.

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
        f"'{DATA_FILES_HEADING}' listing them one per line as `- <path>`, copied exactly from that "
        "listing. Every entry must be a real file. Never list a directory, a glob, or a placeholder "
        "such as 'to be discovered' or 'if available'. If you have not been shown concrete files, "
        "omit the section entirely and instead make discovering the files an explicit first step."
    )
    return base


async def interactive_planning(agent: A1, prompt: str, agent_type: str = "a1") -> str | None:
    """Generate a plan, prompt the user to approve / revise / cancel, loop on revise.

    Returns the (possibly modified) prompt on approval, or `None` if the user
    cancels. If plan generation itself raises, the approval gate is skipped
    and the original prompt is returned so execution still happens.
    """
    base_prompt = build_planning_system_prompt(agent, agent_type)
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
                # contextvars into the worker — the rest of chainlit_app.py
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

        mod_res = await cl.AskUserMessage(
            content="Describe the changes you'd like in the plan:",
            timeout=300,
        ).send()

        if mod_res:
            modification_context = mod_res.get("output", "")
