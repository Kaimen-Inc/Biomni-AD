"""Plan-then-approve workflow for the Chainlit UI.

The agent first proposes a numbered research plan; the user approves,
revises, or cancels before any code executes. Lives in its own module
so the prompts and the interaction loop can be unit-tested independently
of the rest of `chainlit_app.py`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import chainlit as cl
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


def build_planning_system_prompt(agent: A1, agent_type: str) -> str:
    """Compose the planning system prompt for `agent_type`, appending any
    locally-discovered user data inventory the agent has already snapshotted.

    Pure function — no I/O, no Chainlit calls — so it's directly testable.
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

        async with cl.Step(name="📋 Generating Research Plan", type="llm", show_input=False) as step:
            try:
                response = await asyncio.to_thread(agent.llm.invoke, planning_messages)
                plan_text = response.content if hasattr(response, "content") else str(response)
                step.output = plan_text
            except Exception as exc:
                logger.warning("Plan generation failed; proceeding without approval gate", exc_info=True)
                step.output = f"⚠️ Could not generate plan ({exc}). Proceeding without a plan."
                return prompt

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

        mod_res = await cl.AskUserMessage(
            content="Describe the changes you'd like in the plan:",
            timeout=300,
        ).send()

        if mod_res:
            modification_context = mod_res.get("output", "")
