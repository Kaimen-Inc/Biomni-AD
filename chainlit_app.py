"""
Biomni Chainlit UI
==================
Interactive biomedical AI agent with plan-then-approve workflow.

Usage:
    chainlit run chainlit_app.py

Environment variables:
    BIOMNI_LLM      LLM model name (default: claude-sonnet-4-5)
    BIOMNI_PATH     Data directory (default: ./data)
    ANTHROPIC_API_KEY / OPENAI_API_KEY  (as needed by your chosen LLM)
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
    "You are an Alzheimer's disease research assistant planning a task. "
    "Given the user's research question, write a concise numbered plan "
    "of 3 to 7 steps describing exactly how you will solve it. "
    "IMPORTANT: Always follow this strict tool priority order in your plan: "
    "(1) AD/dementia data lake FIRST — query ADNI, ROSMAP, UK Biobank, NACC, or other "
    "available AD-specific datasets before any other source; "
    "(2) web/literature search second (advanced_web_search, search_pubmed, search_biorxiv) "
    "to supplement with published findings; "
    "(3) built-in domain tools third (database queries, biomarker tools); "
    "(4) custom code generation only as a last resort. "
    "Do NOT simulate or fabricate data. "
    "Mention specific datasets, tools, or analyses you will use. "
    "Be specific but brief. Do not execute any code yet."
)

# Auto-detect Azure setup: if DEPLOYMENT_NAME + ENDPOINT_URL are set, default to Azure
_azure_deployment = os.getenv("DEPLOYMENT_NAME")
_azure_endpoint = os.getenv("ENDPOINT_URL")
_azure_default = f"azure-{_azure_deployment}" if (_azure_deployment and _azure_endpoint) else None
DEFAULT_LLM = os.getenv("BIOMNI_LLM") or _azure_default or "claude-sonnet-4-5"
DEFAULT_PATH = os.getenv("BIOMNI_PATH", "./data")
# Set BIOMNI_AGENT=a1 to force the A1 agent on startup (skips the profile selector)
FORCE_AGENT = os.getenv("BIOMNI_AGENT", "").lower()  # "a1" | "ad1" | ""

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")


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
                "**AD1 — Alzheimer's Disease Co-Scientist**\n\n"
                "Specialized for AD/dementia research: ADNI, ROSMAP, UK Biobank, "
                "multi-omics, biomarker discovery, drug repurposing."
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
        label="Amyloid & tau biomarkers in CSF",
        message=(
            "Analyze the relationship between CSF amyloid-β42, p-tau181, and t-tau levels "
            "across MCI and AD patients in the ADNI cohort. Identify which combination best "
            "predicts conversion from MCI to AD within 2 years."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="Differential gene expression in AD brain",
        message=(
            "Perform differential expression analysis comparing AD vs. control samples in the "
            "ROSMAP bulk RNA-seq dataset. Focus on genes in the APP processing pathway and "
            "highlight any that overlap with GWAS hits from the latest AD meta-analysis."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="Drug repurposing for neuroinflammation",
        message=(
            "Identify existing FDA-approved drugs that could be repurposed to target "
            "neuroinflammation in Alzheimer's disease. Cross-reference known TREM2 and "
            "microglia activation pathways with drug-target interaction databases."
        ),
        icon="/public/avatars/ad1.png",
    ),
    cl.Starter(
        label="Single-cell microglia subtypes in AD",
        message=(
            "Using single-cell RNA-seq data, characterize microglia subtypes present in "
            "Alzheimer's disease brain tissue. Identify disease-associated microglia (DAM) "
            "markers and compare their abundance across Braak staging levels."
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
    except Exception as exc:
        await cl.Message(content=f"Failed to initialize {label}: {exc}").send()
        return

    # No greeting message sent here — the centered welcome screen (chainlit.md)
    # stays visible until the user sends their first message.


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
            async with cl.Step(name="🧠 AD Context Detected", show_input=False) as step:
                await run_in_executor(agent._inject_ad_context)
                step.output = (
                    "Specialized AD/dementia data sourcing protocols injected into context."
                )

    # ------------------------------------------------------------------
    # Phase 2: Tool retrieval
    # ------------------------------------------------------------------
    if getattr(agent, "use_tool_retriever", False):
        async with cl.Step(name="🔍 Selecting Resources", show_input=False) as step:
            try:
                resources = await run_in_executor(
                    agent._prepare_resources_for_retrieval, prompt
                )
                if resources:
                    await run_in_executor(
                        agent.update_system_prompt_with_selected_resources, resources
                    )
                    step.output = f"Selected {len(resources)} relevant tools, datasets, and libraries."
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
    initial_files = _get_all_files(os.getcwd())
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
    # Phase 5: Artifact saving (AD1 only)
    # ------------------------------------------------------------------
    if agent_type == "ad1" and final_state and hasattr(agent, "_save_run_artifacts"):
        await _save_ad1_artifacts(agent, final_state, initial_files)


# ---------------------------------------------------------------------------
# Interactive planning helpers
# ---------------------------------------------------------------------------

async def _interactive_planning(agent, prompt: str, agent_type: str = "a1") -> str | None:
    """
    Generate a research plan, show it for user approval, and handle revisions.
    Returns the (possibly modified) prompt on approval, or None if cancelled.
    """
    base_prompt = AD1_PLANNING_SYSTEM_PROMPT if agent_type == "ad1" else PLANNING_SYSTEM_PROMPT
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

        async with cl.Step(name="📋 Generating Research Plan", show_input=False) as step:
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
                async with cl.Step(name="🤔 Thinking", show_input=False) as step:
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

            code_step = cl.Step(name=f"⚡ Executing {language.upper()}", show_input=False)
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

            async with cl.Step(name="👁 Observation", show_input=False) as obs_step:
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    runs_root = os.path.abspath(os.path.join(os.getcwd(), "runs"))
    os.makedirs(runs_root, exist_ok=True)
    current_run_dir = os.path.join(runs_root, run_id)
    os.makedirs(current_run_dir, exist_ok=True)

    # Sync internal state so artifact methods work correctly
    if "messages" in final_state:
        agent.raw_log = list(final_state["messages"])
    agent._conversation_state = final_state

    async with cl.Step(name="📦 Saving Artifacts", show_input=False) as step:
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
