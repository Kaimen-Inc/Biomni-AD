"""Tools that some deployments cannot run, and why.

Most tools need nothing beyond a Python package, and a deployment missing one
simply fails the import. A few need a way out that not every deployment allows,
and advertising those to the agent is worse than useless: the system prompt
tells it to search first, so it reaches for the tool on nearly every question,
and every call is a failed step the user watches.

Import-light (no tool implementations, no LangChain), because the tool
catalogue is filtered through it on every agent build.
"""

from __future__ import annotations

from biomni import credentials
from biomni.llm_proxy import llm_proxy_settings

CLAUDE_WEB_SEARCH = "advanced_web_search_claude"


def claude_web_search_problem() -> str | None:
    """Why ``advanced_web_search_claude`` cannot run here, or ``None`` if it can.

    It calls Anthropic's API directly, with Anthropic's server-side web search
    tool. Behind the platform's LLM proxy that is off limits: every model call
    there must go through the proxy, which meters usage per user and per
    workspace, and an ``ANTHROPIC_API_KEY`` left in the environment would let
    this one tool quietly go around it.
    """
    try:
        proxied = llm_proxy_settings() is not None
    except ValueError:
        proxied = True
    if proxied:
        return (
            f"{CLAUDE_WEB_SEARCH} is not available in this deployment: model calls go through the platform's "
            "LLM proxy, and this tool would call Anthropic directly. Use search_google() instead."
        )

    from biomni.config import default_config

    if "claude" not in default_config.llm.lower():
        return f"{CLAUDE_WEB_SEARCH} needs a Claude model, and this deployment uses {default_config.llm}."
    if not credentials.getenv("ANTHROPIC_API_KEY"):
        return f"{CLAUDE_WEB_SEARCH} needs ANTHROPIC_API_KEY, which is not set."
    return None


def unavailable_tools() -> dict[str, str]:
    """Tool name -> why this deployment cannot run it."""
    problems = {CLAUDE_WEB_SEARCH: claude_web_search_problem()}
    return {name: problem for name, problem in problems.items() if problem}


__all__ = ["CLAUDE_WEB_SEARCH", "claude_web_search_problem", "unavailable_tools"]
