"""The agent's system prompt must name only tools the agent can actually call.

It tells the model which functions to reach for first, on every question. A
name that does not exist - three of the old ones did not - or a tool this
deployment withholds is a failed step the user watches, every time.
"""

from __future__ import annotations

import re

import pytest

try:
    from biomni.agent.a1 import A1

    HAS_AGENT = True
except Exception:  # pragma: no cover - depends on the installed extras
    HAS_AGENT = False

pytestmark = pytest.mark.skipif(not HAS_AGENT, reason="agent stack (pandas/langchain/langgraph) not installed")


class _StubAgent:
    def __init__(self, module2api) -> None:
        self.module2api = module2api

    _advertises_tool = A1._advertises_tool if HAS_AGENT else None
    _tool_priority_instructions = A1._tool_priority_instructions if HAS_AGENT else None


def _full_catalogue() -> dict[str, list[dict]]:
    import importlib

    from biomni.utils import read_module2api

    # Every tool that exists, whatever this machine's deployment would withhold.
    fields = [module.rsplit(".", 1)[1] for module in read_module2api()]
    return {
        f"biomni.tool.{field}": importlib.import_module(f"biomni.tool.tool_description.{field}").description
        for field in fields
    }


def _named_functions(text: str) -> set[str]:
    return set(re.findall(r"\b([a-z_][a-z0-9_]*)\(\)", text))


def test_every_function_the_prompt_names_exists():
    catalogue = _full_catalogue()
    known = {api["name"] for apis in catalogue.values() for api in apis}
    named = _named_functions(_StubAgent(catalogue)._tool_priority_instructions())
    assert named, "the prompt should name the tools to use first"
    assert named <= known, f"named but not a tool: {sorted(named - known)}"


def test_the_claude_web_search_is_named_only_when_advertised():
    catalogue = _full_catalogue()
    assert "advanced_web_search_claude()" in _StubAgent(catalogue)._tool_priority_instructions()

    withheld = {
        module: [api for api in apis if api["name"] != "advanced_web_search_claude"]
        for module, apis in catalogue.items()
    }
    text = _StubAgent(withheld)._tool_priority_instructions()
    assert "advanced_web_search_claude" not in text
    assert "search_google()" in text
