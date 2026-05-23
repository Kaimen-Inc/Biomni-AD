"""Biomni — biomedical AI agent toolkit.

Public API:

    from biomni import A1, AD1, BiomniConfig
    from biomni.artifact import build_run_id, get_all_files

Agent classes are exposed lazily so `import biomni` stays cheap and does
not pull in pandas / langchain at import time. They are loaded on first
attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .version import __version__

__all__ = ["A1", "AD1", "BiomniConfig", "__version__"]

if TYPE_CHECKING:
    from .agent.a1 import A1
    from .agent.ad1 import AD1
    from .config import BiomniConfig


_LAZY_EXPORTS = {
    "A1": ("biomni.agent.a1", "A1"),
    "AD1": ("biomni.agent.ad1", "AD1"),
    "BiomniConfig": ("biomni.config", "BiomniConfig"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'biomni' has no attribute {name!r}")
    module_path, attr = target
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
