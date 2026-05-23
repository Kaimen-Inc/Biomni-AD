"""Tests for the biomni top-level public API surface."""
from __future__ import annotations

import sys

import pytest


def test_import_biomni_is_cheap() -> None:
    """`import biomni` must not eagerly pull in the heavy agent stack."""
    # If biomni was already imported by another test, drop it first
    sys.modules.pop("biomni", None)
    sys.modules.pop("biomni.agent.a1", None)

    import biomni  # noqa: F401

    # The agent module is heavy (pandas, langchain, langgraph) — it must
    # only load on first attribute access.
    assert "biomni.agent.a1" not in sys.modules


def test_version_exported() -> None:
    import biomni

    assert isinstance(biomni.__version__, str)
    assert biomni.__version__  # non-empty


def test_all_lists_lazy_exports() -> None:
    import biomni

    for name in ("A1", "AD1", "BiomniConfig", "__version__"):
        assert name in biomni.__all__


def test_lazy_access_resolves_biomni_config() -> None:
    import biomni

    from biomni.config import BiomniConfig as Direct

    assert biomni.BiomniConfig is Direct


def test_unknown_attribute_raises_attribute_error() -> None:
    import biomni

    with pytest.raises(AttributeError, match="DoesNotExist"):
        biomni.DoesNotExist  # noqa: B018


def test_dir_includes_lazy_names() -> None:
    import biomni

    names = dir(biomni)
    assert "A1" in names
    assert "AD1" in names
    assert "BiomniConfig" in names
