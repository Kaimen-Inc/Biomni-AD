"""Tests for biomni.env_probe - narrowing the advertised catalogue to reality.

The system prompt tells the model to prefer locally installed libraries, so a
catalogue entry the deployment does not have is not a harmless overstatement: it
is a plan the agent will commit to and then fail to execute, several steps in,
in front of the user. These lock in what counts as "installed", and the two
safety valves that stop the filter itself becoming the failure.
"""

from __future__ import annotations

import pytest
from biomni import env_probe


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probing is cached per process; every test starts cold."""
    monkeypatch.delenv("BIOMNI_ADVERTISE_ALL_LIBRARIES", raising=False)
    env_probe._available_names.cache_clear()
    env_probe._r_packages.cache_clear()
    yield
    env_probe._available_names.cache_clear()
    env_probe._r_packages.cache_clear()


# --------------------------------------------------------------------------- #
# Detection strategies
# --------------------------------------------------------------------------- #


def test_installed_distribution_counts_as_available() -> None:
    # pytest is running, so its distribution metadata is necessarily present.
    assert env_probe.is_available("pytest")


def test_importable_module_counts_as_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers names with no queryable distribution, e.g. `harmony`."""
    monkeypatch.setattr(env_probe, "_has_distribution", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_executable", lambda name: False)
    assert env_probe.is_available("json")


def test_executable_counts_as_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most of the bioinformatics catalogue is binaries, not Python packages."""
    monkeypatch.setattr(env_probe, "_has_distribution", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_module", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_executable", lambda name: name == "bwa")

    assert env_probe.is_available("bwa")
    assert not env_probe.is_available("bowtie2")


def test_absent_everywhere_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    for probe in ("_has_distribution", "_has_module", "_has_executable"):
        monkeypatch.setattr(env_probe, probe, lambda name: False)
    assert not env_probe.is_available("definitely-not-installed-xyz")


def test_module_probe_never_imports_the_module() -> None:
    """`find_spec` on a top-level name must not execute the package.

    Importing 113 libraries to test them would cost seconds and run arbitrary
    package-level code in the agent's process.
    """
    import sys

    assert "wave" not in sys.modules  # stdlib module nothing else pulls in
    env_probe._has_module("wave")
    assert "wave" not in sys.modules


def test_broken_package_reads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-installed package raises from find_spec; it would fail on use anyway."""

    def boom(name):
        raise ValueError("broken metadata")

    monkeypatch.setattr(env_probe.importlib.util, "find_spec", boom)
    assert env_probe._has_module("anything") is False


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


def test_headless_opencv_satisfies_the_catalogue_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """We ship opencv-python-headless; both give the same `import cv2`."""
    monkeypatch.setattr(env_probe, "_has_module", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_executable", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_distribution", lambda name: name == "opencv-python-headless")

    assert env_probe.is_available("opencv-python")


def test_alias_covers_a_tool_whose_binary_is_named_differently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_probe, "_has_distribution", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_module", lambda name: False)
    monkeypatch.setattr(env_probe, "_has_executable", lambda name: name == "obabel")

    assert env_probe.is_available("openbabel")


# --------------------------------------------------------------------------- #
# R
# --------------------------------------------------------------------------- #


def test_r_packages_are_absent_without_r(monkeypatch: pytest.MonkeyPatch) -> None:
    """No R in the image means no DESeq2, whatever Python says."""
    monkeypatch.setattr(env_probe.shutil, "which", lambda name, path=None: None)

    assert not env_probe.is_available("DESeq2")
    assert not env_probe.is_available("limma")


def test_r_packages_resolve_from_one_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_probe.shutil, "which", lambda name, path=None: "/usr/bin/Rscript")
    calls = []

    class Result:
        returncode = 0
        stdout = "DESeq2\nMatrix\nggplot2\n"

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(env_probe.subprocess, "run", fake_run)

    assert env_probe.is_available("DESeq2")
    assert env_probe.is_available("ggplot2")
    assert not env_probe.is_available("edgeR")
    # One subprocess for the whole catalogue, not one per package.
    assert len(calls) == 1


def test_a_failing_rscript_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_probe.shutil, "which", lambda name, path=None: "/usr/bin/Rscript")

    def boom(cmd, **kwargs):
        raise OSError("R exploded")

    monkeypatch.setattr(env_probe.subprocess, "run", boom)
    assert not env_probe.is_available("DESeq2")


# --------------------------------------------------------------------------- #
# Filtering, and its safety valves
# --------------------------------------------------------------------------- #


CATALOG = {"scanpy": "single cell", "DESeq2": "differential expression", "bwa": "aligner"}


def test_filter_keeps_only_what_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_probe, "is_available", lambda name: name == "scanpy")
    assert env_probe.filter_library_catalog(CATALOG) == {"scanpy": "single cell"}


def test_filter_preserves_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_probe, "is_available", lambda name: name in {"scanpy", "bwa"})
    out = env_probe.filter_library_catalog(CATALOG)
    assert out["scanpy"] == "single cell"
    assert out["bwa"] == "aligner"


def test_env_flag_disables_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a deployment that installs libraries after the image is built."""
    monkeypatch.setenv("BIOMNI_ADVERTISE_ALL_LIBRARIES", "true")
    monkeypatch.setattr(env_probe, "is_available", lambda name: False)

    assert env_probe.filter_library_catalog(CATALOG) == CATALOG


def test_detecting_nothing_leaves_the_catalogue_alone(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A probe that resolves nothing is broken, not proof of an empty image.

    Handing the agent an empty toolbox would be a worse failure than
    over-advertising, so the filter fails open and says so.
    """
    monkeypatch.setattr(env_probe, "is_available", lambda name: False)

    with caplog.at_level("WARNING", logger="biomni.env_probe"):
        assert env_probe.filter_library_catalog(CATALOG) == CATALOG

    assert any("no advertised library could be detected" in r.message for r in caplog.records)


def test_empty_catalogue_is_returned_unchanged() -> None:
    assert env_probe.filter_library_catalog({}) == {}


def test_probing_is_cached_across_agent_constructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Chainlit app builds an agent per chat; this must not re-probe each time."""
    calls = []

    def counting(name):
        calls.append(name)
        return True

    monkeypatch.setattr(env_probe, "is_available", counting)

    env_probe.filter_library_catalog(CATALOG)
    first = len(calls)
    env_probe.filter_library_catalog(CATALOG)

    assert first == len(CATALOG)
    assert len(calls) == first, "second call re-probed instead of using the cache"


def test_the_real_catalogue_filters_to_something_usable() -> None:
    """End to end against the shipped catalogue, whatever this machine has."""
    from biomni import env_desc

    filtered = env_probe.filter_library_catalog(env_desc.library_content_dict)

    assert 0 < len(filtered) <= len(env_desc.library_content_dict)
    assert set(filtered).issubset(env_desc.library_content_dict)
    # numpy and pandas are hard dependencies of the package itself, so they are
    # present in any environment that can import biomni at all.
    assert {"numpy", "pandas"}.issubset(filtered)


def test_every_path_returns_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A1.add_tool mutates this dict, and the caller passes the module global.

    Handing back the same object would let one chat session's add_tool rewrite
    the catalogue for every session after it in that process - the same
    cross-session leak that the REPL namespace had.
    """
    from biomni import env_desc

    # filtering path
    monkeypatch.setattr(env_probe, "is_available", lambda name: True)
    assert env_probe.filter_library_catalog(CATALOG) is not CATALOG

    # nothing-detected path
    monkeypatch.setattr(env_probe, "is_available", lambda name: False)
    env_probe._available_names.cache_clear()
    assert env_probe.filter_library_catalog(CATALOG) is not CATALOG

    # opt-out path
    monkeypatch.setenv("BIOMNI_ADVERTISE_ALL_LIBRARIES", "true")
    assert env_probe.filter_library_catalog(CATALOG) is not CATALOG

    # and the real module-level catalogue is never handed back by identity
    assert env_probe.filter_library_catalog(env_desc.library_content_dict) is not env_desc.library_content_dict


def test_mutating_the_result_does_not_touch_the_module_catalogue() -> None:
    from biomni import env_desc

    before = dict(env_desc.library_content_dict)
    filtered = env_probe.filter_library_catalog(env_desc.library_content_dict)
    filtered["a_custom_tool_added_by_one_session"] = "should not escape"

    assert env_desc.library_content_dict == before
