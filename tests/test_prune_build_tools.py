"""Tests for docker/prune-build-tools.sh.

The script runs once, inside the image build, against an environment that only
exists there - so without this it is the one build-critical artifact with no way
to be exercised before a release. It deletes files out of the conda prefix, and
the failure mode if it deletes one file too many is an image whose scientific
stack no longer imports.

The prefix here is synthetic: it mirrors conda-forge's gcc layout closely enough
to pin down what the globs match, which is the part that is easy to get wrong
(and is architecture-dependent - the target directory is named for the build
triple).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "docker" / "prune-build-tools.sh"

# The compiler toolchain, which must go.
TOOLCHAIN_DIRS = [
    "lib/gcc/x86_64-conda-linux-gnu/13.2.0",
    "libexec/gcc/x86_64-conda-linux-gnu",
    "x86_64-conda-linux-gnu/sysroot/usr/lib",
    "share/gcc-13.2.0",
]
TOOLCHAIN_BINS = ["gcc", "g++", "gfortran", "cc", "c++", "cpp", "x86_64-conda-linux-gnu-gcc"]

# The compiler *runtime* libraries every compiled extension links against, and
# binutils, which ctypes.util.find_library shells out to on Linux. Removing any
# of these is the failure this test exists to catch.
MUST_SURVIVE = [
    "lib/libgcc_s.so.1",
    "lib/libstdc++.so.6",
    "lib/libgfortran.so.5",
    "bin/ld",
    "bin/objdump",
    "bin/nm",
    "bin/ar",
]


def _make_env(root: Path) -> Path:
    """A synthetic prefix that is a real Python environment.

    Built with ``venv`` rather than by symlinking an interpreter in: the script
    runs ``$PREFIX/bin/python``, and a *symlinked* interpreter resolves its
    prefix to this directory, finds no environment, and falls back to the base
    installation - so the self-check would import the whole development stack on
    every run. A genuine venv has an empty site-packages, so every candidate
    resolves as "not installed" and is skipped, which is both fast and closer to
    what the script meets in the image.
    """
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=120)
    return root


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    """A synthetic conda prefix carrying a toolchain and the libs that must stay."""
    root = _make_env(tmp_path / "env")
    for d in TOOLCHAIN_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(parents=True, exist_ok=True)

    for name in TOOLCHAIN_BINS:
        (root / "bin" / name).write_text("#!/bin/sh\n")
    # Static archives, which are dead weight once nothing can link them.
    (root / "lib/gcc/x86_64-conda-linux-gnu/13.2.0/libgcc.a").write_bytes(b"\0" * 4096)
    (root / "x86_64-conda-linux-gnu/sysroot/usr/lib/libc.a").write_bytes(b"\0" * 4096)

    for rel in MUST_SURVIVE:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("keep me")

    return root


def _run(prefix: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(prefix)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_removes_the_toolchain(prefix: Path) -> None:
    result = _run(prefix)
    assert result.returncode == 0, result.stderr

    for name in TOOLCHAIN_BINS:
        assert not (prefix / "bin" / name).exists(), f"{name} survived"
    assert not (prefix / "lib" / "gcc").exists()
    assert not (prefix / "libexec" / "gcc").exists()
    assert not (prefix / "x86_64-conda-linux-gnu").exists()


def test_keeps_the_runtime_libraries_and_binutils(prefix: Path) -> None:
    """The whole point: strip the compiler, keep what compiled code needs."""
    result = _run(prefix)
    assert result.returncode == 0, result.stderr

    for rel in MUST_SURVIVE:
        assert (prefix / rel).exists(), f"{rel} was removed - compiled extensions would break"


def test_removes_static_archives(prefix: Path) -> None:
    _run(prefix)
    assert not list(prefix.rglob("*.a"))


def test_reports_what_it_reclaimed(prefix: Path) -> None:
    result = _run(prefix)
    assert "prune-build-tools:" in result.stdout
    assert "MB removed" in result.stdout


def test_self_check_runs_and_passes(prefix: Path) -> None:
    """The import check is the safety net; it has to actually execute."""
    result = _run(prefix)
    assert "imported" in result.stdout, result.stdout


def test_self_check_actually_imports_against_a_populated_interpreter(tmp_path: Path) -> None:
    """With a real interpreter the check imports rather than skipping everything.

    This is the branch that catches a prune which took a shared library with it,
    so it must be exercised against an environment that has the stack installed.
    """
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    os.symlink(sys.executable, root / "bin" / "python")

    result = _run(root)

    assert result.returncode == 0, result.stderr
    imported = int(result.stdout.split("imported ")[1].split(",")[0])
    assert imported > 0, f"expected real imports, got: {result.stdout}"


def test_is_a_no_op_on_an_environment_without_compilers(tmp_path: Path) -> None:
    """The minimal environment never had a toolchain; the same script must still pass."""
    root = _make_env(tmp_path / "env")
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "libgcc_s.so.1").write_text("keep me")

    result = _run(root)

    assert result.returncode == 0, result.stderr
    assert (root / "lib" / "libgcc_s.so.1").exists()


def test_arm64_target_directory_is_matched_too(tmp_path: Path) -> None:
    """The target dir is named for the build triple, so the glob must not assume x86."""
    root = _make_env(tmp_path / "env")
    (root / "aarch64-conda-linux-gnu" / "sysroot").mkdir(parents=True)
    (root / "bin" / "aarch64-conda-linux-gnu-gcc").write_text("#!/bin/sh\n")

    assert _run(root).returncode == 0
    assert not (root / "aarch64-conda-linux-gnu").exists()
    assert not (root / "bin" / "aarch64-conda-linux-gnu-gcc").exists()


def test_missing_prefix_is_an_error(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "nope")], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "no such environment" in result.stderr


def test_script_is_executable_and_shellcheck_clean() -> None:
    assert SCRIPT.exists()
    assert os.access(SCRIPT, os.X_OK), "script must be executable"
    syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    if shutil.which("shellcheck"):
        sc = subprocess.run(["shellcheck", "-S", "warning", str(SCRIPT)], capture_output=True, text=True)
        assert sc.returncode == 0, sc.stdout
