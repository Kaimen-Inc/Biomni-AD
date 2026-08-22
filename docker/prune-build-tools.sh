#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prune-build-tools.sh - drop the compiler toolchain from the built conda env.
#
# ``adworkbench_env.yml`` pins conda-forge's ``compilers`` metapackage because
# several pip dependencies (annoy, fanc, macs2, pybedtools) have no wheels and
# build from source. Once they are built, gcc/g++/gfortran are dead weight - but
# the runtime stage copies the environment wholesale, so without this they ship
# in the final image, several hundred megabytes of it.
#
# What is removed is the *toolchain*: the target directory, gcc's internal
# libraries and the driver binaries. The compiler runtime libraries the built
# extensions actually link against - libgcc_s, libstdc++, libgfortran - live
# directly in ``$PREFIX/lib`` and come from separate packages, so they stay.
# Static archives go too: with no compiler in the image nothing can link them.
#
# Consequence worth knowing: agent-generated code can no longer pip-install a
# package that needs compiling. For a server image that is the right trade, and
# it is better as a clear failure than as a surprise.
#
# The script verifies its own work: every module that is still on disk must
# still import, so a bad prune fails the build instead of shipping a broken
# image. It is a no-op on an environment that never had compilers, which is why
# the minimal environment.yml build can run it unchanged.
#
# Usage: prune-build-tools.sh <conda-env-prefix>
# ---------------------------------------------------------------------------

set -euo pipefail

PREFIX="${1:?usage: prune-build-tools.sh <conda-env-prefix>}"
[ -d "$PREFIX" ] || { echo "ERROR: no such environment: $PREFIX" >&2; exit 1; }

before_kb=$(du -sk "$PREFIX" | cut -f1)

# Target directory and sysroot, named for the build triple - glob rather than
# hardcode so this works on both amd64 (x86_64-...) and arm64 (aarch64-...).
rm -rf "$PREFIX"/*-conda-linux-gnu
rm -rf "$PREFIX"/lib/gcc "$PREFIX"/libexec/gcc
rm -rf "$PREFIX"/share/gcc-* 2>/dev/null || true

# Compiler driver binaries and their unprefixed symlinks. binutils (ld, ar, nm,
# objdump, ...) is deliberately left alone: it is small next to gcc, and
# ``ctypes.util.find_library`` shells out to ``ld``/``objdump`` on Linux, so
# removing it would change how some packages locate shared libraries.
if [ -d "$PREFIX/bin" ]; then
    find "$PREFIX/bin" -maxdepth 1 -name '*-conda-linux-gnu-*' -delete
    for tool in cc c++ cpp gcc g++ gfortran; do
        rm -f "$PREFIX/bin/$tool"
    done
fi

# Static archives: nothing can link them once the toolchain is gone.
find "$PREFIX" -name '*.a' -type f -delete 2>/dev/null || true

after_kb=$(du -sk "$PREFIX" | cut -f1)
echo "prune-build-tools: $(( (before_kb - after_kb) / 1024 )) MB removed ($(( before_kb / 1024 )) MB -> $(( after_kb / 1024 )) MB)"

# ---------------------------------------------------------------------------
# Verify. Anything still present on disk must still import - that is what
# catches a prune that took a shared library with it. Modules absent from this
# environment are skipped rather than failed, so the same check serves both the
# full and the minimal environment.
# ---------------------------------------------------------------------------
"$PREFIX/bin/python" - <<'PY'
import importlib
import importlib.util
import sys

CANDIDATES = [
    # core numeric / plotting stack, present in every environment
    "numpy", "scipy", "pandas", "sklearn", "matplotlib",
    # compiled extensions most likely to expose a broken prune
    "h5py", "pysam", "Bio", "cv2", "rdkit", "scanpy", "anndata", "pyarrow",
    "statsmodels", "numba", "llvmlite", "igraph", "cyvcf2", "pybedtools",
    # the app's own stack
    "chainlit", "langchain", "biomni",
]

checked, skipped, failed = [], [], []
for name in CANDIDATES:
    try:
        if importlib.util.find_spec(name) is None:
            skipped.append(name)
            continue
    except Exception:
        skipped.append(name)
        continue
    try:
        importlib.import_module(name)
        checked.append(name)
    except Exception as exc:  # noqa: BLE001 - report every failure, not the first
        failed.append(f"  {name}: {type(exc).__name__}: {exc}")

print(f"prune-build-tools: imported {len(checked)}, skipped {len(skipped)} not installed")
if failed:
    print("prune-build-tools: FAILED - these are on disk but no longer import:", file=sys.stderr)
    print("\n".join(failed), file=sys.stderr)
    sys.exit(1)
PY
