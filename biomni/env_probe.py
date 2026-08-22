"""Which of the advertised libraries this deployment actually has.

:mod:`biomni.env_desc` catalogues ~113 libraries, and the system prompt tells the
model to "prioritize locally installed tools and libraries". That catalogue
describes the *reference* environment, not whichever image is running: built from
``biomni_env/environment.yml`` a container supplies ten of them, and even the AD
Workbench environment has no R and no docking tools. Advertising the remainder
does not make them available - it invites the agent to plan around ``gseapy`` or
``DESeq2`` and then fail at the import, several steps in, in front of the user.

A missing library is unlike a missing data-lake file. That gets fetched on demand
(see ``A1._ensure_data_lake_files``), so it is honest to list it and tag it. A
library cannot be produced at run time, so the catalogue is *filtered* rather
than tagged: the agent should never see a tool it cannot use.

Detection is deliberately side-effect free and cheap, because it runs once per
process and the alternative - importing 113 libraries to see which ones work -
would cost seconds and execute arbitrary package import code. Each name is tried
as, in order:

1. an installed **distribution** (``importlib.metadata``), which is metadata-only
   and covers conda-installed packages too, since those ship ``.dist-info``;
2. an importable **top-level module**, for the handful whose distribution name
   differs from anything queryable (``harmony``);
3. an **executable**, which is how the bioinformatics tools appear - ``bwa``,
   ``samtools``, ``bedtools`` are binaries on ``PATH``, not Python packages;
4. an **R package**, probed with a single cached ``Rscript`` call, and reported
   absent when R is not installed at all.

Set ``BIOMNI_ADVERTISE_ALL_LIBRARIES=true`` to skip filtering, for a deployment
that installs libraries after the image is built.
"""

from __future__ import annotations

import functools
import importlib.metadata as metadata
import importlib.util
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# Catalogue entries whose name is not directly queryable. Everything else goes
# through the default chain, which already covers the common case where the
# catalogue name is the distribution name (``scanpy``) or the executable name
# (``bedtools``).
_ALIASES: dict[str, tuple[str, ...]] = {
    # We ship the headless build; both provide the same ``import cv2``, and
    # headless is the correct choice for a server with no display.
    "opencv-python": ("opencv-python", "opencv-python-headless"),
    # Distributed as a suite of Perl scripts rather than one binary named "Homer".
    "Homer": ("findMotifs.pl", "findMotifsGenome.pl"),
    # The Open Babel CLI is "obabel"; the Python bindings are "openbabel".
    "openbabel": ("openbabel", "obabel"),
    # Packaged under either capitalisation depending on the channel.
    "FastTree": ("FastTree", "fasttree"),
    "PyMassSpec": ("PyMassSpec", "pyms"),
}

# Below this share of the catalogue, assume the probe is broken rather than the
# environment bare, and fail open. The minimal environment resolves ~9% (10 of
# 113), so the threshold has to sit under that.
_MIN_PLAUSIBLE_FRACTION = 0.05

# Probed through R rather than Python. Listed explicitly because there is no way
# to tell an R package from a Python one by name alone.
_R_PACKAGES = frozenset(
    {
        "DESeq2",
        "Matrix",
        "WGCNA",
        "clusterProfiler",
        "dplyr",
        "edgeR",
        "ggplot2",
        "limma",
        "readr",
        "stringr",
        "tidyr",
    }
)


def _has_distribution(name: str) -> bool:
    try:
        metadata.distribution(name)
    except Exception:
        return False
    return True


def _has_module(name: str) -> bool:
    """Whether ``name`` is importable, without importing it.

    Only top-level names are tried, so ``find_spec`` does not import a parent
    package as a side effect.
    """
    if "." in name or not name.isidentifier():
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        # A broken or partially installed package raises here rather than
        # returning None. Treat it as absent: it would fail on use anyway.
        return False


def _has_executable(name: str) -> bool:
    """Whether ``name`` is on ``PATH`` or beside the running interpreter.

    The interpreter's own ``bin`` is checked because the app is routinely started
    without activating the conda environment (``run_chainlit.sh`` does exactly
    that), which leaves the environment's tools off ``PATH`` while they are very
    much installed.
    """
    if shutil.which(name):
        return True
    return bool(shutil.which(name, path=os.path.dirname(sys.executable)))


@functools.lru_cache(maxsize=1)
def _r_packages() -> frozenset[str]:
    """Installed R packages, or an empty set when R is unavailable.

    One subprocess for the whole catalogue, cached for the process. Failure is
    never fatal: a deployment without R simply has no R packages.
    """
    rscript = shutil.which("Rscript") or shutil.which("Rscript", path=os.path.dirname(sys.executable))
    if not rscript:
        return frozenset()
    try:
        result = subprocess.run(
            [rscript, "-e", "cat(rownames(installed.packages()), sep='\\n')"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        logger.debug("could not enumerate R packages", exc_info=True)
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    return frozenset(line.strip() for line in result.stdout.splitlines() if line.strip())


def is_available(name: str) -> bool:
    """Whether the advertised library ``name`` is actually usable here."""
    if name in _R_PACKAGES:
        return name in _r_packages()

    candidates = _ALIASES.get(name, (name,))
    for candidate in candidates:
        if _has_distribution(candidate) or _has_module(candidate) or _has_executable(candidate):
            return True
    # Distribution names are matched case-insensitively and treat "-"/"_" alike
    # by importlib, but module and executable lookups are literal, so try the
    # lowercase form for entries catalogued in title case (FlowIO, PyLabRobot).
    lowered = name.lower()
    return lowered != name and (_has_module(lowered) or _has_executable(lowered))


def advertise_all() -> bool:
    """Whether to skip filtering (``BIOMNI_ADVERTISE_ALL_LIBRARIES``)."""
    return os.getenv("BIOMNI_ADVERTISE_ALL_LIBRARIES", "").strip().lower() in {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=4)
def _available_names(names: tuple[str, ...]) -> frozenset[str]:
    """Cached availability for one catalogue.

    The Chainlit app builds an agent per chat session, so this must not re-probe
    on every new conversation.
    """
    return frozenset(n for n in names if is_available(n))


def filter_library_catalog(catalog: dict[str, str]) -> dict[str, str]:
    """The subset of ``catalog`` this deployment can actually run.

    Every path returns a *copy*, including the two that keep the catalogue
    whole. ``A1.add_tool`` and ``A1.remove_tool`` mutate the dict they are
    given, and the caller passes the module-level ``library_content_dict``
    straight from :mod:`biomni.env_desc`; handing back the same object would let
    one chat session's ``add_tool`` rewrite the catalogue for every session
    afterwards in that process.

    The catalogue is kept whole when ``BIOMNI_ADVERTISE_ALL_LIBRARIES`` is set,
    and also when *nothing* resolves - that means the probe is wrong about this
    environment rather than that the environment is empty, and silently handing
    the agent an empty toolbox would be worse than over-advertising.
    """
    if advertise_all() or not catalog:
        return dict(catalog)

    available = _available_names(tuple(catalog))
    # A proportion, not "zero". numpy, pandas and pyarrow are hard dependencies
    # of the package itself and are catalogue entries, so anything that can
    # import biomni resolves at least those three - an "is it empty" guard could
    # never fire, and a systemic probe failure would quietly leave the agent with
    # a handful of tools instead of tripping the safety valve.
    if len(available) < _MIN_PLAUSIBLE_FRACTION * len(catalog):
        logger.warning(
            "only %d of %d advertised libraries were detected, which looks like a broken probe "
            "rather than a bare environment; leaving the catalogue unfiltered. "
            "Set BIOMNI_ADVERTISE_ALL_LIBRARIES=true to silence this.",
            len(available),
            len(catalog),
        )
        return dict(catalog)

    filtered = {name: desc for name, desc in catalog.items() if name in available}
    missing = len(catalog) - len(filtered)
    if missing:
        logger.info(
            "library catalogue: %d of %d advertised libraries are installed; %d hidden from the agent",
            len(filtered),
            len(catalog),
            missing,
        )
    return filtered


__all__ = ["advertise_all", "filter_library_catalog", "is_available"]
