"""What the browser UI may offer, decided per deployment.

Two switches, both for the AD Workbench (GRIP), where the platform rather than
this app decides how a result leaves a workspace and who is signed in:

``BIOMNI_DISABLE_DOWNLOADS``
    A GRIP workspace lets results out only through its airlock, so the app must
    not offer a second way. This is not data-loss prevention: a plot on screen
    can still be screenshotted and text can still be copied, which is what the
    airlock's review is for. What the switch removes is every download the *app*
    itself offers - the run archive, and any stored file a reopened conversation
    would replay - so the page has no link or button that hands over something
    the app produced. What a user attached to their own message is theirs, and
    stays.

``BIOMNI_HIDE_LOGOUT``
    Behind the gateway the app signs a user straight back in, so Chainlit's
    "Logout" menu item does nothing a user can see but reload the page.

Both default off, because the demo and compose deployments have no airlock and
rely on the download.

``BIOMNI_AIRLOCK_ROOT`` is not a switch but a fact the first one depends on: the
folder the airlock exports from. The app cannot see the platform's airlock, so with
downloads off it warns about results written outside the folder it is told of. Left
unset, that folder is the workspace the app reads its data from, which is how GRIP's
first deployments were laid out. A deployment that gives the app a dataset to read
and a separate folder to write to, with the airlock reading the second, names it
here; without that, every run in it would be flagged.

Nothing here imports Chainlit: CI installs no chainlit extra and must still be able
to test the decisions made in this module.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from biomni.workspace_prefs import FALSE_VALUES, TRUE_VALUES, env_flag

logger = logging.getLogger(__name__)

DISABLE_DOWNLOADS_ENV = "BIOMNI_DISABLE_DOWNLOADS"
HIDE_LOGOUT_ENV = "BIOMNI_HIDE_LOGOUT"
AIRLOCK_ROOT_ENV = "BIOMNI_AIRLOCK_ROOT"

# The element types Chainlit's frontend draws with a download control: a File is
# an <a download> link and a Pdf is a viewer with a "Download PDF" button.
# Images, text and the rest have none - a plot stays visible, which is the point
# of showing it - so they are never withheld.
DOWNLOADABLE_ELEMENT_TYPES = frozenset({"file", "pdf"})

# `.chainlit/config.toml` names the first; the second imports it and adds the
# rule that hides the logout item. Chainlit takes one custom stylesheet, and the
# page reads it per request, so choosing between them once at startup is enough.
DEFAULT_STYLESHEET = "/public/biomni.css"
HIDE_LOGOUT_STYLESHEET = "/public/biomni-hide-logout.css"


# Values already complained about, so a typo is reported once rather than on every
# conversation read and every run.
_reported_values: set[str] = set()


def downloads_disabled() -> bool:
    """Whether the app may not hand out files itself.

    Unlike the other switches, an unrecognised value means *on*. This one enforces
    a platform rule, so a typo such as ``ture`` must not quietly leave downloads
    open on a workspace that may not have them; the cost of guessing wrong the
    other way is a download button that is missing, which an operator notices and
    fixes from the logged reason.
    """
    raw = os.getenv(DISABLE_DOWNLOADS_ENV, "").strip().lower()
    if raw in TRUE_VALUES:
        return True
    if not raw or raw in FALSE_VALUES:
        return False
    if raw not in _reported_values:
        _reported_values.add(raw)
        logger.error(
            "%s=%r is not a recognised value, so downloads are turned OFF to be safe. Use true or false.",
            DISABLE_DOWNLOADS_ENV,
            raw,
        )
    return True


def logout_hidden() -> bool:
    return env_flag(HIDE_LOGOUT_ENV)


def custom_stylesheet(configured: str | None) -> str | None:
    """The stylesheet the page should load: the configured one, or its logout-hiding variant.

    The variant imports ``biomni.css`` and nothing else, so it stands in for that
    one stylesheet. An operator who pointed ``custom_css`` somewhere of their own
    loses it while the switch is on, which is said at startup rather than left
    to be noticed on the page.
    """
    if not logout_hidden():
        return configured
    if configured and configured not in (DEFAULT_STYLESHEET, HIDE_LOGOUT_STYLESHEET):
        logger.warning(
            "%s replaces the configured stylesheet %s with %s, which only imports biomni.css. "
            'To keep your own, leave the switch off and add @import url("biomni-hide-logout.css"); at the top of it.',
            HIDE_LOGOUT_ENV,
            configured,
            HIDE_LOGOUT_STYLESHEET,
        )
    return HIDE_LOGOUT_STYLESHEET


# --------------------------------------------------------------------------- #
# Stored conversations
# --------------------------------------------------------------------------- #


def is_withheld_download(element_type: str | None, owner_step_type: str | None) -> bool:
    """Whether a stored element must not reach a browser while downloads are off.

    What a user attached to their own message is theirs, not something the app
    produced, so it stays; anything else with a download control goes. An element
    whose owning step is unknown counts as not theirs: the safe answer for a
    switch whose whole job is to keep files in.
    """
    return element_type in DOWNLOADABLE_ELEMENT_TYPES and owner_step_type != "user_message"


# The offer the completion message makes while downloads are on, as
# chainlit_app._offer_run_results words it (a test renders that message and
# checks this still matches it). In a conversation saved back then it is a
# promise the page can no longer keep once the archive is withheld.
_DOWNLOAD_OFFER = re.compile(
    r"Download everything this run produced \([^)]*\), or find it on the server at (`[^`]*`)\."
)
_OFFER_WITHDRAWN = r"Results are saved in \1. Downloads are turned off in this deployment."


def withhold_downloads(thread: dict[str, Any]) -> dict[str, Any]:
    """Drop the downloadable outputs from a stored conversation, in place.

    Applied where the data layer hands a thread back, so every way a browser can
    reach one - reopening it, the thread API, the shared view - is covered by the
    same rule. Conversations saved before the switch was turned on carry the run
    archives the old completion message offered; without this they would go on
    handing them out, as a link in the replayed transcript.

    The completion message that offered the archive is reworded to match, since
    "Download everything this run produced" above nothing to click reads as a
    fault. It is found by its sentence, not by the file it owned: an archive over
    ``BIOMNI_THREAD_ELEMENT_MAX_BYTES`` was never stored, which is most real ones,
    and its message has no file to go by. Only assistant messages are touched, so
    nothing a user typed is ever rewritten.
    """
    owners = {step.get("id"): step.get("type") for step in thread.get("steps") or []}
    thread["elements"] = [
        element
        for element in thread.get("elements") or []
        if not is_withheld_download(element.get("type"), owners.get(element.get("forId")))
    ]
    for step in thread.get("steps") or []:
        if step.get("type") == "assistant_message" and isinstance(step.get("output"), str):
            step["output"] = _DOWNLOAD_OFFER.sub(_OFFER_WITHDRAWN, step["output"])
    return thread


# --------------------------------------------------------------------------- #
# What the user is told
# --------------------------------------------------------------------------- #

_AIRLOCK = "To take results out of your workspace, export them through the airlock."


@dataclass(frozen=True)
class AirlockRoot:
    """The folder the airlock exports from, and whether anyone said so."""

    path: str
    # False when it is only the workspace, taken for the airlock's folder because nothing says otherwise.
    declared: bool


def airlock_root(workspace_root: str | None) -> AirlockRoot | None:
    """Where the airlock exports from: ``BIOMNI_AIRLOCK_ROOT``, else the workspace, else nowhere known.

    "Workspace" means two things on GRIP. To this app it is the folder it reads
    data from (``BIOMNI_USER_DATA_PATH``); to the platform it is where the airlock
    works, which a deployment that mounts a dataset and a separate output folder
    makes a different folder. Taking one for the other flagged every run of such a
    deployment as unexportable while its results were being collected without
    trouble, and the warning told the operator to set a variable that was already
    set. Only the operator knows, so the operator can say.
    """
    declared = os.getenv(AIRLOCK_ROOT_ENV, "").strip()
    if declared:
        return AirlockRoot(os.path.abspath(declared), declared=True)
    if workspace_root:
        return AirlockRoot(os.path.abspath(workspace_root), declared=False)
    return None


def _outside_airlock(output_path: str, workspace_root: str | None) -> AirlockRoot | None:
    """The airlock's folder, when downloads are off and ``output_path`` is not inside it.

    ``None`` means nothing to say: downloads are on (they are how results are
    collected), the results are inside, or there is no folder to compare with.
    """
    if not downloads_disabled():
        return None
    root = airlock_root(workspace_root)
    if root is None or is_inside(output_path, root.path):
        return None
    return root


def _stranded(root: AirlockRoot, source: str | None) -> str:
    """What to tell someone whose results are outside the folder the airlock exports from.

    A declared folder makes this a fact, with the way out. That depends on what put
    the results there: a folder chosen in Settings is the user's to change, while a
    default only an operator can, so the advice never asks for a variable to be
    "set" that already is (``source`` is
    :attr:`biomni.workspace_prefs.OutputTarget.source`).

    A workspace taken for the airlock's folder makes it a guess, and says so. Only
    an operator can settle that, with the one setting that does, so there is
    nothing for a user to follow and the message stops there.
    """
    if root.declared:
        tail = "" if source == "preference" else ", or ask an operator to point BIOMNI_OUTPUT_ROOT at one"
        return (
            f"⚠️ Downloads are turned off in this deployment, and the airlock exports only from `{root.path}`, "
            f"so results in this folder cannot be taken out. Choose a folder inside `{root.path}`{tail}."
        )
    return (
        f"⚠️ Downloads are turned off in this deployment, and this folder is outside `{root.path}`, "
        "where Biomni assumes the airlock exports from, so these results may not be exportable. "
        f"If the airlock exports from this folder, an operator can say so with {AIRLOCK_ROOT_ENV}."
    )


def is_inside(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or below it. Pure string work: no filesystem calls.

    This runs on the event loop while the settings dialog is built, on a mount
    where every stat is a network round trip, so symlinks are deliberately not
    resolved; both paths come from configuration spelled the same way.
    """
    path, root = os.path.abspath(path), os.path.abspath(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def unreachable_output_warning(output_path: str, workspace_root: str | None, source: str | None = None) -> str | None:
    """A warning for results that downloads being off would leave nowhere to be collected from.

    The airlock exports from one folder (:func:`airlock_root`). Results written
    elsewhere - the Azure Disk fallback the deployment guide offers while the share
    is not yet writable, or the container's own ``./runs`` - were reachable only
    through the download, so with it off the user could not get them out at all.
    Said where the output folder is chosen, before a long run rather than after it.
    """
    root = _outside_airlock(output_path, workspace_root)
    return None if root is None else _stranded(root, source)


def run_saved_notice(run_dir: str, workspace_root: str | None, source: str | None = None) -> str:
    """The completion message for a run when there is no download to offer.

    Says where the results are and, for a folder the airlock exports from, how to
    get them out; for results outside it, what is known about that. With no folder
    to compare with there is nothing to point at, so it stops at the path.
    """
    saved = f"**Run complete.** Results are saved in:\n`{run_dir}`"
    root = airlock_root(workspace_root)
    if root is None:
        return f"{saved}\n\nDownloads are turned off in this deployment."
    if is_inside(run_dir, root.path):
        return f"{saved}\n\nDownloads are turned off in this deployment. {_AIRLOCK}"
    return f"{saved}\n\n{_stranded(root, source)}"


def airlock_note(output_path: str, workspace_root: str | None) -> str | None:
    """For the startup log: why results written to ``output_path`` may not be able to leave, or ``None``.

    The warnings above reach each user, one chat at a time. An operator reads the
    log, and the same finding belongs there once, with the paths spelled out.
    """
    root = _outside_airlock(output_path, workspace_root)
    if root is None:
        return None
    if root.declared:
        return (
            f"downloads are turned off and {AIRLOCK_ROOT_ENV} says the airlock exports only from {root.path}, "
            f"so results written to {output_path} cannot be taken out; "
            f"point BIOMNI_OUTPUT_ROOT at a folder inside {root.path}"
        )
    return (
        f"downloads are turned off and results are written to {output_path}, outside {root.path}, where the app "
        f"assumes the airlock exports from because {AIRLOCK_ROOT_ENV} is not set; "
        f"if the airlock exports from {output_path} or a folder above it, set {AIRLOCK_ROOT_ENV} to that folder, "
        f"otherwise point BIOMNI_OUTPUT_ROOT at a folder inside {root.path}"
    )
