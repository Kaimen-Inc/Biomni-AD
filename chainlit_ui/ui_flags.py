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
rely on the download. Nothing here imports Chainlit: CI installs no chainlit
extra and must still be able to test the decisions made in this module.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from biomni.workspace_prefs import FALSE_VALUES, TRUE_VALUES, env_flag

logger = logging.getLogger(__name__)

DISABLE_DOWNLOADS_ENV = "BIOMNI_DISABLE_DOWNLOADS"
HIDE_LOGOUT_ENV = "BIOMNI_HIDE_LOGOUT"

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

_OUTSIDE_WORKSPACE = (
    "⚠️ Downloads are turned off in this deployment, and this folder is outside your workspace, so "
    "there is no way to take these results out. Choose a folder inside your workspace, or ask an "
    "operator to set BIOMNI_OUTPUT_ROOT to one."
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


def unreachable_output_warning(output_path: str, workspace_root: str | None) -> str | None:
    """A warning for results that downloads being off would leave nowhere to be collected from.

    The airlock works on the workspace. Results written elsewhere - the Azure
    Disk fallback the deployment guide offers while the share is not yet
    writable, or the container's own ``./runs`` - were reachable only through the
    download, so with it off the user could not get them out at all. Said where
    the output folder is chosen, before a long run rather than after it.
    """
    if not downloads_disabled() or not workspace_root or is_inside(output_path, workspace_root):
        return None
    return _OUTSIDE_WORKSPACE


def run_saved_notice(run_dir: str, workspace_root: str | None) -> str:
    """The completion message for a run when there is no download to offer.

    Says where the results are and, for a workspace the airlock can see, how to
    get them out; for results it cannot see, that they are stranded. With no
    workspace at all there is nothing to point at, so it stops at the path.
    """
    saved = f"**Run complete.** Results are saved in:\n`{run_dir}`"
    if workspace_root is None:
        return f"{saved}\n\nDownloads are turned off in this deployment."
    if is_inside(run_dir, workspace_root):
        return f"{saved}\n\nDownloads are turned off in this deployment. {_AIRLOCK}"
    return f"{saved}\n\n{_OUTSIDE_WORKSPACE}"
