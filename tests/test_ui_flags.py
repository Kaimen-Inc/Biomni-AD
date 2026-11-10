"""The deployment switches for downloads and logout (chainlit_ui.ui_flags).

On the AD Workbench the airlock is the only way results leave a workspace, so
with ``BIOMNI_DISABLE_DOWNLOADS`` the app must offer no file of its own - not
in the run that just finished, and not in a conversation saved before the switch
was turned on, whose stored run archive would otherwise still be a link. These
tests pin both, and what the user is told when there is nothing to download.

The pure parts run anywhere. The data layer and the completion message need
Chainlit and skip without it, as CI installs no chainlit extra.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from biomni.workspace_prefs import OutputTarget
from chainlit_ui import persistence, ui_flags

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _flags_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ui_flags.DISABLE_DOWNLOADS_ENV, ui_flags.HIDE_LOGOUT_ENV, ui_flags.AIRLOCK_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# The switches
# --------------------------------------------------------------------------- #


def test_both_switches_default_to_off() -> None:
    assert ui_flags.downloads_disabled() is False
    assert ui_flags.logout_hidden() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_switches_turn_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, value)
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, value)
    assert ui_flags.downloads_disabled() is True
    assert ui_flags.logout_hidden() is True


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "FALSE", "no", "off"])
def test_empty_and_false_values_leave_them_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A set-but-empty ConfigMap key must not read as "on"."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, value)
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, value)
    assert ui_flags.downloads_disabled() is False
    assert ui_flags.logout_hidden() is False


def test_an_unrecognised_logout_value_leaves_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hiding a menu item is cosmetic, so the shared rule applies: what cannot be read is ignored."""
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, "maybe")
    assert ui_flags.logout_hidden() is False


@pytest.fixture
def _fresh_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui_flags, "_reported_values", set())


@pytest.mark.parametrize("value", ["ture", "enabled", "y", "2", "maybe"])
def test_an_unrecognised_downloads_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch, _fresh_reports: None, value: str
) -> None:
    """A typo must not leave downloads open on a workspace that may not have them."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, value)
    assert ui_flags.downloads_disabled() is True


def test_the_unrecognised_value_is_reported_once_and_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, _fresh_reports: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "ture")
    with caplog.at_level("ERROR", logger=ui_flags.logger.name):
        for _ in range(5):  # every conversation read and every run asks
            assert ui_flags.downloads_disabled() is True
    (record,) = caplog.records
    message = record.getMessage()
    assert ui_flags.DISABLE_DOWNLOADS_ENV in message
    assert "'ture'" in message
    assert "OFF" in message and "true or false" in message


# --------------------------------------------------------------------------- #
# Logout: the stylesheet
# --------------------------------------------------------------------------- #


def test_the_configured_stylesheet_is_kept_while_logout_shows() -> None:
    assert ui_flags.custom_stylesheet(ui_flags.DEFAULT_STYLESHEET) == ui_flags.DEFAULT_STYLESHEET
    assert ui_flags.custom_stylesheet(None) is None


def test_hiding_logout_swaps_in_the_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, "true")
    assert ui_flags.custom_stylesheet(ui_flags.DEFAULT_STYLESHEET) == ui_flags.HIDE_LOGOUT_STYLESHEET


def test_replacing_a_stylesheet_of_the_operators_own_is_said_at_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, "true")
    with caplog.at_level("WARNING", logger=ui_flags.logger.name):
        assert ui_flags.custom_stylesheet("/public/our-brand.css") == ui_flags.HIDE_LOGOUT_STYLESHEET
    (record,) = caplog.records
    message = record.getMessage()
    assert "/public/our-brand.css" in message, "name what was replaced"
    assert '@import url("biomni-hide-logout.css")' in message, "and how to keep it"


@pytest.mark.parametrize("configured", [ui_flags.DEFAULT_STYLESHEET, ui_flags.HIDE_LOGOUT_STYLESHEET, None, ""])
def test_the_stylesheets_this_app_ships_are_swapped_without_comment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, configured: str | None
) -> None:
    monkeypatch.setenv(ui_flags.HIDE_LOGOUT_ENV, "true")
    with caplog.at_level("WARNING", logger=ui_flags.logger.name):
        assert ui_flags.custom_stylesheet(configured) == ui_flags.HIDE_LOGOUT_STYLESHEET
    assert caplog.records == []


def test_the_swap_replaces_the_stylesheet_the_config_names() -> None:
    """The variant imports biomni.css, so it only stands in for that one.

    If the config pointed somewhere else, hiding logout would silently drop that
    stylesheet.
    """
    config = tomllib.loads((REPO / ".chainlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["UI"]["custom_css"] == ui_flags.DEFAULT_STYLESHEET


def test_the_variant_stylesheet_imports_the_base_and_hides_the_item() -> None:
    css = (REPO / "public" / "biomni-hide-logout.css").read_text(encoding="utf-8")
    # Strip comments so the checks see the rules, not the prose about them.
    body = "".join(part.split("/*")[0] for part in css.split("*/"))
    statements = [s.strip() for s in body.split(";") if s.strip()]
    assert statements[0] == '@import url("biomni.css")', "the import must come first or the browser ignores it"
    assert (REPO / "public" / "biomni.css").is_file(), "the relative import has to resolve beside it"
    assert '[role="menuitem"]:has(.lucide-log-out)' in body
    assert "display: none" in body


# --------------------------------------------------------------------------- #
# What is withheld from a stored conversation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("element_type", "owner", "withheld"),
    [
        ("file", "assistant_message", True),  # the run archive
        ("pdf", "assistant_message", True),  # a viewer with its own download button
        ("file", "tool", True),
        ("file", None, True),  # owner unknown: fail closed
        ("file", "user_message", False),  # the user's own attachment is theirs
        ("pdf", "user_message", False),
        ("image", "assistant_message", False),  # a plot stays visible
        ("text", "assistant_message", False),
        (None, "assistant_message", False),
    ],
)
def test_which_elements_are_withheld(element_type: str | None, owner: str | None, withheld: bool) -> None:
    assert ui_flags.is_withheld_download(element_type, owner) is withheld


# What a run's completion message said while downloads were on.
OFFER = "**Run complete.** Download everything this run produced (4 files, 0.1 MB), or find it on the server at `/app/runs/run_1`."
WITHDRAWN = "**Run complete.** Results are saved in `/app/runs/run_1`. Downloads are turned off in this deployment."


def _thread() -> dict:
    return {
        "id": "t1",
        "name": "demo",
        "steps": [
            {"id": "u1", "type": "user_message", "output": "plot it"},
            {"id": "a1", "type": "assistant_message", "output": OFFER},
            # A run whose archive was over BIOMNI_THREAD_ELEMENT_MAX_BYTES: the message was stored, the file never was.
            {"id": "a2", "type": "assistant_message", "output": OFFER},
            # Someone pasting the sentence into the chat: theirs, so never rewritten.
            {"id": "u2", "type": "user_message", "output": OFFER},
        ],
        "elements": [
            {"id": "zip", "type": "file", "forId": "a1", "name": "run.zip"},
            {"id": "png", "type": "image", "forId": "a1", "name": "plot.png"},
            {"id": "pdf", "type": "pdf", "forId": "a1", "name": "report.pdf"},
            {"id": "mine", "type": "file", "forId": "u1", "name": "input.csv"},
            {"id": "lost", "type": "file", "forId": "gone", "name": "orphan.bin"},
        ],
    }


def test_withholding_keeps_plots_and_the_users_own_files() -> None:
    thread = ui_flags.withhold_downloads(_thread())
    assert [e["id"] for e in thread["elements"]] == ["png", "mine"]


def test_the_message_that_carried_the_archive_no_longer_offers_it() -> None:
    thread = ui_flags.withhold_downloads(_thread())
    outputs = {step["id"]: step["output"] for step in thread["steps"]}
    assert outputs["a1"] == WITHDRAWN


def test_a_run_whose_archive_was_too_big_to_be_stored_is_reworded_too() -> None:
    """Most real archives are over the inline limit, so their message is stored with no file to go by."""
    thread = ui_flags.withhold_downloads(_thread())
    outputs = {step["id"]: step["output"] for step in thread["steps"]}
    assert outputs["a2"] == WITHDRAWN


def test_nothing_a_user_typed_is_rewritten() -> None:
    thread = ui_flags.withhold_downloads(_thread())
    outputs = {step["id"]: step["output"] for step in thread["steps"]}
    assert outputs["u1"] == "plot it"
    assert outputs["u2"] == OFFER


def test_withholding_touches_nothing_else_and_works_in_place() -> None:
    original = _thread()
    returned = ui_flags.withhold_downloads(original)
    assert returned is original
    assert [(s["id"], s["type"]) for s in original["steps"]] == [(s["id"], s["type"]) for s in _thread()["steps"]]
    assert original["name"] == "demo"


def test_a_message_with_no_recognisable_offer_is_left_alone() -> None:
    """A conversation from some other version of the message, or with no output at all, is not an error."""
    thread = _thread()
    thread["steps"][1]["output"] = "**Run complete.** Saved to `/app/runs/run_1`."
    thread["steps"][2]["output"] = None
    ui_flags.withhold_downloads(thread)
    assert thread["steps"][1]["output"] == "**Run complete.** Saved to `/app/runs/run_1`."
    assert thread["steps"][2]["output"] is None


@pytest.mark.parametrize("thread", [{}, {"steps": None, "elements": None}, {"elements": []}])
def test_withholding_tolerates_a_thread_with_nothing_in_it(thread: dict) -> None:
    assert ui_flags.withhold_downloads(thread)["elements"] == []


# --------------------------------------------------------------------------- #
# Where results are, and what the user is told
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "root", "inside"),
    [
        ("/app/user-data", "/app/user-data", True),
        ("/app/user-data/biomni-outputs/run_1", "/app/user-data", True),
        ("/app/user-data/", "/app/user-data", True),
        ("/app/runs/run_1", "/app/user-data", False),
        ("/app/user-data-old/run_1", "/app/user-data", False),  # a shared prefix is not containment
        ("/app/user-data/../runs/run_1", "/app/user-data", False),  # nor is a path that climbs out
        ("/app", "/app/user-data", False),
    ],
)
def test_is_inside(path: str, root: str, inside: bool) -> None:
    assert ui_flags.is_inside(path, root) is inside


# --------------------------------------------------------------------------- #
# Where the airlock exports from, and what is said about results outside it
# --------------------------------------------------------------------------- #

# The layout a deployment can be given since the platform provisions an input folder
# and an output folder: the app reads a dataset from one and writes results to the
# other, and the airlock exports from the second. "Your workspace" is then the output
# folder to the platform and the dataset folder to this app.
DATASET = "/app/user-data"
RESULTS = "/workspace/biomni-outputs"
EXPORTS = "/workspace"


def test_the_airlock_is_taken_to_export_from_the_workspace_unless_told_otherwise() -> None:
    assert ui_flags.airlock_root("/app/user-data/") == ui_flags.AirlockRoot("/app/user-data", declared=False)
    assert ui_flags.airlock_root(None) is None


def test_the_operator_can_say_where_the_airlock_exports_from(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, f"{EXPORTS}/")
    assert ui_flags.airlock_root(DATASET) == ui_flags.AirlockRoot(EXPORTS, declared=True)
    assert ui_flags.airlock_root(None) == ui_flags.AirlockRoot(EXPORTS, declared=True), "no workspace is needed"


def test_a_blank_airlock_root_declares_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set-but-empty ConfigMap key must not read as a folder."""
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, "  ")
    assert ui_flags.airlock_root(DATASET) == ui_flags.AirlockRoot(DATASET, declared=False)


def test_no_warning_while_downloads_are_on() -> None:
    """The download is how results outside the airlock's folder are collected, so there is nothing to warn about."""
    assert ui_flags.unreachable_output_warning("/app/runs", DATASET) is None
    assert ui_flags.airlock_note("/app/runs", DATASET) is None


def test_no_warning_for_results_inside_the_workspace_or_with_nothing_to_compare_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    assert ui_flags.unreachable_output_warning(f"{DATASET}/biomni-outputs", DATASET) is None
    assert ui_flags.unreachable_output_warning("/app/runs", None) is None
    assert ui_flags.airlock_note(f"{DATASET}/biomni-outputs", DATASET) is None
    assert ui_flags.airlock_note("/app/runs", None) is None


def test_a_separate_output_folder_is_not_flagged_once_the_airlock_folder_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report that started this: every run of a working deployment ended in a warning."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, EXPORTS)

    assert ui_flags.unreachable_output_warning(RESULTS, DATASET, "env") is None
    assert ui_flags.airlock_note(RESULTS, DATASET) is None
    notice = ui_flags.run_saved_notice(f"{RESULTS}/run_1", DATASET, "env")
    assert f"`{RESULTS}/run_1`" in notice
    assert "export them through the airlock" in notice
    assert "your workspace" not in notice, "the folder is named by its path: 'workspace' means two things here"
    assert "⚠️" not in notice


def test_without_the_declaration_the_warning_says_what_it_assumed_and_how_to_settle_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What that deployment saw before: a verdict about a folder that worked, and a request to set a variable that was set.

    The app is guessing where the airlock reads, so the warning has to read as a
    guess, and name the folder it compared with: "your workspace" was the output
    folder to the people reading it, and the dataset folder to the app.
    """
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")

    warning = ui_flags.unreachable_output_warning(RESULTS, DATASET, "env")

    assert warning is not None
    assert f"`{DATASET}`" in warning
    assert "may not be exportable" in warning
    assert ui_flags.AIRLOCK_ROOT_ENV in warning
    for verdict in ("no way", "outside your workspace", "set BIOMNI_OUTPUT_ROOT"):
        assert verdict not in warning


def test_a_declared_airlock_folder_makes_the_warning_a_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, EXPORTS)

    warning = ui_flags.unreachable_output_warning("/app/runs", DATASET, "env")

    assert warning is not None
    assert f"exports only from `{EXPORTS}`" in warning
    assert "cannot be taken out" in warning
    assert "may not" not in warning
    assert ui_flags.AIRLOCK_ROOT_ENV not in warning, "it is declared; there is nothing left to say"


def test_the_advice_follows_what_put_the_results_there(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never ask for a variable to be set because of a folder that variable did not choose, or that it did."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, EXPORTS)

    chosen = ui_flags.unreachable_output_warning("/app/runs", DATASET, "preference")
    assert chosen is not None
    assert chosen.endswith(f"Choose a folder inside `{EXPORTS}`.")
    assert "BIOMNI_OUTPUT_ROOT" not in chosen, "a folder picked in Settings is the user's to change, not an operator's"

    for source in ("env", "workspace", "cwd-fallback", None):
        warning = ui_flags.unreachable_output_warning("/app/runs", DATASET, source)
        assert warning is not None
        assert warning.endswith(
            f"Choose a folder inside `{EXPORTS}`, or ask an operator to point BIOMNI_OUTPUT_ROOT at one."
        )
        assert "set BIOMNI_OUTPUT_ROOT" not in warning


@pytest.mark.parametrize("source", ["preference", "env", "workspace", "cwd-fallback", None])
def test_a_guess_is_the_same_whatever_chose_the_folder(monkeypatch: pytest.MonkeyPatch, source: str | None) -> None:
    """Only an operator can settle a guess, so there is no folder for a user to choose, and nothing to vary."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    warning = ui_flags.unreachable_output_warning("/app/runs", DATASET, source)
    assert warning == ui_flags.unreachable_output_warning("/app/runs", DATASET)
    assert "choose" not in warning.lower()


def test_the_completion_notice_points_a_run_inside_the_airlock_folder_at_the_airlock() -> None:
    notice = ui_flags.run_saved_notice(f"{DATASET}/biomni-outputs/run_1", DATASET)
    assert f"`{DATASET}/biomni-outputs/run_1`" in notice
    assert "export them through the airlock" in notice
    assert "your workspace" not in notice
    assert "Downloads are turned off" in notice


def test_the_completion_notice_says_when_results_are_stranded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, DATASET)
    notice = ui_flags.run_saved_notice("/app/runs/run_1", DATASET, "env")
    assert "`/app/runs/run_1`" in notice
    assert "cannot be taken out" in notice
    assert "export them through the airlock" not in notice, (
        "do not send them to the airlock for results it cannot reach"
    )


def test_the_completion_notice_for_a_guess_does_not_send_them_to_the_airlock_either() -> None:
    notice = ui_flags.run_saved_notice("/app/runs/run_1", DATASET, "env")
    assert "`/app/runs/run_1`" in notice
    assert "may not be exportable" in notice
    assert "export them through the airlock" not in notice


def test_the_completion_notice_without_a_folder_to_compare_with_stops_at_the_path() -> None:
    notice = ui_flags.run_saved_notice("/srv/runs/run_1", None)
    assert "`/srv/runs/run_1`" in notice
    assert "airlock" not in notice


def test_a_declared_airlock_folder_needs_no_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, EXPORTS)
    assert "export them through the airlock" in ui_flags.run_saved_notice(f"{RESULTS}/run_1", None)
    assert "cannot be taken out" in ui_flags.run_saved_notice("/srv/runs/run_1", None)


def test_the_startup_note_gives_an_operator_both_folders_and_the_way_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logged once at startup, where an operator looks, with the paths spelled out."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")

    guess = ui_flags.airlock_note(RESULTS, DATASET)
    assert guess is not None
    for fact in (RESULTS, DATASET, ui_flags.AIRLOCK_ROOT_ENV, "BIOMNI_OUTPUT_ROOT"):
        assert fact in guess

    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, EXPORTS)
    fact = ui_flags.airlock_note("/app/runs", DATASET)
    assert fact is not None
    for part in ("/app/runs", EXPORTS, "cannot be taken out", "BIOMNI_OUTPUT_ROOT"):
        assert part in fact


# --------------------------------------------------------------------------- #
# The data layer, against a real SQLite database
# --------------------------------------------------------------------------- #

_STEPS = (("u1", "user_message"), ("a1", "assistant_message"))
_ELEMENTS = (
    ("e_zip", "file", "a1", "run.zip", "application/zip"),
    ("e_png", "image", "a1", "plot.png", "image/png"),
    ("e_pdf", "pdf", "a1", "report.pdf", "application/pdf"),
    ("e_mine", "file", "u1", "input.csv", "text/csv"),
)


@pytest.fixture
def layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The app's real data layer over a database holding one conversation.

    The conversation is what a deployment looks like the day the switch is turned
    on: the user's question with an attachment, and an answer carrying a plot, a
    report and the run archive the old completion message offered.
    """
    pytest.importorskip("chainlit", reason="chainlit not installed")
    pytest.importorskip("aiosqlite", reason="aiosqlite not installed")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("BIOMNI_STATE_DIR", str(state))
    data_layer = persistence.build_data_layer(str(tmp_path / "workspace"))
    assert data_layer is not None

    db = persistence.sqlite_path_from_url(persistence.resolve_database_url(str(tmp_path / "workspace")))
    con = sqlite3.connect(db)
    con.execute('INSERT INTO threads ("id", "name", "userId") VALUES (?, ?, ?)', ("t1", "demo", "user-1"))
    for step_id, step_type in _STEPS:
        con.execute(
            'INSERT INTO steps ("id", "name", "type", "threadId", "streaming", "createdAt", "output") '
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (step_id, "step", step_type, "t1", False, "2026-10-28T10:00:00Z", OFFER if step_id == "a1" else "hello"),
        )
    for element_id, element_type, owner, name, mime in _ELEMENTS:
        con.execute(
            'INSERT INTO elements ("id", "threadId", "type", "url", "name", "display", "forId", "mime", "props") '
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (element_id, "t1", element_type, f"data:{mime};base64,AA==", name, "inline", owner, mime, "{}"),
        )
    con.commit()
    con.close()
    yield data_layer
    # Closed here rather than left to the garbage collector: a failing test's traceback keeps the
    # layer alive, and its pooled connection's worker thread then stops the interpreter from exiting,
    # which turns one red test into a job that hangs.
    asyncio.run(data_layer.engine.dispose())


def test_a_reopened_conversation_keeps_every_file_by_default(layer) -> None:
    async def scenario() -> tuple[set[str], dict[str, str]]:
        thread = await layer.get_thread("t1")
        return {e["id"] for e in thread["elements"]}, {s["id"]: s["output"] for s in thread["steps"]}

    elements, outputs = asyncio.run(scenario())
    assert elements == {"e_zip", "e_png", "e_pdf", "e_mine"}
    assert outputs["a1"] == OFFER, "with downloads on the old message is exactly as it was"


def test_a_reopened_conversation_hands_out_no_file_with_downloads_off(layer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")

    async def scenario() -> tuple[set[str], dict[str, str]]:
        thread = await layer.get_thread("t1")
        return {e["id"] for e in thread["elements"]}, {s["id"]: s["output"] for s in thread["steps"]}

    elements, outputs = asyncio.run(scenario())
    assert elements == {"e_png", "e_mine"}, "the plot and the user's own attachment stay; the archive and report go"
    assert sorted(outputs) == ["a1", "u1"], "the transcript itself is untouched"
    assert outputs["a1"] == WITHDRAWN, "and the message no longer promises the archive"
    assert outputs["u1"] == "hello"


def _listed_threads(layer) -> list[dict]:
    """What ``POST /project/threads`` returns: the conversation list the sidebar asks for."""
    from chainlit.types import Pagination, ThreadFilter

    async def scenario() -> list[dict]:
        page = await layer.list_threads(Pagination(first=20), ThreadFilter(userId="user-1"))
        return list(page.data)

    return asyncio.run(scenario())


def test_the_conversation_list_keeps_every_file_by_default(layer) -> None:
    (thread,) = _listed_threads(layer)
    assert {e["id"] for e in thread["elements"]} == {"e_zip", "e_png", "e_pdf", "e_mine"}


def test_the_conversation_list_hands_out_no_file_with_downloads_off(layer, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sidebar's list is built from the same full thread dicts, so it carried every old archive as a data: URL."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    (thread,) = _listed_threads(layer)
    assert {e["id"] for e in thread["elements"]} == {"e_png", "e_mine"}
    assert {s["id"]: s["output"] for s in thread["steps"]}["a1"] == WITHDRAWN


def test_a_missing_conversation_is_still_missing(layer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")

    async def scenario():
        return await layer.get_thread("nope")

    assert not asyncio.run(scenario())


def test_the_element_api_is_unchanged_by_default(layer) -> None:
    async def scenario() -> str:
        return (await layer.get_element("t1", "e_zip"))["name"]

    assert asyncio.run(scenario()) == "run.zip"


def test_the_element_api_hands_out_no_file_with_downloads_off(layer, monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /project/thread/<id>/element/<id>`` returns the element with its data: URL, so it needs the same rule."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")

    async def scenario() -> dict:
        return {
            element_id: await layer.get_element("t1", element_id)
            for element_id in ("e_zip", "e_pdf", "e_png", "e_mine", "no_such_element")
        }

    got = asyncio.run(scenario())
    assert got["e_zip"] is None
    assert got["e_pdf"] is None
    assert got["e_png"]["name"] == "plot.png"
    assert got["e_mine"]["name"] == "input.csv"
    assert got["no_such_element"] is None


# --------------------------------------------------------------------------- #
# The completion message, in the real app module
# --------------------------------------------------------------------------- #


@pytest.fixture
def chainlit_app():
    # Imported in a fixture, not at the top: the module exits the interpreter
    # when its dependencies are missing, which pytest cannot survive during
    # collection (see test_run_packaging.py).
    return pytest.importorskip("chainlit_app")


@pytest.fixture
def sent(chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list:
    """Chainlit's Message and File without a session behind them, recording what is sent."""
    messages: list = []

    class FakeMessage:
        def __init__(self, content: str = "", elements: list | None = None) -> None:
            self.content = content
            self.elements = elements or []

        async def send(self) -> None:
            messages.append(self)

    class FakeFile:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    workspace = SimpleNamespace(workspace_root=str(tmp_path / "workspace"), output=SimpleNamespace(source="env"))
    monkeypatch.setattr(chainlit_app.cl, "Message", FakeMessage)
    monkeypatch.setattr(chainlit_app.cl, "File", FakeFile)
    monkeypatch.setattr(
        chainlit_app.cl,
        "user_session",
        SimpleNamespace(get=lambda key, default=None: workspace if key == "workspace" else default),
    )
    return messages


def _finished_run(tmp_path: Path, *, under: str) -> Path:
    run = tmp_path / under / "run_20261028_abc123_demo"
    run.mkdir(parents=True)
    (run / "report.md").write_text("# findings")
    (run / "results.csv").write_text("gene,logfc\nAPOE,2.4\n")
    return run


def _output_target(path: Path, *, source: str, reason: str | None = None) -> OutputTarget:
    """A resolved output location, without resolving one: what the Settings dialog is built from."""
    return OutputTarget(path=str(path), source=source, writable=True, reason=reason)


def test_a_finished_run_is_offered_as_a_download_by_default(chainlit_app, sent: list, tmp_path: Path) -> None:
    run = _finished_run(tmp_path, under="workspace/biomni-outputs")
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))

    assert len(sent) == 1
    assert [f.kwargs["name"] for f in sent[0].elements] == [f"{run.name}.zip"]
    assert (run.parent / ".packages" / f"{run.name}.zip").is_file()


def test_the_offer_it_makes_is_the_one_a_later_switch_can_withdraw(chainlit_app, sent: list, tmp_path: Path) -> None:
    """The reword in withhold_downloads matches the text written here, or the old message keeps promising an archive."""
    run = _finished_run(tmp_path, under="workspace/biomni-outputs")
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))
    thread = {
        "steps": [{"id": "a1", "type": "assistant_message", "output": sent[0].content}],
        "elements": [{"id": "zip", "type": "file", "forId": "a1", "name": f"{run.name}.zip"}],
    }

    ui_flags.withhold_downloads(thread)

    output = thread["steps"][0]["output"]
    assert "Download everything" not in output
    assert f"`{run}`" in output
    assert "Downloads are turned off" in output


def test_with_downloads_off_a_finished_run_attaches_nothing_and_builds_no_archive(
    chainlit_app, sent: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    run = _finished_run(tmp_path, under="workspace/biomni-outputs")
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))

    assert len(sent) == 1
    assert sent[0].elements == []
    assert str(run) in sent[0].content
    assert "airlock" in sent[0].content
    assert not (run.parent / ".packages").exists(), "no archive should even be built"


def test_with_downloads_off_results_outside_the_workspace_are_flagged(
    chainlit_app, sent: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    run = _finished_run(tmp_path, under="runs")  # not under tmp_path / "workspace"
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))

    assert sent[0].elements == []
    assert "may not be exportable" in sent[0].content
    assert str(tmp_path / "workspace") in sent[0].content, "say which folder it compared with"


def test_a_run_in_a_separate_output_folder_the_airlock_reads_finishes_without_a_warning(
    chainlit_app, sent: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset is the workspace the app reads; the platform's workspace, where the airlock exports from, is another folder."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, str(tmp_path / "platform"))
    run = _finished_run(tmp_path, under="platform/biomni-outputs")
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))

    assert sent[0].elements == []
    assert str(run) in sent[0].content
    assert "export them through the airlock" in sent[0].content
    assert "⚠️" not in sent[0].content


def test_the_settings_dialog_warns_about_stranded_results_only_with_downloads_off(
    chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _output_target(tmp_path / "runs", source="env")
    workspace = str(tmp_path / "workspace")
    assert chainlit_app._output_warnings(output, workspace) == []

    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    warnings = chainlit_app._output_warnings(output, workspace)
    assert len(warnings) == 1 and "may not be exportable" in warnings[0]


def test_the_messages_know_what_chose_the_folder(
    chainlit_app, sent: list, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder the user picked in Settings is theirs to change: no operator, no BIOMNI_OUTPUT_ROOT, in either message."""
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, str(tmp_path / "platform"))
    (warning,) = chainlit_app._output_warnings(
        _output_target(tmp_path / "runs", source="preference"), str(tmp_path / "workspace")
    )
    assert "BIOMNI_OUTPUT_ROOT" not in warning

    chainlit_app.cl.user_session.get("workspace").output.source = "preference"
    run = _finished_run(tmp_path, under="runs")
    asyncio.run(chainlit_app._offer_run_results(str(run), run.name))
    assert "BIOMNI_OUTPUT_ROOT" not in sent[0].content


def test_the_settings_dialog_is_quiet_about_a_separate_output_folder_the_airlock_reads(
    chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, str(tmp_path / "platform"))
    output = _output_target(tmp_path / "platform" / "biomni-outputs", source="env")
    assert chainlit_app._output_warnings(output, str(tmp_path / "workspace")) == []


@pytest.fixture
def layout(chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A dataset folder as the workspace the app reads, and a separate output folder the results go to."""
    for name in ("BIOMNI_USER_DATA_HOST_PATH", "BIOMNI_DATA_PATH", "BIOMNI_PATH", "BIOMNI_STATE_DIR"):
        monkeypatch.delenv(name, raising=False)
    dataset, platform = tmp_path / "dataset", tmp_path / "platform"
    (platform / "biomni-outputs").mkdir(parents=True)
    dataset.mkdir()
    monkeypatch.setenv("BIOMNI_USER_DATA_PATH", str(dataset))
    monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(platform / "biomni-outputs"))
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    return SimpleNamespace(dataset=dataset, platform=platform, results=platform / "biomni-outputs")


def _logged(caplog: pytest.LogCaptureFixture, chainlit_app, event: str, **fields):
    """The one structured line a startup check emitted, as a log record carrying its fields."""
    check = chainlit_app._log_storage_check if event == "storage_check" else chainlit_app._apply_ui_flags
    with caplog.at_level("INFO", logger=chainlit_app.logger.name):
        check()
    (line,) = [
        r
        for r in caplog.records
        if getattr(r, "event", None) == event and all(getattr(r, k, None) == v for k, v in fields.items())
    ]
    return line


def test_the_startup_log_warns_an_operator_that_the_airlock_may_not_reach_the_results(
    chainlit_app, layout: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """The log said nothing at INFO while every user was told. This is where an operator looks."""
    line = _logged(caplog, chainlit_app, "storage_check", location="outputs")
    assert line.levelname == "WARNING"
    for fact in (layout.results, layout.dataset, ui_flags.AIRLOCK_ROOT_ENV):
        assert str(fact) in line.airlock_note


def test_the_startup_log_is_quiet_once_the_airlock_folder_is_declared(
    chainlit_app, layout: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, str(layout.platform))
    line = _logged(caplog, chainlit_app, "storage_check", location="outputs")
    assert line.levelname == "INFO"
    assert line.airlock_note is None


def test_the_startup_log_says_which_folder_the_airlock_is_taken_to_export_from(
    chainlit_app, layout: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    guessed = _logged(caplog, chainlit_app, "ui_flags")
    assert (guessed.airlock_root, guessed.airlock_root_declared) == (str(layout.dataset), False)

    caplog.clear()
    monkeypatch.setenv(ui_flags.AIRLOCK_ROOT_ENV, str(layout.platform))
    declared = _logged(caplog, chainlit_app, "ui_flags")
    assert (declared.airlock_root, declared.airlock_root_declared) == (str(layout.platform), True)

    caplog.clear()
    monkeypatch.delenv(ui_flags.DISABLE_DOWNLOADS_ENV)
    off = _logged(caplog, chainlit_app, "ui_flags")
    assert (off.airlock_root, off.airlock_root_declared) == (None, None), "no airlock is in play while downloads are on"


def test_the_container_local_warning_does_not_ask_for_a_variable_that_is_already_set(
    chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable BIOMNI_OUTPUT_ROOT is passed over, which leaves it set and not in use: fix it, do not set it."""
    unwritable = "BIOMNI_OUTPUT_ROOT (/data/outputs) is not writable by uid 57439: owned by root"
    output = _output_target(tmp_path / "runs", source="cwd-fallback", reason=unwritable)
    assert output.is_ephemeral

    def container_local() -> str:
        (warning,) = [w for w in chainlit_app._output_warnings(output) if "container-local" in w]
        return warning

    monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", "/data/outputs")
    assert "fix BIOMNI_OUTPUT_ROOT, as described above" in container_local()
    assert "set BIOMNI_OUTPUT_ROOT" not in container_local()

    monkeypatch.delenv("BIOMNI_OUTPUT_ROOT")
    assert "ask an operator to set BIOMNI_OUTPUT_ROOT" in container_local()
