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
from chainlit_ui import persistence, ui_flags

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _flags_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ui_flags.DISABLE_DOWNLOADS_ENV, ui_flags.HIDE_LOGOUT_ENV):
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


def test_no_warning_while_downloads_are_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The download is how results outside the workspace are collected, so there is nothing to warn about."""
    assert ui_flags.unreachable_output_warning("/app/runs", "/app/user-data") is None


def test_warning_for_results_the_airlock_cannot_see(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    warning = ui_flags.unreachable_output_warning("/app/runs", "/app/user-data")
    assert warning is not None
    assert "outside your workspace" in warning
    assert "BIOMNI_OUTPUT_ROOT" in warning


def test_no_warning_for_results_inside_the_workspace_or_with_no_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    assert ui_flags.unreachable_output_warning("/app/user-data/biomni-outputs", "/app/user-data") is None
    assert ui_flags.unreachable_output_warning("/app/runs", None) is None


def test_the_completion_notice_points_a_workspace_run_at_the_airlock() -> None:
    notice = ui_flags.run_saved_notice("/app/user-data/biomni-outputs/run_1", "/app/user-data")
    assert "`/app/user-data/biomni-outputs/run_1`" in notice
    assert "airlock" in notice
    assert "Downloads are turned off" in notice


def test_the_completion_notice_says_when_results_are_stranded() -> None:
    notice = ui_flags.run_saved_notice("/app/runs/run_1", "/app/user-data")
    assert "`/app/runs/run_1`" in notice
    assert "outside your workspace" in notice
    assert "airlock" not in notice, "do not send them to the airlock for results it cannot reach"


def test_the_completion_notice_without_a_workspace_stops_at_the_path() -> None:
    notice = ui_flags.run_saved_notice("/srv/runs/run_1", None)
    assert "`/srv/runs/run_1`" in notice
    assert "airlock" not in notice


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

    workspace = SimpleNamespace(workspace_root=str(tmp_path / "workspace"))
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
    assert "outside your workspace" in sent[0].content


def test_the_settings_dialog_warns_about_stranded_results_only_with_downloads_off(
    chainlit_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = SimpleNamespace(writable=True, reason=None, is_ephemeral=False, path=str(tmp_path / "runs"))
    workspace = str(tmp_path / "workspace")
    assert chainlit_app._output_warnings(output, workspace) == []

    monkeypatch.setenv(ui_flags.DISABLE_DOWNLOADS_ENV, "true")
    warnings = chainlit_app._output_warnings(output, workspace)
    assert len(warnings) == 1 and "outside your workspace" in warnings[0]
