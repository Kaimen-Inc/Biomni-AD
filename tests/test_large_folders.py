"""A large folder must never make the page report that the server is unreachable.

Chainlit's browser client pings the server and gives up when the pings go
unanswered: after roughly 45 s without a reply it drops the websocket, and after
the 20 s reconnect timeout shows "Could not reach the server" until the server
answers again. Anything that holds the asyncio event loop for about a minute
therefore produces that message, with the app perfectly healthy behind it.

Two things held the loop on a workspace with large folders, both reading files on
the loop itself: the walk of the whole workspace that precedes every run, and the
settings dialog counting the selected folders again once the scan cache had
expired. The tests here pin the repair from three sides: the loop stays free
while the work runs, the user is told what is going on if it takes a while, and
no scan can slip back onto a coroutine unnoticed.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from biomni import fs_scan
from biomni.artifact import SnapshotCancelled
from biomni.run_registry import RunRegistry
from biomni.workspace_prefs import NullPrefsStore, WorkspacePrefs, resolve_output_dir, resolve_scope

APP_SOURCE = Path(__file__).resolve().parents[1] / "chainlit_app.py"


# --------------------------------------------------------------------------- #
# Nothing that scans a workspace is called from a coroutine (source only: runs without chainlit)
# --------------------------------------------------------------------------- #

# These read directory trees, and a tree on a network mount can take minutes.
_SCANS = {
    "get_all_files",
    "snapshot_files",
    "scan_directory",
    "summarize_scope",
    "build_scope_inventory",
    "list_top_level_dirs",
}
# The one function allowed to call them. Everything above it in the call chain
# is blocking too, so the entry points must only ever be handed to an executor.
_SCANNER = "_survey_workspace"
_BLOCKING_ENTRY_POINTS = {
    "_survey_workspace",
    "_refresh_workspace_view",
    "_build_workspace_session",
    "_apply_workspace_settings",
}


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_made_by(function: ast.AST):
    """Calls in a function's own body, not in functions or lambdas defined inside it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def test_only_the_workspace_survey_scans_directory_trees() -> None:
    """The settings dialog used to count the selected folders itself, on the event loop."""
    tree = ast.parse(APP_SOURCE.read_text())
    offenders = [
        f"{function.name}() line {call.lineno} calls {name}()"
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef) and function.name != _SCANNER
        for call in _calls_made_by(function)
        if (name := _called_name(call)) in _SCANS
    ]
    assert not offenders, (
        "chainlit_app.py scans a directory tree outside _survey_workspace:\n  "
        + "\n  ".join(offenders)
        + "\nA scan on the event loop freezes every connection until it ends. Read the result "
        "from WorkspaceSession (scope_summaries, inventory), or compute it in _survey_workspace, "
        "which only ever runs in an executor."
    )


def test_the_functions_that_scan_are_never_called_from_a_coroutine() -> None:
    """They are handed to run_in_executor as a reference; calling one is how a scan reaches the loop."""
    tree = ast.parse(APP_SOURCE.read_text())
    offenders = [
        f"{function.name}() line {call.lineno} calls {name}()"
        for function in ast.walk(tree)
        if isinstance(function, ast.AsyncFunctionDef)
        for call in _calls_made_by(function)
        if (name := _called_name(call)) in _BLOCKING_ENTRY_POINTS | _SCANS
    ]
    assert not offenders, (
        "a coroutine in chainlit_app.py calls a function that scans the workspace:\n  "
        + "\n  ".join(offenders)
        + "\nPass it to run_in_executor instead of calling it."
    )


# --------------------------------------------------------------------------- #
# The app, with Chainlit's Message and ChatSettings but no session behind them
# --------------------------------------------------------------------------- #


@pytest.fixture
def chainlit_app():
    # Imported in a fixture, not at the top: the module exits the interpreter
    # when its dependencies are missing, which pytest cannot survive during
    # collection (see test_run_packaging.py).
    return pytest.importorskip("chainlit_app")


class _Chat:
    """What reached the browser, in order."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.fail_to_send = False

    def kinds(self) -> list[str]:
        return [event[0] for event in self.events]

    def contents(self, kind: str) -> list[str]:
        return [event[1] for event in self.events if event[0] == kind]


@pytest.fixture
def chat(chainlit_app, monkeypatch: pytest.MonkeyPatch) -> _Chat:
    chat = _Chat()

    class FakeMessage:
        def __init__(self, content: str = "", **_: object) -> None:
            self.content = content
            self.persisted = False

        async def send(self) -> None:
            if chat.fail_to_send:
                raise ConnectionError("the websocket is gone")
            # What Chainlit would do with the message: stored unless it says it already was.
            chat.events.append(("send", self.content, self.persisted))

        async def remove(self) -> None:
            chat.events.append(("remove", self.content))

    monkeypatch.setattr(chainlit_app.cl, "Message", FakeMessage)
    return chat


# --------------------------------------------------------------------------- #
# The notice
# --------------------------------------------------------------------------- #


def test_quick_work_says_nothing(chainlit_app, chat: _Chat) -> None:
    async def scenario() -> None:
        async with chainlit_app._slow_work_notice("busy", delay_s=0.5):
            await asyncio.sleep(0)

    asyncio.run(scenario())
    assert chat.events == []


def test_slow_work_is_announced_while_it_runs_and_the_notice_is_taken_away_after(chainlit_app, chat: _Chat) -> None:
    during: list[str] = []

    async def scenario() -> None:
        async with chainlit_app._slow_work_notice("busy", delay_s=0.02):
            await asyncio.sleep(0.2)
            during.extend(chat.kinds())

    asyncio.run(scenario())
    assert during == ["send"]
    # Said once, and said as something that must not be stored with the conversation:
    # a stored copy would be replayed every time the conversation is reopened.
    assert chat.events == [("send", "busy", True), ("remove", "busy")]


def test_the_notice_goes_away_when_the_work_fails(chainlit_app, chat: _Chat) -> None:
    async def scenario() -> None:
        async with chainlit_app._slow_work_notice("busy", delay_s=0.01):
            await asyncio.sleep(0.1)
            raise RuntimeError("scan failed")

    with pytest.raises(RuntimeError, match="scan failed"):
        asyncio.run(scenario())
    assert chat.kinds() == ["send", "remove"]


def test_stopping_the_work_takes_the_notice_away_and_the_stop_still_arrives(chainlit_app, chat: _Chat) -> None:
    """Swallowing the cancellation would leave a run the user stopped carrying on."""

    async def scenario() -> None:
        async def work() -> None:
            async with chainlit_app._slow_work_notice("busy", delay_s=0.01):
                await asyncio.sleep(30)

        task = asyncio.create_task(work())
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert chat.kinds() == ["send", "remove"]


def test_stopping_the_work_before_the_notice_is_due_sends_nothing(chainlit_app, chat: _Chat) -> None:
    async def scenario() -> None:
        async def work() -> None:
            async with chainlit_app._slow_work_notice("busy", delay_s=30):
                await asyncio.sleep(30)

        task = asyncio.create_task(work())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert chat.events == []


def test_a_notice_that_cannot_be_delivered_does_not_get_in_the_way_of_the_work(chainlit_app, chat: _Chat) -> None:
    chat.fail_to_send = True
    done: list[str] = []

    async def scenario() -> None:
        async with chainlit_app._slow_work_notice("busy", delay_s=0.01):
            await asyncio.sleep(0.1)
            done.append("work finished")

    asyncio.run(scenario())
    assert done == ["work finished"]


def test_the_delay_comes_from_the_module_setting(chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chainlit_app, "_SLOW_WORK_NOTICE_DELAY_S", 0.01)

    async def scenario() -> None:
        async with chainlit_app._slow_work_notice("busy"):
            await asyncio.sleep(0.1)

    asyncio.run(scenario())
    assert chat.kinds() == ["send", "remove"]


# --------------------------------------------------------------------------- #
# The walk before a run
# --------------------------------------------------------------------------- #


def test_the_walk_before_a_run_happens_off_the_event_loop(
    chainlit_app, chat: _Chat, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seen: dict = {}

    def walk(*roots, cancelled=None):
        seen["thread"] = threading.get_ident()
        seen["roots"] = roots
        return {"/data/before.csv"}

    monkeypatch.setattr(chainlit_app, "snapshot_files", walk)
    agent = SimpleNamespace(data_root_dir=str(tmp_path / "data"))

    async def scenario() -> tuple[int, set[str]]:
        return threading.get_ident(), await chainlit_app._snapshot_before_run(agent)

    loop_thread, files = asyncio.run(scenario())
    assert files == {"/data/before.csv"}
    assert seen["thread"] != loop_thread
    # Both places a run can write: where the app runs, and the agent's data root.
    assert seen["roots"] == (str(Path.cwd()), str(tmp_path / "data"))


def test_the_loop_keeps_answering_while_a_large_workspace_is_walked(
    chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug itself. A held loop answers no ping, so the browser gives up on it."""
    monkeypatch.setattr(chainlit_app, "snapshot_files", lambda *roots, **kwargs: time.sleep(0.4) or set())

    async def scenario() -> int:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await chainlit_app._snapshot_before_run(SimpleNamespace())
        finally:
            beat.cancel()
        return ticks

    # About 40 beats in 0.4 s; a walk that held the loop would leave none.
    assert asyncio.run(scenario()) >= 10


def test_a_slow_walk_before_a_run_is_explained_to_the_user(
    chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chainlit_app, "_SLOW_WORK_NOTICE_DELAY_S", 0.05)
    monkeypatch.setattr(chainlit_app, "snapshot_files", lambda *roots, **kwargs: time.sleep(0.3) or set())

    asyncio.run(chainlit_app._snapshot_before_run(SimpleNamespace()))

    assert chat.kinds() == ["send", "remove"]
    assert "Getting your workspace ready" in chat.contents("send")[0]


def test_a_quick_walk_before_a_run_is_not_announced(chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chainlit_app, "snapshot_files", lambda *roots, **kwargs: {"a.csv"})

    assert asyncio.run(chainlit_app._snapshot_before_run(SimpleNamespace())) == {"a.csv"}
    assert chat.events == []


# --------------------------------------------------------------------------- #
# Choosing folders
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    for folder, files in (("studyA", 2), ("studyB", 1)):
        (root / folder).mkdir(parents=True)
        for i in range(files):
            (root / folder / f"table_{i}.csv").write_text("x")
    return root


@pytest.fixture(autouse=True)
def _cold_scan_cache() -> None:
    fs_scan.clear_cache()
    yield
    fs_scan.clear_cache()


@pytest.fixture
def make_session(chainlit_app, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A WorkspaceSession over the synthetic workspace, as the app builds one (without Chainlit's login)."""
    # Results must not land in the repository, wherever the output folder would otherwise fall back to.
    monkeypatch.setenv("BIOMNI_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.chdir(tmp_path)

    def make(selected: tuple[str, ...] = ()):
        prefs = WorkspacePrefs(scope_paths=list(selected))
        root = str(workspace)
        return chainlit_app.WorkspaceSession(
            identity=SimpleNamespace(email=None, workspace_id=None, source="test"),
            prefs=prefs,
            prefs_key="test",
            store=NullPrefsStore("test"),
            registry=RunRegistry(None),
            workspace_root=root,
            scope=resolve_scope(prefs, root),
            output=resolve_output_dir(prefs, workspace_root=root),
        )

    return make


def test_a_survey_records_what_each_selected_folder_holds(chainlit_app, make_session) -> None:
    ws = chainlit_app._survey_workspace(make_session(("studyA", "studyB")))

    assert [(s.label, s.file_count) for s in ws.scope_summaries] == [("studyA", 2), ("studyB", 1)]
    assert "studyA" in ws.inventory and "table_0.csv" in ws.inventory


def test_a_survey_of_an_empty_selection_has_nothing_to_count(chainlit_app, make_session) -> None:
    ws = chainlit_app._survey_workspace(make_session())

    assert ws.scope_summaries == []
    assert "NOT been indexed" in ws.inventory


def test_the_settings_dialog_is_built_from_the_survey_and_scans_nothing(
    chainlit_app, make_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = chainlit_app._survey_workspace(make_session(("studyA",)))
    # By the time the dialog is rebuilt, a slow scan's cache entry can be older than its lifetime.
    fs_scan.clear_cache()

    def refuse(*args, **kwargs):
        raise AssertionError("the dialog walked the selected folders itself")

    monkeypatch.setattr(fs_scan, "_walk", refuse)

    picker = next(widget for widget in chainlit_app._workspace_settings_widgets(ws) if widget.id == "scope_folders")
    assert "Currently active: studyA (2 files)" in picker.description


@pytest.fixture
def settings_env(chainlit_app, chat: _Chat, make_session, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The chat as ``on_settings_update`` finds it: a session holding the workspace, and a dialog to refresh."""

    class FakeChatSettings:
        sent: list = []

        def __init__(self, widgets: list) -> None:
            self.widgets = widgets

        async def send(self) -> None:
            FakeChatSettings.sent.append(self.widgets)

    session = {"workspace": make_session(), "agent": SimpleNamespace()}
    monkeypatch.setattr(chainlit_app.cl, "ChatSettings", FakeChatSettings)
    monkeypatch.setattr(
        chainlit_app.cl,
        "user_session",
        SimpleNamespace(get=lambda key, default=None: session.get(key, default), set=session.__setitem__),
    )
    return SimpleNamespace(session=session, dialogs=FakeChatSettings.sent, chat=chat)


def _submission(*folders: str) -> dict:
    return {"scope_folders": list(folders), "scope_extra_paths": [], "output_dir": "", "remember_settings": True}


def test_changing_the_scope_never_scans_on_the_event_loop(
    chainlit_app, settings_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every scan here is cold, as one is once the folders take longer to scan than the cache lives.

    With a cache that never hits, the dialog's own count used to walk the folders
    again on the loop, holding every connection until it finished.
    """
    monkeypatch.setattr(fs_scan, "cache_ttl_s", lambda: 1e-9)
    walked_on: list[int] = []
    real_walk = fs_scan._walk

    def walk(*args, **kwargs):
        walked_on.append(threading.get_ident())
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(fs_scan, "_walk", walk)

    async def scenario() -> int:
        await chainlit_app.on_settings_update(_submission("studyA", "studyB"))
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())

    assert walked_on, "the folders were never read, so this proves nothing"
    assert loop_thread not in walked_on
    # The user still hears what the selection costs, and the dialog describes it.
    confirmation = next(text for text in settings_env.chat.contents("send") if "Data scope updated" in text)
    assert "`studyA` (2 files)" in confirmation and "`studyB` (1 file)" in confirmation
    (rebuilt,) = settings_env.dialogs
    assert any(widget.id == "scope_folders" for widget in rebuilt)


def test_overlapping_scope_changes_are_applied_one_at_a_time(
    chainlit_app, settings_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user waiting on a slow selection confirms again; the two must not write one session in turn."""
    log: list[tuple[str, tuple[str, ...]]] = []
    real_apply = chainlit_app._apply_workspace_settings

    def apply(ws, settings):
        folders = tuple(settings["scope_folders"])
        log.append(("start", folders))
        time.sleep(0.15)  # long enough for the other submission to arrive and, unguarded, begin
        try:
            return real_apply(ws, settings)
        finally:
            log.append(("end", folders))

    monkeypatch.setattr(chainlit_app, "_apply_workspace_settings", apply)

    async def scenario() -> None:
        await asyncio.gather(
            chainlit_app.on_settings_update(_submission("studyA")),
            chainlit_app.on_settings_update(_submission("studyB")),
        )

    asyncio.run(scenario())

    assert log == [
        ("start", ("studyA",)),
        ("end", ("studyA",)),
        ("start", ("studyB",)),
        ("end", ("studyB",)),
    ]
    ws = settings_env.session["workspace"]
    assert [s.label for s in ws.scope_summaries] == ["studyB"], "the last choice stands"
    assert "studyB" in ws.inventory and "studyA" not in ws.inventory


def test_the_lock_is_per_chat_not_shared_between_chats(chainlit_app, settings_env: SimpleNamespace) -> None:
    async def scenario() -> None:
        await chainlit_app.on_settings_update(_submission("studyA"))

    asyncio.run(scenario())
    first = settings_env.session["settings_lock"]
    assert isinstance(first, asyncio.Lock)
    assert not first.locked(), "released once the update is done, or the next one would wait for ever"


def test_a_failed_scope_change_does_not_hold_the_next_one_up(
    chainlit_app, settings_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real_apply = chainlit_app._apply_workspace_settings

    def apply(ws, settings):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("the share went away")
        return real_apply(ws, settings)

    monkeypatch.setattr(chainlit_app, "_apply_workspace_settings", apply)

    async def scenario() -> None:
        await chainlit_app.on_settings_update(_submission("studyA"))
        await chainlit_app.on_settings_update(_submission("studyB"))

    asyncio.run(scenario())
    assert any("Could not apply those settings" in text for text in settings_env.chat.contents("send"))
    assert any("Data scope updated" in text for text in settings_env.chat.contents("send"))


# --------------------------------------------------------------------------- #
# Stop
# --------------------------------------------------------------------------- #


def test_stopping_a_run_before_it_starts_ends_the_walk_instead_of_leaving_it_running(
    chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thread cannot be cancelled from the loop. Left alone, it reads the rest of the share for nothing."""
    started, ended = threading.Event(), threading.Event()
    outcome: list[str] = []

    def walk(*roots, cancelled=None):
        started.set()
        # Gives up on its own if Stop never arrives, so a regression fails this test instead of hanging the run.
        give_up = time.monotonic() + 4
        while not cancelled() and time.monotonic() < give_up:
            time.sleep(0.01)
        outcome.append("stopped" if cancelled() else "gave up")
        ended.set()
        raise SnapshotCancelled("stopped")

    monkeypatch.setattr(chainlit_app, "snapshot_files", walk)
    monkeypatch.setattr(chainlit_app, "_SLOW_WORK_NOTICE_DELAY_S", 0.01)

    async def scenario() -> None:
        task = asyncio.create_task(chainlit_app._snapshot_before_run(SimpleNamespace()))
        for _ in range(200):  # until the walk is under way
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert started.is_set()
    assert ended.wait(timeout=2), "the walk went on after Stop"
    assert outcome == ["stopped"]
    assert chat.kinds() == ["send", "remove"], "and the notice is gone"


def test_a_walk_that_finishes_is_not_told_to_stop_early(
    chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags: list[bool] = []

    def walk(*roots, cancelled=None):
        flags.append(cancelled())
        return {"a.csv"}

    monkeypatch.setattr(chainlit_app, "snapshot_files", walk)

    assert asyncio.run(chainlit_app._snapshot_before_run(SimpleNamespace())) == {"a.csv"}
    assert flags == [False]


# --------------------------------------------------------------------------- #
# Opening a chat
# --------------------------------------------------------------------------- #


def test_a_slow_start_says_the_chat_is_being_got_ready_and_does_not_blame_the_folders(
    chainlit_app, chat: _Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first chat after a restart is slow for reasons that have nothing to do with the workspace."""
    monkeypatch.setattr(chainlit_app, "_SLOW_WORK_NOTICE_DELAY_S", 0.01)

    async def slow_open(resumed_thread=None) -> None:
        await asyncio.sleep(0.1)

    monkeypatch.setattr(chainlit_app, "_open_session", slow_open)

    asyncio.run(chainlit_app._start_session())

    assert chat.kinds() == ["send", "remove"]
    (text,) = chat.contents("send")
    assert text.startswith("⏳ **Getting your chat ready.**")
    assert "Large folders" not in text, "it must not claim a cause it cannot know"
