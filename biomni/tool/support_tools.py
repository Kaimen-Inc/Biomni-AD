"""Execution environment for LLM-generated code, isolated per chat session.

The REPL state here used to be three module-level globals shared by the whole
process. That is correct for a notebook and wrong for a server: the Chainlit app
runs one chat per session and several sessions at once, so two people asking a
question at the same moment executed against the same namespace, the same plot
list and - worst of all - the same ``sys.stdout``. Concretely, run A would
receive run B's printed output and B's ``OUTPUT_DIR``, so A's generated files
landed in B's run directory; and because each execution restored whatever
``sys.stdout`` happened to be installed when it started, a single overlap left
the process writing to a discarded buffer, silently swallowing every subsequent
print.

State is therefore keyed by the chat session (``biomni.observability.session_id_var``,
which the Chainlit app binds per chat and which :func:`biomni.utils.run_with_timeout`
propagates into the execution thread). Outside a bound session - notebooks, the
CLI, tests - every caller shares one default session, which is exactly the old
single-namespace behaviour.

Stdout is routed rather than swapped: one installed router dispatches each write
to the buffer of the execution that is running in the calling context, falling
back to the real stream. Nothing restores a stale handle, so concurrent
executions cannot corrupt each other's capture or the process's stdout.

**Known residual.** ``matplotlib.pyplot``'s figure manager is process-global, so
two sessions plotting at the same instant can still capture each other's
figures. Fixing that means driving generated code to the object-oriented API,
which is not something this layer can impose.
"""

import base64
import contextvars
import io
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from io import StringIO

from biomni import credentials
from biomni.credentials import scrubbed_environ
from biomni.observability import session_id_var

logger = logging.getLogger(__name__)


def _positive_int(env_name: str, default: int) -> int:
    """An int from the environment, never fatal.

    Parsed at import, so an unparseable value - a typo, or a ConfigMap key with
    an empty value - would otherwise raise on ``import biomni.tool.support_tools``
    and the app would never boot. Matching the guards biomni/status.py already
    has for its own windows.
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r; using %d", env_name, raw, default)
        return default
    if value < 1:
        logger.warning("ignoring non-positive %s=%r; using %d", env_name, raw, default)
        return default
    return value


# Session used by every caller that has not bound a session id: notebooks, the
# CLI, direct library use. Keeping them on one shared session preserves the
# "variables persist between calls" contract those callers rely on.
_DEFAULT_SESSION = "__default__"


# Upper bound on retained session state. Chat sessions end without a reliable
# callback (a closed tab never says goodbye), so the registry evicts the least
# recently used entry instead of trusting a teardown hook to fire. On any real
# deployment the cap is never reached; it exists so a long-lived server cannot
# grow a namespace per visitor forever.
_MAX_SESSIONS = _positive_int("BIOMNI_MAX_REPL_SESSIONS", 32)

# Upper bound on figures left open across executions. Capture deliberately no
# longer closes them - doing so mid-savefig blanked the user's next save - but
# pyplot's figure manager is process-global, so without a bound every figure any
# chat ever drew stays resident for the life of the pod. Closing the oldest
# preserves the Jupyter-like behaviour that makes a figure usable in a later
# step, while keeping the registry from growing without limit.
_MAX_OPEN_FIGURES = _positive_int("BIOMNI_MAX_OPEN_FIGURES", 50)


def _open_figure_numbers() -> list[int]:
    """Figure numbers currently open, or an empty list if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt

        return sorted(plt.get_fignums())
    except Exception:
        return []


def _bound_open_figures(preexisting: set[int]) -> int:
    """Close figures *this execution* created beyond the cap.

    ``preexisting`` is what was open before the snippet ran, and is never
    touched: pyplot allocates numbers monotonically, so the lowest-numbered
    figures are the longest-lived, which means they belong to another chat that
    is still working on them. Closing those reintroduced exactly the blank-file
    bug across sessions - the other chat's next ``plt.figure(3); plt.savefig()``
    would silently write an empty image. If a session's own figures are not
    enough to get under the cap, the registry is left too large rather than
    reaching into somebody else's.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return 0
    numbers = _open_figure_numbers()
    excess = len(numbers) - _MAX_OPEN_FIGURES
    if excess <= 0:
        return 0
    mine = [n for n in numbers if n not in preexisting]
    closed = 0
    for number in mine[: min(excess, len(mine))]:
        try:
            plt.close(number)
            closed += 1
        except Exception:
            logger.debug("could not close figure %s", number, exc_info=True)
    if closed:
        logger.info("closed %d figure(s) this execution created, beyond the %d kept open", closed, _MAX_OPEN_FIGURES)
    return closed


# How long a session's state is protected from eviction, regardless of how many
# other chats have run since. The cap below bounds memory; this bounds the
# damage, because eviction is silent from the user's side: their next step comes
# back "name 'adata' is not defined" for a frame they correctly believe they
# loaded.
_SESSION_TTL_SECONDS = float(_positive_int("BIOMNI_REPL_SESSION_TTL_SECONDS", 6 * 60 * 60))


class _ReplSession:
    """Per-chat execution state: the exec namespace and captured plots."""

    __slots__ = ("last_used", "namespace", "plots")

    def __init__(self, now: float) -> None:
        self.namespace: dict = {}
        self.plots: list[str] = []
        self.last_used: float = now


_sessions: OrderedDict[str, _ReplSession] = OrderedDict()
_sessions_lock = threading.Lock()

# Buffer for the execution running in the current context. Set for the duration
# of one ``run_python_repl`` call; ``None`` means "not inside an execution", and
# writes fall through to the real stream.
_active_buffer: contextvars.ContextVar[StringIO | None] = contextvars.ContextVar("biomni_repl_stdout", default=None)


def current_session_key() -> str:
    """Session key for the calling context, or the shared default."""
    return session_id_var.get() or _DEFAULT_SESSION


def _evict_locked(now: float) -> None:
    """Bring the registry back under the cap. Caller holds the lock.

    Idle entries go first, oldest first. A session is "idle" only once it has
    gone ``_SESSION_TTL_SECONDS`` without being touched, so a researcher reading
    a result for twenty minutes is not evicted merely because a busy deployment
    ran thirty other chats meanwhile - which is the same user the status
    endpoint's open-session window exists to protect, and evicting them here
    would undo that protection one layer down.

    If nothing is idle the cap still has to hold, so the least-recently-used
    live session is dropped - but at WARNING, because that one costs somebody
    their working state and means the cap is set too low for the deployment.
    """
    while len(_sessions) > _MAX_SESSIONS:
        # Genuinely idle entries first, oldest of those first; only if none has
        # aged out is a live session taken, because that costs somebody their
        # working state. Both branches evict - the cap has to hold either way -
        # but which entry goes, and how loudly, is the point.
        idle = [(k, v) for k, v in _sessions.items() if now - v.last_used >= _SESSION_TTL_SECONDS]
        if idle:
            key, session = idle[0]
            _sessions.pop(key)
            logger.info("released idle REPL session state (key=%s, idle=%.0fs)", key, now - session.last_used)
            continue
        key, session = next(iter(_sessions.items()))
        _sessions.pop(key)
        logger.warning(
            "evicted REPL session state that was still in use (key=%s, idle=%.0fs); "
            "raise BIOMNI_MAX_REPL_SESSIONS above %d - that chat's variables are gone",
            key,
            now - session.last_used,
            _MAX_SESSIONS,
        )


def _session() -> _ReplSession:
    key = current_session_key()
    now = time.monotonic()
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = _ReplSession(now)
            _sessions[key] = session
            _evict_locked(now)
        else:
            session.last_used = now
            _sessions.move_to_end(key)
        return session


def get_repl_namespace() -> dict:
    """The ``exec`` namespace for the calling session.

    This is what callers inject into (custom tools, ``OUTPUT_DIR``); mutating
    the returned dict mutates the live namespace.
    """
    return _session().namespace


def discard_repl_session(session_key: str | None = None) -> None:
    """Drop one session's REPL state. Defaults to the calling session."""
    key = session_key or current_session_key()
    with _sessions_lock:
        _sessions.pop(key, None)


class _StdoutRouter(io.TextIOBase):
    """Dispatches writes to the running execution's buffer, else to ``fallback``.

    Installed once on ``sys.stdout`` and left there. The alternative - swapping
    ``sys.stdout`` per execution - is what made concurrent runs cross-capture
    and permanently break the stream, because each execution restored the handle
    it found rather than the one it replaced.
    """

    def __init__(self, fallback):
        self._fallback = fallback

    @property
    def fallback(self):
        return self._fallback

    def _target(self):
        return _active_buffer.get() or self._fallback

    def write(self, s) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        target = self._target()
        if hasattr(target, "flush"):
            target.flush()

    def isatty(self) -> bool:
        # False while capturing: the buffer is not a terminal, and answering for
        # the real stream made rich and tqdm write ANSI escapes into text that
        # ends up in the agent's observation.
        if _active_buffer.get() is not None:
            return False
        return getattr(self._fallback, "isatty", lambda: False)()

    def fileno(self) -> int:
        # No descriptor while capturing. subprocess(stdout=sys.stdout) would
        # otherwise be handed the real fd and write past the buffer straight to
        # the container log, so the agent never sees that output.
        if _active_buffer.get() is not None:
            raise io.UnsupportedOperation("fileno: stdout is captured for this execution")
        return self._fallback.fileno()

    @property
    def encoding(self):
        return getattr(self._fallback, "encoding", "utf-8")

    def writable(self) -> bool:
        return True

    def __getattr__(self, name):
        # Everything not overridden above is answered by the stream we wrap.
        # The router is installed on the first code step and never removed, so
        # without this the standard binary-stdout idioms break process-wide and
        # permanently once any chat has run code:
        #   fig.savefig(sys.stdout.buffer, format="png")
        #   df.to_parquet(sys.stdout.buffer)
        #   sys.stdout.reconfigure(encoding="utf-8")
        # all raised AttributeError, and only on a server that had served a
        # previous session - the same snippet worked in a fresh process.
        return getattr(self._fallback, name)


_router_lock = threading.Lock()


def _ensure_stdout_router() -> None:
    """Install the router, re-wrapping if something else replaced ``sys.stdout``.

    Self-healing rather than install-once: Chainlit, uvicorn or a notebook may
    install their own stream after we do, and a router still pointing at a
    superseded fallback would send output nowhere.

    Locked because concurrent chats reach this at the same moment, and an
    unguarded check-then-set lets both see a foreign stream and wrap it twice -
    ``router(router(real))``, which observability's single-level ``fallback``
    unwrap cannot see through, so log records would be captured into whichever
    snippet happened to be running.
    """
    with _router_lock:
        current = sys.stdout
        if isinstance(current, _StdoutRouter):
            return
        # Unwrap any router already buried under a foreign wrapper so the chain
        # never nests: routing through a stale router would send output to a
        # fallback that is no longer the real stream.
        inner = getattr(current, "fallback", None)
        while isinstance(inner, _StdoutRouter):
            current, inner = inner, getattr(inner, "fallback", None)
        sys.stdout = _StdoutRouter(current)


def run_python_repl(command: str) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """

    def execute_in_repl(command: str) -> str:
        """Execute the command in this session's namespace, capturing its output."""
        namespace = get_repl_namespace()
        preexisting_figures = set(_open_figure_numbers())
        buffer = StringIO()
        _ensure_stdout_router()
        token = _active_buffer.set(buffer)

        try:
            # Apply matplotlib monkey patches before execution
            _apply_matplotlib_patches()

            # Provider credentials are stripped from os.environ for the duration
            # of the call: this code is written by the model, from a prompt any
            # user can supply, and it runs un-sandboxed in this process.
            with scrubbed_environ():
                exec(command, namespace)
            output = buffer.getvalue()

            # Capture any matplotlib plots that were generated
            # _capture_matplotlib_plots()

        except Exception as e:
            # Keep whatever the snippet printed before it failed - the agent
            # uses it to work out which step broke, and discarding it turns an
            # informative traceback into a bare error string.
            partial = buffer.getvalue()
            output = f"{partial}Error: {str(e)}" if partial else f"Error: {str(e)}"
        finally:
            _active_buffer.reset(token)
            _bound_open_figures(preexisting_figures)
        return output

    command = command.strip("```").strip()
    return execute_in_repl(command)


def _capture_matplotlib_plots(target=None):
    """Capture ``target`` (default: the current figure) as a base64 PNG.

    Only ever one figure, and never via ``plt.figure(n)``. Iterating every open
    figure was wrong three ways once figures stopped being closed: it re-encoded
    the whole registry on every ``savefig`` - N(N+1)/2 renders at dpi=150 with
    ``bbox_inches='tight'``, measured at 6.5x slower for 20 figures - it swept in
    figures belonging to *other* chat sessions, since pyplot's registry is
    process-global while the plot list is per session, and ``plt.figure(n)``
    makes each one current as a side effect, so a bare ``plt.savefig()`` after a
    capture wrote whichever figure the loop happened to leave selected.
    """
    plots = _session().plots
    try:
        import matplotlib

        # Enforce a headless backend for threaded/server execution contexts.
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = target if target is not None else (plt.gcf() if plt.get_fignums() else None)
        if figure is not None:
            for fig in (figure,):
                # Save figure to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)

                # Convert to base64
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plot_data = f"data:image/png;base64,{image_data}"

                # Add to captured plots if not already there
                if plot_data not in plots:
                    plots.append(plot_data)

                # The figure is deliberately NOT closed. This runs from the
                # savefig/show monkey patches, i.e. in the middle of the user's
                # own plotting code, and closing here destroyed the figure they
                # were still working on: the extremely common
                #   plt.savefig("x.png"); plt.savefig("x.pdf")
                # wrote a correct PNG and then a blank 1 KB PDF, because by the
                # second call there was no figure left. Capture must observe,
                # not dispose - the generated code owns the figure's lifetime.
                # Duplicates are already prevented by the identity check above.

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not capture matplotlib plots: {e}")


def _apply_matplotlib_patches():
    """Apply simple monkey patches to matplotlib functions to automatically capture plots."""
    try:
        import matplotlib

        # On macOS, GUI backends (e.g. MacOSX) crash when used outside the main thread.
        # Biomni executes Python snippets in worker threads, so force a non-GUI backend.
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        # Only patch if matplotlib is available and not already patched
        if hasattr(plt, "_biomni_patched"):
            return

        # Store original functions
        original_show = plt.show
        original_savefig = plt.savefig

        def show_with_capture(*args, **kwargs):
            """Enhanced show function that captures plots before displaying them."""
            # Capture before showing - this figure, not every open one.
            _capture_matplotlib_plots(plt.gcf() if plt.get_fignums() else None)
            # Print a message to indicate plot was generated
            print("Plot generated and displayed")
            # Call the original show function
            return original_show(*args, **kwargs)

        def savefig_with_capture(*args, **kwargs):
            """Enhanced savefig function that captures plots after saving them."""
            # Get the filename from args if provided
            filename = args[0] if args else kwargs.get("fname", "unknown")
            # Call the original savefig function
            result = original_savefig(*args, **kwargs)
            # Capture the plot after saving - this figure, not every open one.
            _capture_matplotlib_plots(plt.gcf() if plt.get_fignums() else None)
            # Print a message to indicate plot was saved
            print(f"Plot saved to: {filename}")
            return result

        # Replace functions with enhanced versions
        plt.show = show_with_capture
        plt.savefig = savefig_with_capture

        # Mark as patched to avoid double-patching
        plt._biomni_patched = True

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not apply matplotlib patches: {e}")


def get_captured_plots():
    """Get the plots captured for the calling session."""
    return _session().plots.copy()


def clear_captured_plots():
    """Clear the plots captured for the calling session."""
    _session().plots.clear()


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Parameters
    ----------
        function_name (str): Fully qualified function name (e.g., 'bioagentos.tool.support_tools.write_python_code')

    Returns
    -------
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


# def request_human_feedback(question, context, reason_for_uncertainty):
#     """
#     Request human feedback on a question.

#     Parameters:
#         question (str): The question that needs human feedback.
#         context (str): Context or details that help the human understand the situation.
#         reason_for_uncertainty (str): Explanation for why the LLM is uncertain about its answer.

#     Returns:
#         str: The feedback provided by the human.
#     """
#     print("Requesting human feedback...")
#     print(f"Question: {question}")
#     print(f"Context: {context}")
#     print(f"Reason for Uncertainty: {reason_for_uncertainty}")

#     # Capture human feedback
#     human_response = input("Please provide your feedback: ")

#     return human_response


def download_synapse_data(
    entity_ids: str | list[str],
    download_location: str = ".",
    follow_link: bool = False,
    recursive: bool = False,
    timeout: int = 300,
    entity_type: str = "dataset",
):
    """Download data from Synapse using entity IDs.

    Uses the synapse CLI to download files, folders, or projects from Synapse.
    Requires SYNAPSE_AUTH_TOKEN environment variable for authentication.
    Automatically installs synapseclient if not available.

    CRITICAL: Always check entity type from query_synapse() search results or user hints and pass the correct entity_type!
    The default entity_type="dataset" may not be appropriate for your entity.

    IMPORTANT: Multiple entity IDs are only supported for entity_type="file".
    For datasets, folders, and projects, only a single entity_id is supported.

    Parameters
    ----------
    entity_ids : str or list of str
        Synapse entity ID(s) to download.
        - For files: Can be a single ID string or list of ID strings
        - For datasets/folders/projects: Must be a single ID string only
    download_location : str, default "."
        Directory where files will be downloaded (current directory by default)
    follow_link : bool, default False
        Whether to follow links to download the linked entity
    recursive : bool, default False
        Whether to recursively download folders and their contents
        ONLY valid for entity_type="folder" - ignored for other types
    timeout : int, default 300
        Timeout in seconds for each download operation
    entity_type : str, default "dataset"
        Type of Synapse entity ("dataset", "file", "folder", "project")
        MUST match the actual entity type from search results or user hints!
        The default "dataset" should only be used for actual datasets.
        Check the 'node_type' field in search results to determine correct type.

    Returns
    -------
    dict
        Dictionary containing download results and any errors

    Notes
    -----
    Requires SYNAPSE_AUTH_TOKEN environment variable with your Synapse personal
    access token for authentication.

    AGENT USAGE GUIDANCE:
    1. Always check the 'node_type' field from query_synapse() search results or user hints
    2. Pass the correct entity_type parameter matching the node_type
    3. Do NOT rely on the default entity_type="dataset" unless confirmed
    4. For multiple downloads, ensure all entities are of type "file"
    5. Only use recursive=True with entity_type="folder"

    Examples
    --------
    # After searching with query_synapse(), check node_type and use appropriate entity_type:

    # If search result shows 'node_type': 'dataset'
    download_synapse_data("syn123456", entity_type="dataset")

    # If search result shows 'node_type': 'file'
    download_synapse_data("syn654321", entity_type="file")

    # If search result shows 'node_type': 'folder'
    download_synapse_data("syn789012", entity_type="folder", recursive=True)

    # Multiple files (only if all are 'node_type': 'file')
    download_synapse_data(["syn111", "syn222"], entity_type="file")
    """
    import os
    import subprocess

    # Check for required authentication token
    synapse_token = credentials.getenv("SYNAPSE_AUTH_TOKEN")
    if not synapse_token:
        return {
            "success": False,
            "error": "SYNAPSE_AUTH_TOKEN environment variable is required for downloading",
            "suggestion": "Set SYNAPSE_AUTH_TOKEN with your Synapse personal access token",
        }

    # Check if synapse CLI is available
    try:
        subprocess.run(["synapse", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try to install synapseclient
            print("Installing synapseclient...")
            subprocess.run(["pip", "install", "synapseclient"], check=True)
            print("✓ synapseclient installed successfully")
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Failed to install synapseclient: {e}",
                "suggestion": "Please install manually: pip install synapseclient",
            }

    # Ensure entity_ids is a list
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    # Validate that multiple IDs are only used with file entity type
    if len(entity_ids) > 1 and entity_type != "file":
        return {
            "success": False,
            "error": f"Multiple entity IDs are only supported for entity_type='file'. "
            f"For entity_type='{entity_type}', only a single entity_id is supported.",
            "suggestion": "Use a single entity_id string instead of a list, or change entity_type to 'file'",
        }

    # Validate that recursive is only used with folder entity type
    if recursive and entity_type != "folder":
        return {
            "success": False,
            "error": f"recursive=True is only valid for entity_type='folder'. "
            f"For entity_type='{entity_type}', recursive should be False.",
            "suggestion": "Set recursive=False, or change entity_type to 'folder' if appropriate",
        }

    # Create download directory if it doesn't exist
    os.makedirs(download_location, exist_ok=True)

    results = []
    errors = []

    for entity_id in entity_ids:
        try:
            # Build synapse download command with authentication
            if entity_type == "dataset":
                # For datasets, use query syntax to download the actual files
                cmd = [
                    "synapse",
                    "-p",
                    synapse_token,
                    "get",
                    "-q",
                    f"select * from {entity_id}",
                    "--downloadLocation",
                    download_location,
                ]
            else:
                # For files, folders, projects, use direct ID
                cmd = ["synapse", "-p", synapse_token, "get", entity_id, "--downloadLocation", download_location]

            # Add recursive flag only for folders (validation above ensures recursive is only True for folders)
            if entity_type == "folder" and recursive:
                cmd.append("-r")

            if follow_link:
                cmd.append("--followLink")

            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)

            results.append(
                {
                    "entity_id": entity_id,
                    "success": True,
                    "stdout": result.stdout,
                    "download_location": download_location,
                }
            )

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to download {entity_id}: {e.stderr if e.stderr else str(e)}"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})
        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout for {entity_id} (>{timeout} seconds)"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})

    # Summary
    successful_downloads = [r for r in results if r["success"]]
    failed_downloads = [r for r in results if not r["success"]]

    return {
        "success": len(failed_downloads) == 0,
        "total_requested": len(entity_ids),
        "successful": len(successful_downloads),
        "failed": len(failed_downloads),
        "download_location": download_location,
        "results": results,
        "errors": errors if errors else None,
    }
