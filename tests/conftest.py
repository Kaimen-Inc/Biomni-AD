"""Process-wide state that one test must not leak into the next."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture(autouse=True)
def _restore_thread_pool_submit():
    """Undo ``carry_context_into_thread_pools``, which running generated code installs.

    The app installs it once and for good. In a test run that would change how
    every later test's thread pools behave, depending on which test ran first.
    """
    submit = ThreadPoolExecutor.submit
    yield
    ThreadPoolExecutor.submit = submit  # type: ignore[method-assign]
