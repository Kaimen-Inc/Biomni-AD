"""Chainlit UI helpers extracted from `chainlit_app.py`.

`chainlit_ui.planning` imports `chainlit` at module load time, so it
must be imported only after the env guard at the top of
`chainlit_app.py` has confirmed the heavy biomedical deps (pandas,
langchain) are available. `chainlit_ui.datasets` is pure-Python and
has no chainlit dependency.
"""
