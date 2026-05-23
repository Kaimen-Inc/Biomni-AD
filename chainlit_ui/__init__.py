"""Chainlit UI helpers extracted from `chainlit_app.py`.

These submodules import `chainlit` at module load time, so they must be
imported only after the env guard at the top of `chainlit_app.py` has
confirmed the heavy biomedical deps (pandas, langchain) are available.
"""
