"""Fetch the real app's page with the logout item switched off; run by test_ui_flags_app.py.

A separate interpreter because chainlit_app reads its settings when imported and
registers its callbacks on Chainlit's module-global server, so it cannot be set
up twice, with different settings, in one test process.

Prints one line, ``UI_FLAGS_RESULT {"checks": [[label, got, want], ...]}``.
"""

import json
import os
import sys
from pathlib import Path

import dotenv

# The environment is the test's to define: a developer's .env, which the app
# loads over it, could otherwise switch the setting back off.
dotenv.load_dotenv = lambda *args, **kwargs: False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(os.environ["CHAINLIT_APP_ROOT"])

import chainlit_app  # noqa: F401  (reads BIOMNI_HIDE_LOGOUT and picks the stylesheet)
from chainlit.server import app
from starlette.testclient import TestClient

checks: list[list] = []


def check(label, got, want):
    checks.append([label, got, want])


client = TestClient(app)
page = client.get("/")
check("the page loads", page.status_code, 200)
check("it links the logout-hiding stylesheet", 'href="/public/biomni-hide-logout.css"' in page.text, True)
check("and not the plain one", 'href="/public/biomni.css"' in page.text, False)

variant = client.get("/public/biomni-hide-logout.css")
check("the stylesheet is served", variant.status_code, 200)
check("it hides the item by its icon", "lucide-log-out" in variant.text, True)
check("the stylesheet it imports is served beside it", client.get("/public/biomni.css").status_code, 200)

print("UI_FLAGS_RESULT " + json.dumps({"checks": checks}), flush=True)
