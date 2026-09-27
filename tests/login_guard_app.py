"""Sign in to the real app through Chainlit's HTTP API; run by test_login_guard.py.

A separate interpreter because chainlit_app reads its settings when imported and
registers its callbacks on Chainlit's module-global server, so it cannot be set
up twice, with different settings, in one test process.

Prints one line, ``LOGIN_GUARD_RESULT {"checks": [[label, got, want], ...]}``.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import dotenv

# The environment is the test's to define: a developer's .env, which the app
# loads over it, could otherwise switch the gateway trust off.
dotenv.load_dotenv = lambda *args, **kwargs: False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(os.environ["CHAINLIT_APP_ROOT"])

import chainlit_app  # noqa: F401  (registers the callbacks and installs the guard)
from chainlit.auth import create_jwt, get_current_user
from chainlit.data import get_data_layer
from chainlit.server import app
from chainlit.user import User
from starlette.testclient import TestClient

checks: list[list] = []


def check(label, got, want):
    checks.append([label, got, want])


def gateway(sub, workspace="ws-1"):
    """The headers the platform's gateway sends for ``sub`` in ``workspace``."""
    return {
        "Ai-App-User-Context": f"email={sub}@example.org,given_name={sub.title()},family_name=X,sub={sub}",
        "Ai-App-Workspace-Context": f"uuid={workspace}",
    }


def cookie_for(user):
    client = TestClient(app)
    client.cookies.set("access_token", create_jwt(user))
    return client


check("the guard is installed", get_current_user in app.dependency_overrides, True)

alice = TestClient(app)
check("alice signs in", alice.post("/auth/header", headers=gateway("alice")).status_code, 200)
me = alice.get("/user", headers=gateway("alice"))
check("alice's cookie with alice's headers", me.status_code, 200)
check("alice's cookie carries her subject", me.json()["metadata"]["user_id"], "alice")
check("alice's cookie with bob's headers", alice.get("/user", headers=gateway("bob")).status_code, 401)
check("alice's cookie in another workspace", alice.get("/user", headers=gateway("alice", "ws-2")).status_code, 401)
check(
    "alice's cookie on /project/settings as bob",
    alice.get("/project/settings", headers=gateway("bob")).status_code,
    401,
)
check("health checks need no sign-in", alice.get("/healthz", headers=gateway("bob")).status_code, 200)

# Cookies the previous release issued while the gateway was not trusted: one
# shared identity for every browser, and no user_id in the metadata.
released = {"email": "", "workspace_id": "", "given_name": "", "family_name": ""}
for identity in ("local", "anonymous"):
    stale = cookie_for(User(identifier=identity, display_name=identity, metadata={"source": identity, **released}))
    check(f"a released '{identity}' cookie as alice", stale.get("/user", headers=gateway("alice")).status_code, 401)
    listing = stale.post("/project/threads", headers=gateway("carol"), json={"pagination": {"first": 20}, "filter": {}})
    check(f"a released '{identity}' cookie listing threads as carol", listing.status_code, 401)

# A gateway cookie from the previous release: the right person, but it cannot
# say which subject it was issued for.
old_bob = cookie_for(
    User(
        identifier="bob-81b637d8fc@ws-1",
        display_name="Bob X",
        metadata={"source": "headers", "email": "bob@example.org", "workspace_id": "ws-1", "given_name": "Bob"},
    )
)
check("a released gateway cookie for bob, as bob", old_bob.get("/user", headers=gateway("bob")).status_code, 401)
check("bob signs in again", old_bob.post("/auth/header", headers=gateway("bob")).status_code, 200)
check("bob's new cookie", old_bob.get("/user", headers=gateway("bob")).status_code, 200)

# A client-sent copy of the context header under an underscore name.
forged = TestClient(app)
forged.post("/auth/header", headers={**gateway("carol"), "Ai_App_User_Context": "sub=alice"})
check(
    "an underscore copy of the header does not sign in as alice",
    forged.get("/user", headers=gateway("carol")).json()["metadata"]["user_id"],
    "carol",
)


async def write_alices_thread():
    await get_data_layer().update_thread("t-alice-1", name="alice: APOE4 carriers", user_id=me.json()["id"])


asyncio.run(write_alices_thread())
carol = TestClient(app)
carol.post("/auth/header", headers=gateway("carol"))
listed = carol.post("/project/threads", headers=gateway("carol"), json={"pagination": {"first": 20}, "filter": {}})
check("carol's thread list", [t.get("name") for t in listed.json().get("data", [])], [])
opened = carol.get("/project/thread/t-alice-1", headers=gateway("carol")).status_code
check("carol cannot open alice's thread", opened in (401, 403, 404), True)

print("LOGIN_GUARD_RESULT " + json.dumps({"checks": checks}), flush=True)
