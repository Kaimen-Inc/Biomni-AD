# Biomni Configuration Guide

## Quick Start

**Recommended approach**: Use environment variables or modify `default_config` for consistent behavior across your entire application.

```python
from biomni.config import default_config
from biomni.agent import A1

# Option 1: Modify global defaults (affects everything)
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200

# Option 2: Use environment variables (set in .env file)
# BIOMNI_LLM=gpt-4
# BIOMNI_TIMEOUT_SECONDS=1200

agent = A1()  # Uses your configuration
```

## Configuration Methods

### 1. Environment Variables (Recommended for Production)

Create a `.env` file in your project:

```bash
# Required API Keys (set at least one provider profile)
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Optional Settings
BIOMNI_LLM=claude-3-5-sonnet-20241022
BIOMNI_TIMEOUT_SECONDS=1200
BIOMNI_PATH=/path/to/data
```

### 2. Runtime Configuration (Recommended for Scripts)

```python
from biomni.config import default_config

# Changes apply to all agents and database queries
default_config.llm = "gpt-4"
default_config.timeout_seconds = 1200
```

### 3. Direct Parameters (Use with Caution)

```python
# ⚠️ Only affects this agent's reasoning, NOT database queries
agent = A1(llm="claude-3-5-sonnet-20241022")
```

## Common Examples

### Using Different Models

```python
# Use GPT-4 everywhere
default_config.llm = "gpt-4"
agent = A1()
```

### Cost Optimization (Different Models for Agent vs Database)

```python
# Cheaper model for database queries
default_config.llm = "claude-3-5-haiku-20241022"

# More powerful model for agent reasoning
agent = A1(llm="claude-3-5-sonnet-20241022")
```

### Custom/Local Models

```python
default_config.source = "Custom"
default_config.base_url = "http://localhost:8000/v1"
default_config.api_key = "local_key"
default_config.llm = "local-llama-70b"
```

## All Available Settings

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
AWS_BEARER_TOKEN_BEDROCK=your_key
AWS_REGION=us-east-1

# Azure Anthropic (Claude via Azure AI Foundry)
ENDPOINT_URL=https://your-resource.services.ai.azure.com/anthropic/
DEPLOYMENT_NAME=your_claude_deployment_name
AZURE_ANTHROPIC_API_KEY=your_key

# Azure OpenAI (GPT via Azure OpenAI)
OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key

# Optional custom direct endpoints
ANTHROPIC_BASE_URL=https://api.anthropic.com
OPENAI_BASE_URL=https://api.openai.com/v1

# Biomni Settings
BIOMNI_PATH=/path/to/data                   # Default: ~/.biomni/data (the app's own data directory;
                                             #          /app/data in the container image)
BIOMNI_DATA_LAKE_PATH=/data/data_lake        # Default: data/biomni_data/data_lake in the checkout
                                             #          (/app/data/biomni_data/data_lake in the image).
                                             #          Reference datasets, ~16.5 GB if all are used,
                                             #          each downloaded the first time a question needs it.
BIOMNI_USER_DATA_PATH=/app/user-data         # Default: unset. The user's workspace: their own input
                                             #          data, shown in Settings and read by the agent.
                                             #          /readyz waits for it when set.
BIOMNI_DATA_PATH=/app/user-data              # Default: unset. Older name for the workspace, read
                                             #          when BIOMNI_USER_DATA_PATH is not set
BIOMNI_USER_DATA_HOST_PATH=/mnt/data         # Default: unset. Compose only: the host folder mounted
                                             #          at /app/user-data (also read as a workspace
                                             #          when the path exists inside the container)
BIOMNI_TIMEOUT_SECONDS=1200                 # Default: 600  (per code/tool step)
BIOMNI_LLM=model_name                        # Default: claude-sonnet-4-5 (BIOMNI_LLM_MODEL is an alias)
BIOMNI_TEMPERATURE=0.7                      # Default: 0.7
BIOMNI_USE_TOOL_RETRIEVER=true             # Default: true
BIOMNI_AUTO_NETWORK_LIMITED_MODE=true      # Default: true
LLM_SOURCE=Anthropic                        # Preferred source selector
BIOMNI_SOURCE=Anthropic                     # Also supported (backward compatibility)
BIOMNI_CUSTOM_BASE_URL=http://localhost:8000/v1
BIOMNI_CUSTOM_API_KEY=custom_key
BIOMNI_COMMERCIAL_MODE=true                  # Default: false (leave out datasets and know-how whose
                                             #          licence does not allow commercial use)

# Chat interface
BIOMNI_AGENT=ad1                             # Default: unset (the user picks AD1 or A1 as the chat
                                             #          profile; set a1 or ad1 to fix the agent)
BIOMNI_ALWAYS_PLAN=true                      # Default: false (short questions are answered directly;
                                             #          true sends every question through a plan the
                                             #          user approves)
CHAINLIT_HOST=0.0.0.0                        # Default: 0.0.0.0 (container entrypoint only)
CHAINLIT_PORT=8000                           # Default: 8000    (container entrypoint only)

# Tokens for individual data services, used by the tools that query them
PROTOCOLS_IO_ACCESS_TOKEN=token              # protocols.io (BIOMNI_PROTOCOLS_IO_ACCESS_TOKEN also read)
SYNAPSE_AUTH_TOKEN=token                     # Synapse

# Platform LLM proxy (GRIP; see docs/grip_deployment.md). When set, every model call
# goes through it, whatever the provider settings above say, and each request names
# the session's user and workspace in X-User-Id and X-Workspace-Id.
BIOMNI_LLM_PROXY_URL=https://proxy/v1        # Default: unset (model calls go direct). With or without /v1
BIOMNI_LLM_PROXY_API_KEY=proxy_token         # The proxy's token, sent as Authorization: Bearer
BIOMNI_LLM_PROXY_SCHEMA=openai               # Default: openai (/v1/chat/completions); or anthropic (/v1/messages)

# LLM resilience
BIOMNI_LLM_MAX_RETRIES=3                     # Default: 3    (provider-SDK 429/5xx backoff)
BIOMNI_LLM_REQUEST_TIMEOUT=120               # Default: 120  (seconds per LLM HTTP call; none/0 disables)
BIOMNI_ENABLE_PROMPT_CACHING=true            # Default: true (Anthropic system-prompt cache; also
                                             #          through the LLM proxy with the anthropic schema)
BIOMNI_RUN_TIMEOUT_SECONDS=600               # Default: unset (total wall-clock budget per run; bounds
                                             #          the number of ReAct turns. Recommended for
                                             #          interactive/demo so slow queries fail fast.)

# Observability / telemetry (see ARCHITECTURE.md "Observability & Telemetry")
LOG_LEVEL=INFO                               # Default: INFO  (DEBUG for triage)
BIOMNI_LOG_FORMAT=json                        # Default: json (one JSON object/line); use "text" for dev
BIOMNI_ENABLE_LLM_TELEMETRY=true             # Default: false (library); true in the container image.
                                             #          Emits a per-run llm_usage token/cost event.
BIOMNI_RUN_HEARTBEAT_SECONDS=15              # Default: 15   (run-liveness heartbeat cadence; Chainlit)
BIOMNI_LOG_REDACT_EMAILS=false               # Default: false (opt-in scrub of e-mail-shaped PII in logs)
# Stamped on every log line, to tell services, builds and environments apart
OTEL_SERVICE_NAME=biomni-ad                  # Default: biomni (BIOMNI_SERVICE_NAME also read)
BIOMNI_VERSION=1.2.3                         # Default: the installed package version
BIOMNI_ENV=prod                              # Default: unset (DEPLOY_ENV, then ENVIRONMENT, also read)
BIOMNI_GIT_SHA=abc1234                       # Default: set by the image build (GIT_SHA also read);
BIOMNI_GIT_REF=feat/adworkbench              #          reported by /healthz and /readyz (GIT_REF also read)

# Workspace scanning (bounds every directory walk; see biomni/fs_scan.py)
BIOMNI_WORKSPACE_MAX_FILES=10000             # Default: 10000 (hard file cap per scan)
BIOMNI_WORKSPACE_SCAN_TIMEOUT_S=15           # Default: 15    (wall-clock budget per scan, seconds)
BIOMNI_WORKSPACE_SCAN_TTL_S=60               # Default: 60    (how long a scan result stays fresh)

# Output directory (see "Workspace scope and outputs" below)
BIOMNI_OUTPUT_ROOT=/data/outputs             # Default: unset (else <workspace>/biomni-outputs, else ./runs)
BIOMNI_DEFAULT_SCOPE_PATHS=studyA,studyB     # Default: unset (seed a starting data scope for new users)

# Per-user state: preferences, run records and chat history
BIOMNI_STATE_DIR=/data/state                 # Default: unset (else <workspace>/.biomni, else no persistence)
BIOMNI_PREFS_DIR=/data/state/prefs           # Default: $BIOMNI_STATE_DIR/prefs
BIOMNI_RUNS_STATE_DIR=/data/state/runs       # Default: $BIOMNI_STATE_DIR/runs
BIOMNI_RUN_STALE_AFTER_S=180                 # Default: 180   (heartbeat age after which a run is
                                             #          reported as interrupted, not running)
BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE=true      # Default: false (single-user mode: with no auth gateway
                                             #          every session is a separate anonymous user, so
                                             #          nothing saved can ever be read back. This makes
                                             #          all sessions one shared local user instead -
                                             #          for development and single-user deployments.)

# Chat history (the conversation list in the left sidebar)
BIOMNI_THREADS_DB_URL=postgresql+asyncpg://… # Default: sqlite at $BIOMNI_STATE_DIR/threads/threads.db
                                             #          (else <workspace>/.biomni/threads). Set this to
                                             #          share history across replicas. The SQLite schema
                                             #          is created automatically; any other database is
                                             #          expected to be migrated by an operator.
BIOMNI_THREAD_ELEMENT_MAX_BYTES=2097152      # Default: 2 MiB (per-attachment cap for archiving into the
                                             #          history database; larger files stay only in the
                                             #          run output directory)
CHAINLIT_AUTH_SECRET=…                       # Default: generated once and stored next to the thread
                                             #          database. Set explicitly for multi-replica
                                             #          deployments, or browser sessions break on
                                             #          rollout.

# Authentication gateway. Identity headers are IGNORED unless this is enabled:
# without a gateway stripping client-supplied copies, anyone could send
# `Ai-App-User-Context: sub=<someone else>` and read or overwrite that person's
# settings and run history. Enable it only when a gateway is actually in front.
BIOMNI_TRUST_AUTH_HEADERS=true                # Default: false (fails closed)

# Header names. GRIP's composite headers, Ai-App-User-Context (sub, email,
# given_name, family_name) and Ai-App-Workspace-Context (uuid), are read by
# default and need no setting. Change these two only for a gateway that sends
# the same key=value lists under other names:
BIOMNI_AUTH_USER_CONTEXT_HEADER=ai-app-user-context
BIOMNI_AUTH_WORKSPACE_CONTEXT_HEADER=ai-app-workspace-context

# Single-value headers, for gateways that send one field per header
# (oauth2-proxy, Envoy). Each takes a comma-separated list, tried before the
# defaults. They are read only for a request that carries neither context
# header: a gateway strips its own headers from what clients send, not every
# name another gateway might use, so reading them next to GRIP's would let a
# client fill in a field the gateway left out. Pointing one at a composite
# header has no effect, and the app logs a warning at startup saying so.
BIOMNI_AUTH_USER_ID_HEADER=x-auth-request-user-id
BIOMNI_AUTH_EMAIL_HEADER=x-auth-request-email
BIOMNI_AUTH_WORKSPACE_HEADER=x-workspace-id
```

## Workspace scope and outputs

A deployment mounts the user's workspace at `BIOMNI_USER_DATA_PATH`. Users pick
which folders of it the agent should use from the ⚙️ Settings panel; only the
selected folders are scanned and described to the model, so a workspace with
thousands of files costs no more than a small one.
With nothing selected, the workspace is advertised by top-level folder name only
(one directory listing) and the agent enumerates on demand.

**Output directory** resolves to the first writable candidate:

1. the user's setting in the ⚙️ Settings panel,
2. `BIOMNI_OUTPUT_ROOT`,
3. `<workspace>/biomni-outputs/` when the workspace mount is writable,
4. `./runs/` next to the process.

Only the last is container-local: results written there are lost when the pod
restarts, and the UI says so - unless a volume is mounted over that very path,
which the resolver checks against the mount table rather than assuming.
Set `BIOMNI_OUTPUT_ROOT` to a mounted volume, or make the workspace mount
read-write, to keep results.

"First **writable** candidate" is load-bearing, and the usual reason a
configured `BIOMNI_OUTPUT_ROOT` appears to be ignored is that it is not
writable by the image's non-root user (UID 57439).
Writability is tested with `os.access(W_OK|X_OK)` on the nearest existing
ancestor, so the check reflects the mount's reported ownership and mode.
When a candidate is skipped, `resolve_output_dir` logs a WARNING naming the
setting and the rejected path, and the UI reports it on the resolved location -
a silently working fallback is precisely the case where a mis-set variable would
otherwise go unnoticed.

Two mount-level causes account for nearly all of it:

- **Block volumes** (Azure Disk, EBS, any PVC the kubelet formats) mount as
  `root:root 0755`. Fix with `securityContext.fsGroup: 57439` on the pod.
- **SMB/NFS shares** (Azure Files) ignore `fsGroup` entirely. Fix with
  `mountOptions: [uid=57439, gid=57439, dir_mode=0770, file_mode=0770, mfsymlinks]`
  on the StorageClass or PV.

Changed mount options reach a running deployment only when the node mounts the share afresh.
A StorageClass's options are copied into a PersistentVolume when it is created, and the node keeps a share mounted while any pod there uses it.
So change the PersistentVolume's `spec.mountOptions` as well, then scale the deployment to zero and back up; [grip_deployment.md](grip_deployment.md#storage) has the details and a command to check what the container sees.
At startup the app logs a `storage_check` line for the workspace, the output location and `BIOMNI_STATE_DIR`, with the mount options in effect and, for any location it cannot use, the reason and the fix.

Files attached to a message with 📎 are copied into `<output dir>/uploads/`
under their original names, and it is that path the agent is given.
Chainlit's own copy stays where it puts it - a scratch tree it deletes when the
session ends and wipes on shutdown, with a UUID for a filename - so without this
an attachment would be unreachable by the user's next visit, and the agent would
never learn what the file was called.
An attachment keeps working when the output directory is not writable; it just
does not outlive the session.

**Preferences and run records** follow the same shape:
`BIOMNI_PREFS_DIR` / `BIOMNI_RUNS_STATE_DIR`, else `BIOMNI_STATE_DIR/{prefs,runs}`,
else `<workspace>/.biomni/` when writable, else no persistence (settings apply to
the session only).
Storing them under the workspace is the only option that survives the
application being deprovisioned without extra infrastructure, since it is the
user's own storage.

**Chat history** is stored the same way, in a database rather than files:
`BIOMNI_THREADS_DB_URL`, else `BIOMNI_STATE_DIR/threads/threads.db`, else
`<workspace>/.biomni/threads/threads.db`, else disabled.
Every conversation - the questions, the plans, the code steps and the answers -
is written as it happens, so a user who closes the tab finds the conversation
again in the left sidebar and can carry on in it.
Attachments up to `BIOMNI_THREAD_ELEMENT_MAX_BYTES` are archived with the
conversation; anything larger is left in the run's output directory only.

History is enabled only when the storage key is stable enough for a user to find
their own conversations again: behind a gateway (`BIOMNI_TRUST_AUTH_HEADERS`),
or in single-user mode (`BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE`).
Otherwise each page load is a new anonymous user, and the sidebar would fill
with threads nobody could reopen.

Closing the browser does not stop a run.
The work carries on server-side and keeps writing into its own conversation, so
reopening that conversation from the list on the left shows everything that
happened while nobody was watching.
If the run is still going when it is reopened, the rest of it streams into the
page, and Stop applies to the run itself rather than to the tab.

The limit is the process: a run is bound to the one executing it, so a restart
or a pod eviction ends it, and the run record says `interrupted` rather than
claiming a result nobody produced.
Detaching execution into a worker that outlives the request would be a
different piece of infrastructure; nothing here silently loses work today.

Preferences are keyed by the user id the authentication gateway asserts, scoped
by workspace id when one is supplied. Those headers are only believed when
`BIOMNI_TRUST_AUTH_HEADERS` is enabled - the app cannot tell a gateway-forwarded
header from a hand-crafted one, so it fails closed and treats every session as
anonymous until a deployment declares that a gateway is in front. Keys are slugged and hashed, so a header
value can never escape its directory, and an e-mail-only identity hashes to an
opaque key rather than writing the address into a filename. With no gateway in
front, the key is per-session and nothing persists across sessions.

File-backed preferences assume a single replica (or `ReadWriteMany` storage).
Check this before scaling the deployment out.

### Python Configuration

```python
import os

from biomni.config import default_config

# All available settings
default_config.path = os.path.expanduser("~/.biomni/data")  # "~" is not expanded for you
default_config.timeout_seconds = 600
default_config.llm = "claude-sonnet-4-5"
default_config.temperature = 0.7
default_config.use_tool_retriever = True
default_config.auto_network_limited_mode = True
default_config.source = None  # Auto-detected
default_config.base_url = None  # For custom models
default_config.api_key = None  # For custom models
```

## Deployment settings

These matter to whoever runs the app rather than to whoever writes an analysis.
All are optional; the defaults are what a single-user local run wants.

### Serving and access

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_DEMO_PASSWORD` | unset | Puts Chainlit's own login in front of the app: one shared password, any username, and the username becomes the identity - so each person gets a separate conversation history. The histories are separate, not private: anyone with the password can sign in under a username someone else used and see that history. For a deployment with **no** authentication gateway. Leave unset behind a gateway, where it would add a second and weaker door in front of a real one. |
| `BIOMNI_BIND_ADDRESS` | `0.0.0.0` | Address the compose deployment publishes on. Set to `127.0.0.1` when a TLS proxy sits in front, so the app is reachable only through it. Note a `ufw deny` will **not** close a published port - Docker writes iptables rules ahead of ufw's chain - so binding to loopback is the reliable way. |
| `BIOMNI_TRUST_AUTH_HEADERS` | off | Believe the gateway's identity headers. Fails closed: until this is set, headers are ignored and every session is anonymous. Turn it on in the same change that puts a gateway in front. |
| `BIOMNI_AUTH_ISSUER_HEADER` | see identity.py | Header carrying the OIDC issuer. When an issuer is present it is folded into the storage key, because a subject id is unique within a realm rather than globally. **Send it from the first deployment or not at all** - introducing it later re-keys every existing user, who then finds an empty history. |

### Model access

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_LLM_PROXY_URL` | unset | Routes every model call through a platform LLM proxy, such as GRIP's in front of Azure AI Foundry, ahead of every provider setting. Each request carries the proxy's token as a bearer token and the session's user and workspace ids as `X-User-Id` and `X-Workspace-Id`, and a call with no gateway identity behind it is not sent. See [grip_deployment.md](grip_deployment.md#llm-proxy). |
| `BIOMNI_LLM_PROXY_API_KEY` | unset | The proxy's token. `/readyz` reports not ready until it is set, and while `BIOMNI_TRUST_AUTH_HEADERS` is off, since the proxy then refuses every call. |
| `BIOMNI_LLM_PROXY_SCHEMA` | `openai` | The API to speak to the proxy: `openai` calls `/v1/chat/completions`, `anthropic` calls `/v1/messages`. |

### Storage

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_USER_DATA_PATH` | unset | The user's workspace. `/readyz` reports not ready until it exists. |
| `BIOMNI_DATA_LAKE_PATH` | inside the checkout or image | The reference datasets. Point it at a volume: inside the container they count against the node's ephemeral storage and are downloaded again after every restart. |
| `BIOMNI_STATE_DIR` | unset | Preferences, run records and the chat-history database. **The conversation list on the left needs this to be writable**; without it the app falls back to the workspace root, and if that is mounted read-only there is no thread list at all. Set by both the compose file and the Kubernetes manifest. |
| `BIOMNI_STATE_HOST_PATH` | `./state` | Host path compose mounts at `BIOMNI_STATE_DIR`. |
| `BIOMNI_OUTPUT_ROOT` | unset | Where run artifacts go. Declare it whenever outputs land on a volume: without it resolution falls through to a candidate flagged *ephemeral*, and the UI warns that results are lost on restart even when they are not. |

### Run artifacts

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_MAX_PACKAGE_MB` | `200` | A finished run is offered to the user as one downloadable archive. Runs larger than this are reported by size instead - a single run can write a multi-GB intermediate, and zipping that would stall the app and the browser. |
| `BIOMNI_MAX_PACKAGES` | `20` | How many archives to keep. Each roughly duplicates the run it came from and the run directories are never deleted, so without a bound the output volume fills about twice as fast. Archives are derived data; dropping one costs only regenerating it. |

### Execution and memory

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_MAX_REPL_SESSIONS` | `32` | How many chats' Python state is held in memory. Each retained session costs whatever that conversation loaded, so keep it well under the container's memory limit rather than raising it freely. |
| `BIOMNI_REPL_SESSION_TTL_SECONDS` | `21600` (6h) | How long a chat's state is protected from eviction regardless of how many other chats have run. Eviction is silent from the user's side - their next step fails with `NameError` for a frame they correctly believe they loaded - so idle sessions are dropped first and evicting a live one is logged as a warning. |
| `BIOMNI_MAX_OPEN_FIGURES` | `50` | Matplotlib figures left open across executions. Capture does not close them - doing so mid-`savefig` blanked the next save - but pyplot's figure registry is process-global, so the oldest are closed beyond this cap to stop every figure any chat ever drew staying resident. |
| `BIOMNI_ADVERTISE_ALL_LIBRARIES` | off | Skip the check that narrows the advertised library catalogue to what is installed. Normally the agent is only told about libraries it can actually import; set this for a deployment that installs more after the image is built. |
| `BIOMNI_SCRUB_ENV_EXTRA` / `BIOMNI_SCRUB_ENV_ALLOW` | unset | Add to, or exempt from, the credential-shaped names hidden from generated code while it runs. Both are read once at startup, so code running in the sandbox cannot re-arm them. |

### Literature search

| Variable | Default | What it does |
|---|---|---|
| `NCBI_API_KEY` | unset | An NCBI API key for PubMed searches. NCBI takes three requests a second from an IP without one and ten with one, and the app paces its requests to match, across every chat. Like the other credentials it is hidden from generated code, and it is sent in the request body, where an error message cannot quote it. |
| `NCBI_EMAIL` | unset | A contact address sent with each PubMed request, as NCBI's usage policy asks, so NCBI can get in touch about this deployment's traffic before blocking it. |

### Monitoring

| Variable | Default | What it does |
|---|---|---|
| `BIOMNI_STATUS_INACTIVITY_SECONDS` | `14400` (4h) | How long `GET /status` keeps reporting `active` after the last activity, with nobody connected. |
| `BIOMNI_STATUS_OPEN_SESSION_INACTIVITY_SECONDS` | `43200` (12h) | The same while a browser session is connected. Longer on purpose: a platform that reclaims pods on `active: false` would otherwise destroy a connected researcher's in-memory state. Never applied as shorter than the window above. |

## Important Notes

- **For pip-installed packages**: You can't edit the package files, but you can still use environment variables or modify `default_config` at runtime
- **Configuration consistency**: Database queries always use `default_config`, regardless of agent parameters
- **Priority order**: Direct params > Runtime config > Env vars > Defaults

## Troubleshooting

**Deployed on GRIP**:
- See the troubleshooting table in [grip_deployment.md](grip_deployment.md#troubleshooting)

**API Key Not Found**:
- Check `.env` file exists in your working directory
- Verify with: `echo $ANTHROPIC_API_KEY`

**Configuration Not Applied**:
- Changes to `default_config` only affect agents created after the change
- Direct parameters only affect that specific agent, not database queries

**Model Not Found**:
- Check spelling of model name
- For Azure, prefix with "azure-" (e.g., "azure-gpt-4o")
- Ensure you have the right API key for that provider

**Provider Collision Avoidance**:
- Keep unused provider keys empty in `.env` to avoid accidental routing
- For Azure Anthropic, set `LLM_SOURCE=Anthropic` and `BIOMNI_LLM` to your `DEPLOYMENT_NAME`
- For Azure OpenAI, set `LLM_SOURCE=AzureOpenAI` and `BIOMNI_LLM` to `azure-<deployment_name>`
