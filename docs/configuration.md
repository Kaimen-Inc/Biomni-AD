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
BIOMNI_PATH=/path/to/data                   # Default: ./data
BIOMNI_TIMEOUT_SECONDS=1200                 # Default: 600  (per code/tool step)
BIOMNI_LLM=model_name                        # Default: claude-sonnet-4-20250514
BIOMNI_TEMPERATURE=0.7                      # Default: 0.7
BIOMNI_USE_TOOL_RETRIEVER=true             # Default: true
BIOMNI_AUTO_NETWORK_LIMITED_MODE=true      # Default: true
LLM_SOURCE=Anthropic                        # Preferred source selector
BIOMNI_SOURCE=Anthropic                     # Also supported (backward compatibility)
BIOMNI_CUSTOM_BASE_URL=http://localhost:8000/v1
BIOMNI_CUSTOM_API_KEY=custom_key

# LLM resilience
BIOMNI_LLM_MAX_RETRIES=3                     # Default: 3    (provider-SDK 429/5xx backoff)
BIOMNI_LLM_REQUEST_TIMEOUT=120               # Default: 120  (seconds per LLM HTTP call; none/0 disables)
BIOMNI_ENABLE_PROMPT_CACHING=true            # Default: true (Anthropic system-prompt cache)
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
# `x-user-id: <someone else>` and read or overwrite that person's settings and
# run history. Enable it only when a gateway is actually in front.
BIOMNI_TRUST_AUTH_HEADERS=true                # Default: false (fails closed)

# Header names (each accepts a comma-separated list, additive to the defaults)
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
restarts, and the UI says so. Set `BIOMNI_OUTPUT_ROOT` to a mounted volume, or
make the workspace mount read-write, to keep results.

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
from biomni.config import default_config

# All available settings
default_config.path = "./data"
default_config.timeout_seconds = 600
default_config.llm = "claude-sonnet-4-20250514"
default_config.temperature = 0.7
default_config.use_tool_retriever = True
default_config.auto_network_limited_mode = True
default_config.source = None  # Auto-detected
default_config.base_url = None  # For custom models
default_config.api_key = None  # For custom models
```

## Important Notes

- **For pip-installed packages**: You can't edit the package files, but you can still use environment variables or modify `default_config` at runtime
- **Configuration consistency**: Database queries always use `default_config`, regardless of agent parameters
- **Priority order**: Direct params > Runtime config > Env vars > Defaults

## Troubleshooting

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
