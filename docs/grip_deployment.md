# Deploying Biomni-AD on the AD Workbench (GRIP)

This guide is for the team that runs Biomni-AD as an AI App on the AD Workbench GRIP platform.
It follows the "AI Apps Onboarding V3" checklist: what Biomni-AD provides for each item, and what the deployment has to set.
[configuration.md](configuration.md) describes every setting; this page is the part that matters on GRIP.

## Checklist

| Onboarding item | Biomni-AD |
|---|---|
| Docker image | `ghcr.io/kaimen-inc/biomni-ad`, see [Image](#image) |
| Kubernetes manifest | [`deploy/k8s/biomni-ad.yaml`](../deploy/k8s/biomni-ad.yaml), with the changes in [Kubernetes manifest](#kubernetes-manifest), or [`docker-compose.yml`](../docker-compose.yml) for a single machine |
| Environment variables | [Settings](#settings) below, and all of them in [configuration.md](configuration.md) |
| Sample input data | Not required. A small synthetic table for a smoke test is in [`samples/`](../samples/README.md) |
| Minimum and recommended resources | [Resources](#resources) |
| User and workspace headers | `Ai-App-User-Context` and `Ai-App-Workspace-Context` are read exactly as the gateway sends them, see [Identity](#identity) |
| `GET /status` | Implemented, `schema_version: 1`, see [Status and health](#status-and-health) |
| Azure Foundry through the LLM proxy | OpenAI and Anthropic schemas, see [LLM proxy](#llm-proxy) |
| Clear mount points for input and output | [Storage](#storage) |
| Configuration bundled, overridable by environment | Defaults are built into the image, and every setting is an environment variable |
| Automated initialisation | Nothing to run: the app creates what it needs when it starts, and fetches reference data the first time a question needs it |

## Image

`ghcr.io/kaimen-inc/biomni-ad:<tag>`, where `<tag>` is one of:

- `sha-<commit>`, the commit's first seven hex digits: one per build, never moved.
  Deploy this one.
- `feat-adworkbench`: the latest build of the AD Workbench branch.
- `latest`: the latest build of the default branch.

The container runs as the non-root user `mambauser`, uid and gid 57439, and serves HTTP on port 8000.
It needs no init container and no setup step.
Logs are one JSON object per line on stdout.

## Settings

### Set these

| Variable | Value on GRIP | Why |
|---|---|---|
| `BIOMNI_TRUST_AUTH_HEADERS` | `true` | Declares that the platform's authentication gateway is in front, so its `Ai-App-*` headers can be believed. It is off by default, so that an app reachable without a gateway cannot be impersonated with a hand-written header. While it is off, every session is anonymous: nothing is saved, and every model call is refused, because the LLM proxy needs a user and a workspace for each call. `/readyz` reports not ready for that reason while the proxy is configured. |
| `BIOMNI_LLM_PROXY_URL` | the proxy's address, for example `https://<proxy-host>/v1` | Sends every model call through the platform's LLM proxy. The address works with or without the trailing `/v1`. |
| `BIOMNI_LLM_PROXY_API_KEY` | the proxy's API token, from a Secret | Sent as `Authorization: Bearer <token>` on every call. |
| `BIOMNI_LLM` | the name of the model the proxy serves | Sent as the `model` of every request. |
| `BIOMNI_LLM_PROXY_SCHEMA` | `openai` (the default) or `anthropic` | The API the proxy speaks for that model. |
| `BIOMNI_USER_DATA_PATH` | `/app/user-data` | The user's workspace. |
| `BIOMNI_OUTPUT_ROOT` | a directory on a writable volume, see [Where results go](#where-results-go) | Where each run's results are written. |
| `BIOMNI_STATE_DIR` | `/app/state` | Each user's settings, run records and chat history. |
| `BIOMNI_DATA_LAKE_PATH` | a directory on a volume, for example `/app/state/data_lake` | The reference datasets, downloaded the first time a question needs each one. Left unset they go inside the container, see [Resources](#resources). |
| `CHAINLIT_AUTH_SECRET` | a long random string, from a Secret | Signs the session cookie. With one replica it can be left unset: the app generates one and keeps it in `BIOMNI_STATE_DIR`. Set it before running more than one replica, or users are signed out whenever a request reaches a different pod. |

### Do not set these

| Variable | Why not |
|---|---|
| `BIOMNI_AUTH_USER_ID_HEADER`, `BIOMNI_AUTH_EMAIL_HEADER`, `BIOMNI_AUTH_WORKSPACE_HEADER` | They name headers that carry one plain value each, for gateways other than GRIP's. Pointed at `Ai-App-User-Context` or `Ai-App-Workspace-Context` they have no effect, and the app logs a warning at startup telling you to remove them. |
| `BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE` | For a deployment with no gateway: it makes every session one shared user. Remove it if it was set for testing before the gateway was in place. |
| `BIOMNI_DEMO_PASSWORD` | Puts a password login in front of the app, for a deployment with no gateway. |
| Provider keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `AZURE_*` | Unused while `BIOMNI_LLM_PROXY_URL` is set, since every model call goes to the proxy. |

## Kubernetes manifest

[`deploy/k8s/biomni-ad.yaml`](../deploy/k8s/biomni-ad.yaml) deploys the app with the volumes and paths described under [Storage](#storage): Azure Disks at `/app/state` and `/app/runs`, and an Azure Files share at `/app/user-data`.
It is a generic manifest.
It exposes the app through a plain Service with no gateway in front, so it leaves the gateway and the LLM proxy off.
On GRIP, change three things, each of which the manifest shows where it applies:

1. Set `BIOMNI_TRUST_AUTH_HEADERS` to `"true"`, and let traffic reach the Service only through the authentication gateway.
2. Set `BIOMNI_LLM_PROXY_URL`, `BIOMNI_LLM_PROXY_SCHEMA` and `BIOMNI_LLM`, and in the Secret replace `ANTHROPIC_API_KEY` with `BIOMNI_LLM_PROXY_API_KEY`.
3. Bind the `biomni-ad-workspace` claim to the workspace's own Azure Files share, through a PersistentVolume with the mount options in [Azure Files over SMB](#azure-files-over-smb-mount-options), instead of letting it provision a new share.

## Identity

The platform's authentication gateway sends two headers:

```text
Ai-App-User-Context: email=<EMAIL>,given_name=<given_name>,family_name=<family_name>,sub=<sub>
Ai-App-Workspace-Context: uuid=<uuid>
```

Biomni-AD reads both as they are, with no configuration beyond `BIOMNI_TRUST_AUTH_HEADERS=true`.

- `sub` identifies the user.
  Their settings, run records and chat history are stored under it, separately for each workspace `uuid`.
- `email`, `given_name` and `family_name` are only displayed.
- Values are read as CSV.
  A value containing a comma is quoted, either on its own (`family_name="Smith, Jr."`) or together with its key (`"family_name=Smith, Jr."`), and a quote inside a quoted value is doubled.
  A header with an unterminated quote, or with `sub`, `email`, `iss` or `uuid` given twice, is ignored whole, since valid CSV cannot produce it.
- Names sent as UTF-8, such as `João`, display correctly.
- An optional `iss` key, the OIDC issuer, becomes part of the storage key when present.
  Send it from the first deployment or never: adding it later gives every existing user a new, empty history.

The gateway must remove `Ai-App-User-Context` and `Ai-App-Workspace-Context` from what clients send, and set its own on every request it forwards.
Biomni-AD cannot tell a header the gateway set from one a client forged, which is why trusting them is a setting.
While a request carries either context header, Biomni-AD takes the identity from those headers alone.
It then ignores the single-value headers that other gateways use (`X-User-Id`, `X-Workspace-Id`, `X-Auth-Request-*` and similar), so a client cannot fill in a field the gateway left out.

A browser signs in once: Chainlit exchanges the gateway's headers for a signed session cookie, valid for 15 days.
If the gateway's headers later name a different user than the cookie, as when someone else signs in on the same browser, the cookie is refused and the page signs in again as the user the gateway names.
A cookie issued by an earlier version of Biomni-AD is refused the same way, once.

To check that users are recognised:

- At startup the `access_config` log line shows `trust_auth_headers=true` and the header names in use, followed by a warning for any setting that has no effect.
- Each chat logs a `workspace_session` line with `identity_source=headers`.
  Any other value means the session is not signed in through the gateway.
- In the app, the menu under the avatar at the top right shows the user's name.
  With `BIOMNI_TRUST_AUTH_HEADERS` off it shows `anonymous`, or `local` under `BIOMNI_ALLOW_ANONYMOUS_PERSISTENCE`.

If the gateway is ever reconfigured to send the same fields under other header names, `BIOMNI_AUTH_USER_CONTEXT_HEADER` and `BIOMNI_AUTH_WORKSPACE_CONTEXT_HEADER` are the settings to change.

## LLM proxy

With `BIOMNI_LLM_PROXY_URL` set, every model call Biomni-AD makes goes to the proxy: the agent's reasoning, its research plans, and the model calls some of its tools make.
Each request carries:

| Header | Value |
|---|---|
| `Authorization` | `Bearer <BIOMNI_LLM_PROXY_API_KEY>` |
| `X-User-Id` | `sub` from the session's `Ai-App-User-Context` |
| `X-Workspace-Id` | `uuid` from the session's `Ai-App-Workspace-Context` |

The ids are those of the chat the call is made for, attached as each request is built, so when several people use the app at once each is charged for their own calls.
That includes the calls made by the analysis code the agent writes, from the code step itself and from the `concurrent.futures` thread pools it starts.
A call with no gateway identity behind it is never sent: the user is told why, and the proxy never sees a request it would have to refuse.

| `BIOMNI_LLM_PROXY_SCHEMA` | Endpoint | Notes |
|---|---|---|
| `openai` | `POST <proxy>/v1/chat/completions` | For reasoning models (`gpt-5*` and the o-series: `o1`, `o3`, `o4-mini` and so on), `temperature` and `stop` are left out, since those models reject them. |
| `anthropic` | `POST <proxy>/v1/messages` | The token is sent as a bearer token, and no `x-api-key` header is sent. The system prompt carries a `cache_control` marker, so the provider can reuse it from one turn to the next. |

If the proxy refuses requests that carry `cache_control`, set `BIOMNI_ENABLE_PROMPT_CACHING=false`.
Biomni-AD does not use `/v1/embeddings`.
The one tool that would reach a model provider directly, the Claude web search, is withdrawn whenever the proxy is configured, and the agent searches with its other tools instead.

When the proxy refuses a call, the user reads why in plain words rather than a stack trace, for example:

- HTTP 429: "The language model usage limit has been reached (HTTP 429). The platform limits model usage per user and per workspace. Try again later."
- HTTP 403: "The platform's LLM proxy refused the request (HTTP 403)", followed by the proxy's own explanation.
- HTTP 401: the token was rejected, and the message names `BIOMNI_LLM_PROXY_API_KEY`.
- HTTP 404: the model is not served, and the message names `BIOMNI_LLM`.

The proxy's explanation is taken from its JSON error body (`error.message`, `error`, `message` or `detail`).
A body that is not JSON, such as a gateway's HTML error page, is left out.
A refusal ends the run where it is, with what it had produced so far, instead of the run carrying on without a model.

### Network access

Besides the proxy, the app makes outbound HTTPS requests for its data:

- `https://biomni-release.s3.amazonaws.com`, the public data lake, fetched one dataset at a time as questions need them;
- the hosts of its web and literature search, listed [below](#web-and-literature-search);
- public biomedical services its tools query, such as UniProt, Ensembl and the GWAS Catalog;
- the hosts in the Alzheimer's dataset catalogues AD1 draws on, chiefly NIAGADS (`st1.niagads.org`, `dss.niagads.org`) and Zenodo (`zenodo.org`), from which the agent's code downloads a dataset when a question needs it.

The analysis code the agent writes can also fetch other public URLs a question or a catalogue points it to.
An egress allowlist therefore limits which questions can be answered, and a question that needs a blocked host fails at that step with the connection error.

### Web and literature search

| Tool | Searches | Host |
|---|---|---|
| `search_google` | the web, read from Google's results page | `www.google.com` |
| `query_pubmed` | PubMed, through NCBI's E-utilities API | `eutils.ncbi.nlm.nih.gov` |
| `query_arxiv` | arXiv, through its API | `export.arxiv.org` |
| `query_scholar` | Google Scholar, read from its results page | `scholar.google.com` |

A search that cannot be made says so instead of coming back empty.
The agent reads, for example, "Google search is unavailable", the reason, and which tools to use instead.
The app also logs a warning naming the service and the reason:

```bash
kubectl logs deploy/biomni-ad | grep "search unavailable"
```

Two refusals are expected:

- Google answers automated searches with a page that holds no results, so `search_google` reports itself unavailable on every call.
- arXiv throttles a client that sends it many requests, answering HTTP 406 until it lets up.
  The app keeps to arXiv's limit of one request every three seconds across all chats.
  It retries a throttled or failed request for about half a minute, and if arXiv still refuses, leaves it alone for two minutes.

PubMed searches keep to NCBI's limit of three requests a second, and retry when NCBI is busy.
Set `NCBI_API_KEY` to raise the limit to ten, and `NCBI_EMAIL` to give NCBI a contact address for this deployment (see [configuration.md](configuration.md#literature-search)).

## Storage

The app runs as uid 57439 and uses these locations:

| Location | Set by | Holds | Needs |
|---|---|---|---|
| `/app/user-data` | `BIOMNI_USER_DATA_PATH` | the user's workspace: their input data | read, and write too if results are to go into it |
| a results directory | `BIOMNI_OUTPUT_ROOT` | a folder per run: figures, tables and the code that produced them | write |
| `/app/state` | `BIOMNI_STATE_DIR` | each user's settings, run records and chat history | write |
| a reference-data directory | `BIOMNI_DATA_LAKE_PATH` | reference datasets, downloaded the first time a question needs them | write; inside the container at `/app/data/biomni_data/data_lake` unless pointed at a volume |

On GRIP today, `/app/runs` and `/app/state` are Azure Disks, and `/app/user-data` is an Azure Files share mounted over SMB.
The two kinds of volume are given to uid 57439 in different ways.

### Azure Disk: `fsGroup`

A new disk mounts as `root:root 0755`.
`securityContext.fsGroup: 57439` on the pod, as in the manifest, makes the kubelet give it to the app's group.

### Azure Files over SMB: mount options

SMB has no Unix ownership of its own.
The owner and mode the container sees come only from the mount options, and `fsGroup` is ignored.
Mounted without them, the share appears as `root:root 0755`: readable, but not writable by uid 57439.
Set these on the StorageClass or the PersistentVolume:

```yaml
mountOptions:
  - uid=57439
  - gid=57439
  - dir_mode=0770
  - file_mode=0770
  - mfsymlinks
```

Two things decide whether changed options actually reach the container:

1. A StorageClass's options are copied into each PersistentVolume when it is created, so changing the StorageClass does not change a volume that already exists.
   Change the PersistentVolume's `spec.mountOptions` too.
2. Options take effect when the node mounts the share, and the node keeps it mounted while any pod there uses the volume, so a rolling restart can start the new pod on the old mount.
   Scale the deployment to zero, wait for its pods to be gone, then scale it back up.

A partly applied set of options is worse than none.
With `dir_mode=0770` in effect but not `uid=57439` and `gid=57439`, the share belongs to root and grants nothing to anyone else, so the app cannot even read the workspace.

A share mounted with `cifsacl` or `modefromsid` is different: the file server's ACLs decide the owner and mode, and those are what must grant uid 57439 access.

### Checking what the container sees

```bash
kubectl exec deploy/biomni-ad -- sh -c 'id; grep " /app/user-data " /proc/self/mounts; ls -ldn /app/user-data'
```

Mounted correctly, this shows uid 57439, mount options that include `uid=57439` and `gid=57439`, and the directory as `drwxrwx--- ... 57439 57439`.

The app makes the same check itself when it starts, and logs one `storage_check` line per location with the live mount options.
When it cannot read or write a location, the line says why and what to change:

```bash
kubectl logs deploy/biomni-ad | grep storage_check
```

For a share mounted without ownership options, the reason reads:

```text
/app/user-data is owned by uid 0, gid 0 with mode 0755 on an SMB share (/<share>). Its owner and mode come only from the mount options, which are uid=0,gid=0,dir_mode=0755 on the mount this container sees. Mount it with uid=57439,gid=57439,dir_mode=0770,file_mode=0770; new options only apply once the share is mounted afresh on the node
```

The share is named without its server, so the storage account does not appear in logs or in the app.
Users see the same explanation in the app: at the start of each chat when the workspace cannot be read, and in the Settings dialog when results cannot go where the deployment meant them to.

### Where results go

Results go to the first of these that the app can write to: the folder the user chose in Settings, `BIOMNI_OUTPUT_ROOT`, `<workspace>/biomni-outputs`, and `./runs` in the app's working directory.
A location that cannot be written to is passed over, and the app says so in the log and in Settings.

- `BIOMNI_OUTPUT_ROOT=/app/user-data/biomni-outputs` puts results in the workspace, next to the user's other files, once the share is writable by uid 57439.
- `BIOMNI_OUTPUT_ROOT=/app/runs`, on an Azure Disk, keeps results durable in the meantime.
  Users download each run's results from the app.

Avoid pointing `BIOMNI_OUTPUT_ROOT` at the workspace root itself, since every run's folder would then appear at the top level of the workspace.

## Status and health

| Endpoint | Answers |
|---|---|
| `GET /status` | The platform's activity check, below. Also usable as a health check. |
| `GET /healthz` | Liveness: the process is serving. |
| `GET /readyz` | Readiness: the workspace at `BIOMNI_USER_DATA_PATH` exists, and a model can be reached. With the proxy configured, it reports not ready (HTTP 503) until `BIOMNI_LLM_PROXY_API_KEY` is set and `BIOMNI_TRUST_AUTH_HEADERS` is on. The response body names the check that failed. |

`GET /status` returns:

```json
{"active": true, "last_activity": 1790440933, "schema_version": 1, "background_jobs": 1, "open_sessions": 1, "internal_processes": {"active_threads": 5, "database_connections": 3}}
```

- `active` is true while any run is in progress, however long it takes, and for 4 hours after the last user activity (`BIOMNI_STATUS_INACTIVITY_SECONDS`).
- While a browser session is connected, the idle window is 12 hours instead (`BIOMNI_STATUS_OPEN_SESSION_INACTIVITY_SECONDS`), because reclaiming the pod would discard the in-memory state of that user's analysis.
- Both windows fall inside the platform's range of 4 to 24 hours.
- Health checks and `/status` requests themselves never count as activity.
- `last_activity` is a Unix timestamp in seconds.

## Resources

| | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| Memory | 8 GiB | 16 GiB |
| GPU | none | none |
| Disk for reference data | 20 GiB | 20 GiB |

These come from a measured run of the "Multi-omics AD risk gene portrait" starter, a full AD1 analysis drawing on GWAS, eQTL, proteomic and CRISPR datasets: 11 minutes, 23 code steps, 31 model calls.

- **No GPU.**
  Every model runs behind the LLM proxy, and the analysis code the agent writes is ordinary CPU work: pandas, SciPy, R.
- **CPU.**
  The run spent most of its time waiting for the model, and used at most one core, during code steps.
  The minimum keeps a core free for the interface while a run computes; the recommendation lets several people's runs compute at once.
- **Memory.**
  The run peaked at about 1 GB.
  What an analysis loads matters more: most large reference datasets, such as GeneBass's variant tables, are 1 to 1.7 GB on disk and several GB once loaded, and each open chat keeps what it loaded in memory until the chat is evicted (`BIOMNI_MAX_REPL_SESSIONS`).
  The minimum fits one person working with those, the recommendation several.
  The exception is BindingDB (`BindingDB_All_202409.tsv`): 6.25 GB on disk and about 13 GiB once loaded whole, more than the minimum and most of the recommendation.
  An analysis that loads all of it can exhaust the pod's memory, and the container then restarts for everyone using it.
- **Disk.**
  Reference datasets are downloaded the first time a question needs them, about 16.5 GB if every one is used: 15 GB of Biomni's data lake, 6.25 GB of it a single BindingDB table, and 1.4 GB of AD1's catalogue data.
  By default they are written inside the container, where they count against the node's ephemeral storage and are downloaded again after every restart.
  Point `BIOMNI_DATA_LAKE_PATH` at a directory on a volume, as the manifest does.
  A download cut short, by a pod restart say, leaves no half-written dataset behind: the file is fetched again the next time a question needs it.

The same run sent about 800,000 input and 20,000 output tokens through the proxy, and at its busiest exceeded 100,000 input tokens a minute.
Per-user limits below that slow long runs down with retries, and eventually stop them with the usage-limit message above.

## Smoke test

After deploying, sign in through the platform and check, in order:

1. The menu under the avatar at the top right shows your name, and no warning appears when the chat opens.
2. Copy [`samples/synthetic_gwas_hits.csv`](../samples/synthetic_gwas_hits.csv) into the workspace and ask: "Plot the -log10 p-values in synthetic_gwas_hits.csv as a bar chart."
   Approve the plan it proposes.
   The run should read the file, draw the chart, and save it in a new run folder under `BIOMNI_OUTPUT_ROOT`.
3. The proxy's log shows each model call with the user's `sub` as `X-User-Id` and the workspace `uuid` as `X-Workspace-Id`.
4. Settings, reopened after a page reload, still show what was saved, and the conversation is listed on the left.

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| "The assistant cannot answer in this session" when a chat opens, and Settings say "Settings apply to this session only: sign-in details from the authentication gateway were received but ignored, because BIOMNI_TRUST_AUTH_HEADERS is not enabled" | Trust in the gateway's headers is off | Set `BIOMNI_TRUST_AUTH_HEADERS=true` |
| The page says "Login to access the app" and "Unable to sign in", and the log says "rejecting a session with no gateway identity headers (BIOMNI_TRUST_AUTH_HEADERS is on)" | The request reached the app without `Ai-App-User-Context`, with neither `sub` nor `email` in it, or with a header ignored as malformed (a warning just before says which) | Check that every route to the app passes through the gateway, and that the gateway writes valid CSV |
| "The assistant cannot answer in this session", and Settings say "Settings apply to this session only: the authentication gateway did not identify this session" | The chat connected without the gateway's headers, and the browser's sign-in could not stand in for them | Reload the page, which signs in again |
| "This browser is still signed in to Biomni as a different user than the one the platform reports" | Someone else used the app in this browser, and their session cookie is still there | Reload the page, which signs in as the user the gateway names |
| Startup warning "`BIOMNI_AUTH_..._HEADER=...` has no effect" | A single-value header setting points at a context header | Remove the variable |
| "This app cannot read your workspace" | The workspace share's mount options do not give uid 57439 access | [Azure Files over SMB](#azure-files-over-smb-mount-options) |
| Settings say "The configured location could not be used, so results go to `<directory>` instead" | `BIOMNI_OUTPUT_ROOT`, or the folder chosen in Settings, is not writable by uid 57439 | The reason shown names the fix; see [Storage](#storage) |
| The agent reports "Google search is unavailable", "arXiv search is unavailable" or "Google Scholar search is unavailable" | Google refuses automated searches; arXiv is throttling the app; Google Scholar blocked it | Nothing to fix in the deployment: the agent searches with the other tools. See [Web and literature search](#web-and-literature-search) |
| "The language model usage limit has been reached (HTTP 429)" | The platform's per-user or per-workspace limit | Wait, or raise the limit in the proxy |
| An HTTP 401, 403 or 404 from "the platform's LLM proxy" | The token, the workspace's access to the model, or the model name | `BIOMNI_LLM_PROXY_API_KEY`, the proxy's policy, or `BIOMNI_LLM` |
| `/readyz` returns 503 | See `checks` in the response body | Usually the workspace mount, `BIOMNI_LLM_PROXY_API_KEY`, or `BIOMNI_TRUST_AUTH_HEADERS` |
