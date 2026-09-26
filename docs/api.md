# API Reference (public)

This document describes the public API surface intended for external users.

Ecosystem context:
- AbstractFramework: https://github.com/lpalbou/AbstractFramework
- AbstractCore (providers + tool schemas): https://github.com/lpalbou/abstractcore
- AbstractRuntime (durable workflows + storage/ledger): https://github.com/lpalbou/abstractruntime

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/agents.md`](agents.md)
- [`docs/tools.md`](tools.md)
- `src/abstractagent/__init__.py` (top-level exports)

## Top-level imports (`import abstractagent`)

The package re-exports a subset of agents and tools from `src/abstractagent/__init__.py`.

### Agents

```python
from abstractagent import (
    BaseAgent,
    ReactAgent,
    create_react_workflow,
    create_react_agent,
    CodeActAgent,
    create_codeact_workflow,
    create_codeact_agent,
)
```

Notes:
- **MemAct is experimental** and deliberately not re-exported at the package
  top-level — the deep import (`abstractagent.agents.memact`, see below) is
  the experimental label made structural. Distinct architecture worth keeping;
  expect its surface to move faster than ReAct/CodeAct.

### Loop hooks

```python
from abstractagent import LoopHooks, HookEvent, DEFAULT_EVENT_MAP
```

All three agents and workflow factories accept `hooks=LoopHooks(...)` for
listen/steer/capture on the running loop. See [`docs/hooks.md`](hooks.md)
for the event vocabulary and contract.

### Tools

```python
from abstractagent import (
    ALL_TOOLS,
    list_files,
    read_file,
    search_files,
    write_file,
    edit_file,
    execute_command,
    web_search,
    fetch_url,
    execute_python,
    self_improve,
)
```

The canonical tool bundle is `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/tools/__init__.py`).

Schema-only built-ins (`abstractagent.logic.builtins`) are tool *definitions* the adapters
map to runtime effects or that a host executes — they are deliberately NOT in `ALL_TOOLS`:

- `ASK_USER_TOOL` — mapped to a durable wait (pause/resume).
- `OPEN_ATTACHMENT_TOOL` — host-owned attachment resolution.
- `READ_SKILL_TOOL` — progressive skill disclosure: the host attaches a skills INDEX via the
  `skills_block` prompt slot and maps `read_skill` in its tool executor to its skill shelf.
  Expose it only with both halves wired (index + executor) — either half alone manufactures
  dead calls. Facade path: `agent.logic.add_tools([READ_SKILL_TOOL])` post-construction.
  Contract: `docs/skills-attachment.md`.

### Task reset (queued work)

```python
from abstractagent.agents.task_reset import reset_react_task

next_vars = reset_react_task(
    finished_state.vars,
    "the next task",
    carry_messages=True,          # default: transcript carries over (session continuity)
    handoff_note="prior task: done — output at ...",  # optional durable user message
)
run_id = runtime.start(workflow=wf, vars=next_vars)
```

A pure BETWEEN-RUNS transform for hosts that feed an agent task after task (work doors,
schedulers). Per-task state resets (scratchpad, `_temp`, iteration/token counters, the
steering inbox — stale guidance must not amend the wrong task); door-owned state persists
in content (`_runtime` grants/slots/substrate, deep-copied so the next run can never rewrite
the finished run's record); messages carry by default, `carry_messages=False` isolates.
The input dict is never mutated. Never apply it to a live run.

## Factories (recommended entrypoint)

### `create_react_agent(...) -> ReactAgent`

File: `src/abstractagent/agents/react.py`

Key parameters:
- `provider`, `model`: optional — when omitted, they resolve from
  AbstractCore config global defaults (`abstractcore --config`); when nothing is
  configured, a packaged fallback pair applies with a loud `#FALLBACK`
  `UserWarning` (`src/abstractagent/agents/defaults.py`). Explicit values win untouched.
- `llm_kwargs`: forwarded to the underlying AbstractCore client (e.g. base URL, timeouts,
  and on Ollama `num_ctx` — forwarded per call; set it to the model's
  maximum window, see the context-overflow FAQ)
- `tools`: list of tool callables; defaults to `abstractagent.tools.ALL_TOOLS`
- `review_mode`: verifier round on final answers — default **on**
  (failures degrade to accept-with-`#FALLBACK`, never a dead run)
- `max_output_tokens`: output-token cap (`max_tokens` is kept as a compatibility
  alias on ReAct and means the same knob there)
- `run_store`, `ledger_store`: pass persistent stores to enable resume across restarts

Per-run controls (via `ReactAgent.start(...)`):
- `allowed_tools=[...]`: allowlist of tool names (enforced by the runtime’s tool-calls handler)
- `temperature`, `seed`: sampling controls stored under `vars["_runtime"]`

Host-level `_runtime` slots (raw-workflow hosts; see `docs/skills-attachment.md`):
- `skills_block`: skills attachment slot (composed before `system_prompt_extra`)
- `system_prompt_extra`: behavioral-directive slot
- `speculation`: speculative decoding (MTP) control forwarded on every LLM call —
  `False` (Off), `True`, or a Core dict such as `{"mode": "native_mtp", "num_draft_tokens": 2}`;
  unset follows Core's route default. Delegated children inherit it.
- `stream`: live token streaming for the run's LLM calls. `True` asks the runtime
  to stream every LLM call to its live-delta sink (the value must be the boolean
  `True`; anything else is treated as unset); `False` turns streaming off
  explicitly; unset leaves the host's default. Delegated children in ReAct, CodeAct
  and MemAct inherit the value, `False` included, so a streamed run keeps
  streaming when it delegates. Live token deltas need an AbstractRuntime release
  with live token streaming and a host that installs a delta sink (for example
  AbstractGateway).
- `delegate_substrates`: `{name: {provider, model, description?, thinking?, speculation?}}`
  palette for the `delegate_agent` tool's optional `substrate` argument
  (unknown names fail as loud tool errors; the palette never propagates to
  grandchildren). A profile may pin the child's reasoning effort via
  `thinking` (same values as `_runtime.thinking`); absent means the child
  inherits the parent's value, invalid warns and inherits.

### `create_codeact_agent(...) -> CodeActAgent`

File: `src/abstractagent/agents/codeact.py`

Defaults:
- `tools=None` defaults to `[execute_python]`
- `provider`/`model` resolve like ReAct's (config defaults, loud fallback)

Limits (CodeAct and MemAct): `max_tokens` is the context ACCOUNTING
ceiling (drives `warn_tokens_pct` / the `context_warning` emit), not an output
cap — use `max_output_tokens` for output capping.

Per-run controls (via `CodeActAgent.start(...)`):
- `allowed_tools=[...]`
- `temperature`, `seed`

### `create_memact_agent(...) -> MemActAgent`

File: `src/abstractagent/agents/memact.py`

Defaults:
- `tools=None` defaults to `abstractagent.tools.ALL_TOOLS`

Import:

```python
from abstractagent.agents.memact import MemActAgent, create_memact_agent
```

## Agent API (common methods)

All agents inherit `BaseAgent` (`src/abstractagent/agents/base.py`).

### Lifecycle

- `start(task: str, **kwargs) -> str`: starts a new run and returns a `run_id`
- `step() -> RunState`: advances one runtime step
- `run_to_completion() -> RunState`: ticks until the run completes or waits
- `cancel(reason: str | None = None) -> RunState`: calls `Runtime.cancel_run`; with
  AbstractRuntime 0.4.32 this also stops the run's in-flight LLM/tool effect, and the
  interrupted step is recorded in the ledger as `StepStatus.CANCELLED`

### Pause / resume

- `attach(run_id: str) -> RunState`: re-attach to an existing run
- `save_state(path: str) -> None`: store identifiers needed to re-attach later
- `load_state(path: str) -> RunState | None`: load the state file and attach
- `resume(response: str) -> RunState`: resume an `ASK_USER` wait

Persistence details: [`docs/persistence.md`](persistence.md)

### Observability

- `get_ledger() -> list`: durable effect ledger entries for the current run
- `get_node_traces() -> dict`: runtime-owned traces (when available)
- `inject_message(message: str) -> None`: add guidance for the next reasoning step

## Outputs

When a run completes, workflows return `state.output` with at least:
- `answer` (string)
- `iterations` (int)
- `messages` (list of message dicts)
- `outcome` (string): `final_answer` | `iteration_budget` — machine-readable terminal
  cause (same vocabulary as the `turn_end` hook event)
- the loop's skip flag, always present as a bool at BOTH terminals:
  `review_skipped` (ReAct/CodeAct) or `finalize_skipped` (MemAct)

ReAct also returns:
- `report` (string)
- `scratchpad` (dict, including `cycles`)
- `stop_reason` (dict: `code`, `finished`, `budget_exhausted`, `label`, `headline`, `remedy`) and `notices`
  (list of `{code, severity, text}`). `stop_reason.code = "no_tool_call"` means the model announced tool use,
  or wrote tool calls that could not run, twice (once after a re-prompt); `output.no_tool_call_stop` then names
  the reason. See [`docs/agents.md`](agents.md#replies-that-announce-tools-without-calling-them-all-three-loops).

All three loops add `no_tool_call_stop` (`reason`, `detail`, `cycle`, `error`) and `stop_reason` (`code: "no_tool_call"`) when a step ended that way.

Budget-exhausted terminals (`outcome: iteration_budget`): ReAct runs a tool-free
conclusion call; CodeAct/MemAct answer with the agent's last assistant words
(MemAct prefers its held draft) and append a final assistant message marked
`kind=final_answer, budget_exhausted: true` — the `answer` is never a raw tool
observation.

Source of truth:
- ReAct: `src/abstractagent/adapters/react_runtime.py` (`done_node`, `max_iterations_node`)
- CodeAct: `src/abstractagent/adapters/codeact_runtime.py` (`done_node`, `max_iterations_node`)
- MemAct: `src/abstractagent/adapters/memact_runtime.py` (`done_node`, `max_iterations_node`)

## Native-loop bundles (gateway catalog)

Manifest-only workflow bundles declare a native loop factory instead of
embedded VisualFlow JSON. Gateway loads them via
`abstractagent.adapters.materialize_native_loop_spec` and registers the
returned `WorkflowSpec`.

### Pack audit

```python
from abstractagent.adapters.native_loop_registry import (
    audit_native_loop_manifest,
    materialize_native_loop_spec,
)

audit_native_loop_manifest(
    metadata=manifest["metadata"],
    interfaces=manifest["interfaces"],
    flows=manifest.get("flows") or [],
)
spec = materialize_native_loop_spec(
    metadata["native_loop_factory"],
    bundle_ref="react-agent@0.1.0",
    entrypoint="react",
    metadata=manifest["metadata"],
)
```

Audit rules:
- `interfaces` must include `abstractcode.agent.v1`
- `metadata` keys must be in the allowed set (`loop_family`,
  `native_loop_factory`, `max_iterations_default`, `headless_policy`,
  `allowed_tools`)
- shipped `flows` must not contain `wait_until` poller nodes (status-poller
  subflows defeat native termination)

Supported factories: `react`, `codeact`, `memact`.

### Thin-client discovery contract

After gateway restart/reload, `GET /api/gateway/bundles` exposes
`metadata.native_loop_factory` and `interfaces` on each bundle item.

Thin clients (e.g. abstractcode-tui) starting a native-loop run must:

1. Select the bundle entrypoint whose `loop_family` matches the desired
   agent loop (`react`, `codeact`, `memact`).
2. Treat the **root run** as the answer source — not a
   `visual_react_agent_*` child subrun. Native loops terminate at flow end
   with no status-poller subflow.
3. Pass the user prompt directly in `input_data` — native entrypoints have
   no VisualFlow input schema (`input_schema` 404 is expected until a stub
   lands; not blocking headless benches).

Workflow id for materialized specs:
`{bundle_ref}:{entrypoint}` (e.g. `react-agent@0.1.0:react`).

### Operator pack and gateway reload

Pack manifest-only `.flow` bundles from this repo (gateway serves them; no
VisualFlow JSON inside):

```bash
cd abstractagent
python scripts/build_native_loop_bundle.py \
  --output-dir /tmp/native-loop-bundles \
  --version 0.1.0 \
  --bundle-id react-agent react \
  --bundle-id codeact-agent codeact \
  --bundle-id memact-agent memact
```

Each file is `{bundle_id}@{version}.flow` containing only `manifest.json`.
Omit repeated `--bundle-id` flags to pack the script defaults
(`codeact-agent`, `memact-agent`).

Upload to a running gateway and reload the catalog (replace host/port/token):

```bash
for f in /tmp/native-loop-bundles/*.flow; do
  curl -sS -X POST "http://127.0.0.1:8080/api/gateway/bundles/upload?reload=true" \
    -H "Authorization: Bearer $ABSTRACTGATEWAY_AUTH_TOKEN" \
    -F "file=@${f}"
done
```

Verify discovery after reload:

```bash
curl -sS "http://127.0.0.1:8080/api/gateway/bundles" \
  -H "Authorization: Bearer $ABSTRACTGATEWAY_AUTH_TOKEN" \
  | jq '.[] | select(.metadata.native_loop_factory) | {bundle_id, entrypoints, native_loop_factory: .metadata.native_loop_factory}'
```

Requirements:
- The gateway process must import `abstractagent` (editable install in the
  gateway venv is the usual dev setup). Loader calls
  `abstractagent.materialize_native_loop_spec` at catalog load time.
- Dev-tree smoke before upload:

```bash
python scripts/verify_native_loop_gateway_import.py
```

- If a bundle was already registered under the same version, tombstone the old
  catalog version or bump `--version` before re-upload.
- A full gateway restart also picks up bundles on disk; `reload=true` avoids
  bouncing a shared `:8080` stack when only the catalog changed.
