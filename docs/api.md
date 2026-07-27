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
- `provider`, `model`: optional since 2026-07-13 — when omitted, they resolve from
  AbstractCore config global defaults (`abstractcore --config`); when nothing is
  configured, a packaged fallback pair applies with a loud `#FALLBACK`
  `UserWarning` (`src/abstractagent/agents/defaults.py`). Explicit values win untouched.
- `llm_kwargs`: forwarded to the underlying AbstractCore client (e.g. base URL, timeouts,
  and on Ollama `num_ctx` — forwarded per-call since 2026-07-13; set it to the model's
  maximum window, see the context-overflow FAQ)
- `tools`: list of tool callables; defaults to `abstractagent.tools.ALL_TOOLS`
- `review_mode`: verifier round on final answers — default **on** since 2026-07-13
  (failures degrade to accept-with-`#FALLBACK`, never a dead run)
- `max_output_tokens`: output-token cap (the honest name; `max_tokens` is kept as a
  compat alias on ReAct and means the same knob there)
- `run_store`, `ledger_store`: pass persistent stores to enable resume across restarts

Per-run controls (via `ReactAgent.start(...)`):
- `allowed_tools=[...]`: allowlist of tool names (enforced by the runtime’s tool-calls handler)
- `temperature`, `seed`: sampling controls stored under `vars["_runtime"]`

Host-level `_runtime` slots (raw-workflow hosts; see `docs/skills-attachment.md`):
- `skills_block`: skills attachment slot (composed before `system_prompt_extra`)
- `system_prompt_extra`: behavioral-directive slot
- `delegate_substrates`: `{name: {provider, model, description?, thinking?}}`
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

Limits honesty (CodeAct and MemAct): `max_tokens` is the context ACCOUNTING
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
- `cancel(reason: str | None = None) -> RunState`

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

Budget-exhausted terminals (`outcome: iteration_budget`): ReAct runs a tool-free
conclusion call; CodeAct/MemAct answer with the agent's last assistant words
(MemAct prefers its held draft) and append a final assistant message marked
`kind=final_answer, budget_exhausted: true` — the `answer` is never a raw tool
observation.

Source of truth:
- ReAct: `src/abstractagent/adapters/react_runtime.py` (`done_node`, `max_iterations_node`)
- CodeAct: `src/abstractagent/adapters/codeact_runtime.py` (`done_node`, `max_iterations_node`)
- MemAct: `src/abstractagent/adapters/memact_runtime.py` (`done_node`, `max_iterations_node`)
