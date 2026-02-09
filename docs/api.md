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
- **MemAct** is not currently re-exported at the package top-level.
  Import it from `abstractagent.agents.memact` (see below).

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

## Factories (recommended entrypoint)

### `create_react_agent(...) -> ReactAgent`

File: `src/abstractagent/agents/react.py`

Key parameters:
- `provider`, `model`: passed to `abstractruntime.integrations.abstractcore.create_local_runtime`
- `llm_kwargs`: forwarded to the underlying AbstractCore client (e.g. base URL, timeouts)
- `tools`: list of tool callables; defaults to `abstractagent.tools.ALL_TOOLS`
- `run_store`, `ledger_store`: pass persistent stores to enable resume across restarts

Per-run controls (via `ReactAgent.start(...)`):
- `allowed_tools=[...]`: allowlist of tool names (enforced by the runtime’s tool-calls handler)
- `temperature`, `seed`: sampling controls stored under `vars["_runtime"]`

### `create_codeact_agent(...) -> CodeActAgent`

File: `src/abstractagent/agents/codeact.py`

Defaults:
- `tools=None` defaults to `[execute_python]`

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

ReAct also returns:
- `report` (string)
- `scratchpad` (dict, including `cycles`)

Source of truth:
- ReAct: `src/abstractagent/adapters/react_runtime.py` (`done_node`, `max_iterations_node`)
- CodeAct: `src/abstractagent/adapters/codeact_runtime.py` (`done_node`, `max_iterations_node`)
- MemAct: `src/abstractagent/adapters/memact_runtime.py` (`done_node`, `max_iterations_node`)
