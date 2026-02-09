# Agents

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/tools.md`](tools.md)
- [`docs/persistence.md`](persistence.md)
- [`docs/architecture.md`](architecture.md)

AbstractAgent ships three agent patterns implemented as **(logic → runtime workflow → API wrapper)**:

| Agent | Best for | Key behavior | Entry points |
|---|---|---|---|
| **ReAct** | Tool-first workflows | Loops until the model returns **no tool calls** | `abstractagent.agents.react.ReactAgent`, `create_react_agent()` |
| **CodeAct** | Python-centric tasks | Treats fenced ` ```python ... ``` ` as `execute_python` | `abstractagent.agents.codeact.CodeActAgent`, `create_codeact_agent()` |
| **MemAct** | Memory-enhanced sessions | Uses runtime-owned `active_memory` blocks | `abstractagent.agents.memact.MemActAgent`, `create_memact_agent()` |

Note: `MemActAgent` / `create_memact_agent` are **not** re-exported at the package top-level. Import them from
`abstractagent.agents.memact` (see [`docs/api.md`](api.md)).

## ReAct (`ReactAgent`)

Files:
- Workflow: `src/abstractagent/adapters/react_runtime.py` (`create_react_workflow`)
- Logic: `src/abstractagent/logic/react.py` (`ReActLogic`)
- API: `src/abstractagent/agents/react.py` (`ReactAgent`, `create_react_agent`)

Typical usage:

```python
from abstractagent import create_react_agent

agent = create_react_agent()
agent.start("Search for TODOs and summarize what needs fixing")
state = agent.run_to_completion()
print(state.output["answer"])
```

Notes (code reality):
- ReAct persists its **loop trace** under `vars["scratchpad"]["cycles"]` (not as assistant “thought” messages).
- ReAct disables runtime-level trimming and sends the full `context.messages` window by default.
- `ReactAgent(..., plan_mode=..., review_mode=...)` stores flags under `vars["_runtime"]`, but the current ReAct adapter does not apply them.
- `create_react_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/react.py`).

## CodeAct (`CodeActAgent`)

Files:
- Workflow: `src/abstractagent/adapters/codeact_runtime.py` (`create_codeact_workflow`)
- Logic: `src/abstractagent/logic/codeact.py` (`CodeActLogic`)
- API: `src/abstractagent/agents/codeact.py` (`CodeActAgent`, `create_codeact_agent`)

Behavior highlights:
- If the model emits a fenced Python block, the adapter executes it via a `TOOL_CALLS` effect targeting `execute_python`.
- Optional `plan_mode` and `review_mode` are implemented for CodeAct in `src/abstractagent/adapters/codeact_runtime.py`.
- `create_codeact_agent(tools=None)` defaults to `[execute_python]` (`src/abstractagent/agents/codeact.py`).

## MemAct (`MemActAgent`) (advanced)

Files:
- Workflow: `src/abstractagent/adapters/memact_runtime.py` (`create_memact_workflow`)
- Logic: `src/abstractagent/logic/memact.py` (`MemActLogic`)
- API: `src/abstractagent/agents/memact.py` (`MemActAgent`, `create_memact_agent`)

MemAct relies on the runtime’s active memory subsystem:
- the workflow ensures memory exists (`abstractruntime.memory.active_memory.ensure_memact_memory`)
- memory blocks are injected into the system prompt and updated via structured steps
- `create_memact_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/memact.py`).

## Common API and output contract

All agents inherit `BaseAgent` (`src/abstractagent/agents/base.py`):
- `start(task, ...) -> run_id`
- `step() -> RunState`
- `run_to_completion() -> RunState`
- `attach(run_id)` / `save_state(path)` / `load_state(path)`
- `cancel()` / `get_ledger()` / `inject_message(...)`

On completion, the workflow returns `state.output` with (at minimum):
- `answer`: final assistant answer (string)
- `iterations`: how many loop iterations ran (int)
- `messages`: the durable conversation transcript (list of message dicts)

ReAct also returns:
- `report`: a deterministic “what happened” summary derived from the scratchpad
- `scratchpad`: the full scratchpad dict (including `cycles`)
