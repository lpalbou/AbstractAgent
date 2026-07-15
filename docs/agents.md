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
| **CodeAct** | Python-centric tasks | Python via `execute_python`; fenced ` ```python ... ``` ` extracted on prompted-tools models | `abstractagent.agents.codeact.CodeActAgent`, `create_codeact_agent()` |
| **MemAct** (experimental) | Memory-enhanced sessions | Uses runtime-owned `active_memory` blocks | `abstractagent.agents.memact.MemActAgent`, `create_memact_agent()` |

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
- `ReactAgent(..., review_mode=...)` is honored and defaults **on** (since 2026-07-13; the 0027 containment
  met the recorded re-flip condition): the ReAct adapter runs a verifier round on final answers
  (`maybe_review`/`review` nodes), and a failed verifier call degrades to accepting the held answer with a
  loud `#FALLBACK` marker (never a dead run). Pass `review_mode=False` to opt out. `plan_mode` is stored
  under `vars["_runtime"]` but the ReAct adapter does not read it — plan management rides the
  always-available `update_plan` tool and the `check_plan` gate instead.
- CodeAct's fenced-code fallback is **capability-conditional** (since 2026-07-14, A1): the default is
  `not supports_native_tools` (the bit the runtime seeds per run from model capabilities) — native-tools
  models neither get taught the ```python fence nor have prose fences extracted (teaching a channel that
  competes with the trained tool_calls channel manufactured zero-tool fabrication); prompted models keep
  the fence taught + extracted. Explicit `_runtime.codeact_fenced_fallback` wins both ways; an absent
  capability bit fails safe (fence ON). One resolver (`CodeActLogic.fenced_fallback_enabled`) drives both
  the prompt line and the parser gate so they can never disagree.
- `create_react_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/react.py`).

## CodeAct (`CodeActAgent`)

Files:
- Workflow: `src/abstractagent/adapters/codeact_runtime.py` (`create_codeact_workflow`)
- Logic: `src/abstractagent/logic/codeact.py` (`CodeActLogic`)
- API: `src/abstractagent/agents/codeact.py` (`CodeActAgent`, `create_codeact_agent`)

Behavior highlights:
- If the model emits a fenced Python block AND the fenced channel is enabled (`codeact_fenced_fallback` — default: enabled only on prompted-tools models), the adapter executes it via a `TOOL_CALLS` effect targeting `execute_python`. Routed runs: the capability bit describes the model it was DERIVED from, not necessarily the run's `_runtime.model`. The facade stamps `_runtime.tool_support_model` at start; when the effective model differs from the stamp, the resolver distrusts the bit and fails toward fence ON (a prompted model without the fence completes silently with code-as-prose — the worse failure). Hosts that route runs to different models without that stamp should pin `codeact_fenced_fallback` explicitly per run.
- Optional `plan_mode` and `review_mode` are implemented for CodeAct in `src/abstractagent/adapters/codeact_runtime.py`.
- `create_codeact_agent(tools=None)` defaults to `[execute_python]` (`src/abstractagent/agents/codeact.py`).

## MemAct (`MemActAgent`) — EXPERIMENTAL

**Status: experimental.** MemAct is kept because it is a genuinely distinct
architecture (Letta-style per-turn memory rewriting; the memory machinery
lives runtime-side and is separable), but it has one opt-in consumer and less
production mileage than ReAct/CodeAct. It is deliberately NOT re-exported at
the package top level — the deep import is the experimental label made
structural. Expect its surface to move faster than the siblings'.

Files:
- Workflow: `src/abstractagent/adapters/memact_runtime.py` (`create_memact_workflow`)
- Logic: `src/abstractagent/logic/memact.py` (`MemActLogic`)
- API: `src/abstractagent/agents/memact.py` (`MemActAgent`, `create_memact_agent`)

MemAct relies on the runtime’s active memory subsystem:
- the workflow ensures memory exists (`abstractruntime.memory.active_memory.ensure_memact_memory`)
- memory blocks are injected into the system prompt and updated via structured steps
- `create_memact_agent(tools=None)` defaults to `abstractagent.tools.ALL_TOOLS` (`src/abstractagent/agents/memact.py`).
- MemAct has **no** plan or review nodes. `plan_mode`/`review_mode`/`review_max_rounds`
  are DEPRECATED-IGNORED for one release (they were accepted-and-ignored dead knobs;
  the 2026-07-13 hard removal broke shipped call sites, so a shim now warns loudly
  with `DeprecationWarning` on non-default values and ignores legacy defaults —
  removal lands next release). A MemAct verifier is a deliberate non-goal while
  MemAct stays experimental.

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
