# AbstractAgent — Architecture (Current)

> Updated: 2026-01-02  
> Scope: this describes **what is implemented today** in this monorepo (no “future” design claims).

AbstractAgent is the **agent-pattern library** of the AbstractFramework. It provides portable agent behaviors implemented as:
- a **pure logic layer** (prompting + parsing; no runtime imports)
- a **runtime adapter** that turns that logic into an `abstractruntime.WorkflowSpec`
- a small **Agent API** wrapper (`BaseAgent`) for start/step/resume/attach/cancel

Agents run on **AbstractRuntime** (durable execution + persistence) and use **AbstractCore** for tool schemas and tool-call parsing.

## Repository Layout

```
abstractagent/
  src/abstractagent/
    agents/         # BaseAgent + concrete agents (ReAct / CodeAct / MemAct)
    adapters/       # logic -> WorkflowSpec (effects + durable vars schema)
    logic/          # Pure prompting/parsing logic (no runtime imports)
    tools/          # Agent-facing tool bundles
    repl.py         # Small REPL helpers
```

## Core Data Model (RunState.vars conventions)

AbstractAgent follows the runtime conventions from `abstractruntime.core.vars`:
- `context`: user-facing context (`task`, conversation `messages`, optional extras)
- `scratchpad`: agent loop state (`iteration`, `used_tools`, etc.)
- `_runtime`: runtime/host-managed metadata (provider/model, tool specs, inbox)
- `_temp`: step-to-step ephemeral values (`llm_response`, `pending_tool_calls`, …)
- `_limits`: canonical resource limits (`max_iterations`, `max_tokens`, …)

Both ReAct and CodeAct store conversation history under `context["messages"]` as dicts:
- `{role, content, timestamp, metadata}` with stable `metadata["message_id"]` (generated in adapters)

## ReAct Agent (Implemented)

### Pieces
- API wrapper: `abstractagent/src/abstractagent/agents/react.py` (`ReactAgent`)
- Pure logic: `abstractagent/src/abstractagent/logic/react.py` (`ReActLogic`)
- Runtime workflow: `abstractagent/src/abstractagent/adapters/react_runtime.py` (`create_react_workflow`)

### Runtime workflow shape
`create_react_workflow(...)` builds a `WorkflowSpec` with nodes:
- `init` → normalize/migrate vars, seed messages, compute/store toolset metadata
- `reason` → `EffectType.LLM_CALL` with `{prompt, tools?, system_prompt?, provider?, model?, params}`
- `parse` → parse tool calls (native or fallback parse-from-content); decide next step
- `act` → either:
  - schema-only built-ins → `ASK_USER` / `MEMORY_QUERY` / `MEMORY_TAG` / `MEMORY_NOTE` / `MEMORY_COMPACT`, or
  - regular tool calls → `TOOL_CALLS`
- `observe` → append tool observations to `context.messages`, then loop
- `finalize` / `finalize_parse` → optional final synthesis pass if tools were used
- `done` / `max_iterations` → complete with `{answer, iterations, messages}`

### Tool schema, allowlists, and execution boundary
- Tool **schemas** are `abstractcore.tools.ToolDefinition` objects serialized with `to_dict()`.
- ReAct stores durable tool metadata under `run.vars["_runtime"]`:
  - `tool_specs`: list of tool schema dicts
  - `toolset_id`: SHA-256 hash of the ordered tool specs (audit/debug)
  - `allowed_tools`: allowlist of tool names (defaults to the workflow’s tool list; can be overridden durably)
- The adapter enforces `allowed_tools` before requesting `EffectType.TOOL_CALLS`.
- Tool **execution** is performed by the runtime’s configured `ToolExecutor` (e.g. `MappingToolExecutor`).

### Tool-call parsing robustness
`ReActLogic.parse_response(...)` parses:
1) native `response["tool_calls"]` when present, else
2) fallback parse-from-content using `abstractcore.tools.parser` (handles tags like `<|tool_call|>`).

This matters for OSS/local providers where tool calls may appear in the assistant `content` instead of structured fields.

## CodeAct Agent (Implemented)

### Pieces
- API wrapper: `abstractagent/src/abstractagent/agents/codeact.py` (`CodeActAgent`)
- Pure logic: `abstractagent/src/abstractagent/logic/codeact.py` (`CodeActLogic`)
- Runtime workflow: `abstractagent/src/abstractagent/adapters/codeact_runtime.py` (`create_codeact_workflow`)

CodeAct is ReAct-like, but the primary action is executing Python:
- the model can call tools like `execute_python`, or include fenced ` ```python ... ``` ` blocks
- the adapter translates code execution into `EffectType.TOOL_CALLS` with a single call to `execute_python`

It supports the same schema-only built-ins as ReAct (`ask_user`, `recall_memory`, `remember`, `remember_note`, `compact_memory`) via runtime effects.

Notes:
- `recall_memory` supports `scope` routing (`run|session|global|all`) for cross-subrun recall without extra host glue.
- `remember_note` supports `scope` routing (`run|session|global`) for durable note storage into session/global indexes.

## MemAct Agent (Implemented)

MemAct is a **memory-enhanced** agent (Letta-like) that uses a runtime-owned Active Memory system.

Key separation boundary:
- **ReAct/CodeAct** are conventional SOTA agents (chat history + tool loop).
- **MemAct** is the only agent that leverages `abstractruntime.memory.active_memory`.

### Pieces
- API wrapper: `abstractagent/src/abstractagent/agents/memact.py` (`MemActAgent`)
- Pure logic: `abstractagent/src/abstractagent/logic/memact.py` (`MemActLogic`)
- Runtime workflow: `abstractagent/src/abstractagent/adapters/memact_runtime.py` (`create_memact_workflow`)

### Runtime workflow shape (high level)
- `init` → seed `context.messages`, ensure `_runtime.active_memory`
- `reason` → `EffectType.LLM_CALL` with:
  - chat history (`context.messages`)
  - a **system prompt** containing memory blocks rendered by `render_memact_system_prompt(...)`
- `parse` → parse tool calls; route to `act/observe` loop or to `finalize`
- `finalize` → enforce a single structured JSON envelope (`response_schema`)
- `finalize_parse` → apply the envelope deterministically to `_runtime.active_memory` (`apply_memact_envelope`)
- `done` → append final answer to history and complete

MemAct’s memory blocks are updated by the model via the structured envelope; timestamps are runtime-owned.

## BaseAgent API (Durable lifecycle)

Implemented in `abstractagent/src/abstractagent/agents/base.py`:
- `start(task) -> run_id`: creates a new run using the agent’s `WorkflowSpec`
- `step() -> RunState`: advances one runtime step (`Runtime.tick(..., max_steps=1)`)
- `run_to_completion()`: ticks until the run completes or blocks
- `resume(response)`: resumes an `ASK_USER` wait with `payload={"response": ...}`
- `attach(run_id)`: attach to an existing run (workflow/actor/session validation)
- `save_state(path)` / `load_state(path)`: persist only the identifiers needed to re-attach (durable state remains in the RunStore)
- `cancel(reason)`: durable cancel via `Runtime.cancel_run(...)`
- `get_ledger()`: read the durable ledger via `Runtime.get_ledger(...)`
- `get_node_trace(s)`: convenience accessors over `vars["_runtime"]["node_traces"]` (runtime-owned)

### Message injection (inbox)
`BaseAgent.inject_message(...)` appends guidance into `vars["_runtime"]["inbox"]` and persists it via the configured RunStore.
The ReAct/CodeAct adapters read and clear the inbox at the start of each reasoning step.

## Tools (What ships with AbstractAgent)

`abstractagent/src/abstractagent/tools/__init__.py` exposes:
- canonical tools re-exported from `abstractcore.tools.common_tools` (file/web/system tools)
- agent-specific tools: `execute_python`, `self_improve`
- a default tool bundle `ALL_TOOLS`

## Integration with AbstractFlow

AbstractFlow’s visual **Agent** node delegates execution to AbstractAgent by starting a subworkflow:
- compilation: `abstractflow/abstractflow/compiler.py` emits `EffectType.START_SUBWORKFLOW`
- the subworkflow vars include:
  - `context.task` / `context.messages` (and extra context keys)
  - `_runtime.provider`, `_runtime.model`, `_runtime.system_prompt`, `_runtime.allowed_tools`
  - `_limits` copied from the parent run (with defaults filled)

This keeps “agent semantics” consistent across hosts (AbstractFlow UI, AbstractCode CLI, future hosts): durable state and the ledger remain runtime-owned.

## What AbstractAgent Owns vs Uses

**AbstractAgent owns**
- ReAct and CodeAct prompting/parsing logic
- runtime adapters that map those patterns onto AbstractRuntime effects
- an ergonomic API wrapper (`BaseAgent`) and a small tool bundle

**AbstractAgent uses**
- **AbstractRuntime**: durability, effects, waits, ledger, run control
- **AbstractCore**: tool schemas + tool-call parsing helpers
