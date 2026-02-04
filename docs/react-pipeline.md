# ReAct pipeline (implemented)

> Updated: 2026-02-04  
> Scope: describes what is implemented in this repository (no roadmap claims).

Related:
- [`docs/architecture.md`](architecture.md)
- [`docs/agents.md`](agents.md)
- [`docs/tools.md`](tools.md)
- `src/abstractagent/adapters/react_runtime.py` (`create_react_workflow`)

This document explains the end-to-end pipeline of the **ReAct** workflow:
- how each iteration is prompted (`LLM_CALL`)
- how tool calls are queued and executed (`TOOL_CALLS` and runtime effects)
- where the durable scratchpad lives, and what gets injected back into prompts

## Separation of responsibilities

- **AbstractCore**: LLM calls and tool-call normalization across providers/models. AbstractAgent expects tool requests to arrive as structured `response["tool_calls"]`.
- **AbstractRuntime**: executes effects (`LLM_CALL`, `TOOL_CALLS`, `ASK_USER`, …), persists run vars, and records the durable ledger.
- **AbstractAgent (this repo)**: defines ReAct prompting/parsing logic and maps it onto runtime effects.

## Durable state used by ReAct (`RunState.vars`)

See `ensure_react_vars(...)` in `src/abstractagent/adapters/react_runtime.py`.

- `context.task` (string)
- `context.messages` (durable transcript, list of message dicts)
- `scratchpad.iteration` (int)
- `scratchpad.cycles` (list of per-iteration dicts with `thought/tool_calls/observations`)
- `_runtime.tool_specs`, `_runtime.toolset_id`, `_runtime.allowed_tools`, `_runtime.inbox`
- `_temp.llm_response`, `_temp.pending_tool_calls`, `_temp.tool_results`, `_temp.user_response`, `_temp.final_answer`
- `_limits.*` (iteration and message/token controls)

## Workflow graph (nodes + effects)

`create_react_workflow(...)` defines these nodes:

- `init` (pure): normalize/migrate vars, seed the initial user message, compute tool metadata
- `reason` → `EffectType.LLM_CALL`
- `parse` (pure): decide whether to act (tool calls) or finish
- `act` → `EffectType.TOOL_CALLS` or a runtime-native effect (ask_user/memory/vars/subworkflow)
- `observe` (pure): append tool observations to `context.messages` and to the current cycle
- `handle_user_response` (pure): append user response and continue
- `done` (terminal): return final output
- `max_iterations` (terminal): deterministic conclusion when the iteration budget is exhausted

## Cycle mechanics (Reason → Parse → Act → Observe)

### 1) Reason (`LLM_CALL`)

Implemented in `reason_node` (`src/abstractagent/adapters/react_runtime.py`):
- builds the base system prompt via `ReActLogic.build_request(...)` (`src/abstractagent/logic/react.py`)
- injects a bounded scratchpad view of recent cycles (`_render_cycles_for_system_prompt`)
- sends the durable transcript as provider-safe `messages` (`_sanitize_llm_messages`)
- includes tool schemas (`_runtime.tool_specs`) so the model can emit structured tool calls

### 2) Parse (tool calls vs final answer)

Implemented in `parse_node`:
- reads `response["content"]` and `response["tool_calls"]` via `ReActLogic.parse_response(...)`
- appends a new cycle entry to `scratchpad["cycles"]`:
  - `{"i": iteration, "thought": content, "tool_calls": [...], "observations": [...]}` (observations are filled in later)

Important behavior (code reality):
- When tool calls exist, the adapter stores the “thought” content in the scratchpad (cycle entry) and appends an assistant message with `tool_calls` metadata and empty content (OpenAI-compatible transcript).
- When tool calls do not exist, the adapter usually treats `content` as the final answer and moves to `done`.
  A followthrough heuristic can instead retry the loop when the message looks like “I will do X next” but emitted no tool calls
  (enabled by default; disable with `_runtime.check_plan=false`).

### 3) Act (runtime effects)

Implemented in `act_node`:
- maintains a durable queue under `_temp.pending_tool_calls`
- translates schema-only tools into runtime-native effects:
  - `ask_user` → `ASK_USER`
  - `recall_memory` → `MEMORY_QUERY`
  - `inspect_vars` → `VARS_QUERY`
  - `remember` → `MEMORY_TAG`
  - `remember_note` → `MEMORY_NOTE`
  - `compact_memory` → `MEMORY_COMPACT`
  - `delegate_agent` → `START_SUBWORKFLOW` (wrapped as a tool-style observation)
- `open_attachment` is also included as a tool schema (`OPEN_ATTACHMENT_TOOL` in `src/abstractagent/logic/builtins.py`),
  but it is executed as a runtime-owned tool by AbstractRuntime’s AbstractCore integration (see [`docs/tools.md`](tools.md)).
- batches regular tools into a single `TOOL_CALLS` effect (payload includes `allowed_tools`)

Loop guard:
- For side-effect tools (`write_file`, `edit_file`, `execute_command`), the adapter can detect and skip repeating identical tool calls after a success (see `parse_node`).

### 4) Observe (tool results → transcript + scratchpad)

Implemented in `observe_node`:
- appends each tool result to `context.messages` as `role="tool"` with metadata `{name, call_id, success}`
- writes the structured observation list into the current cycle’s `observations`

## Max-iterations conclusion

When the iteration cap is reached, `max_iterations_node` performs a tool-free conclusion pass:
- runs one last `LLM_CALL` with a “max iterations reached” directive and a bounded scratchpad view
- completes the run directly with:
  - `output = {answer, report, iterations, messages, scratchpad}`

## Known limitations (by design or not yet wired)

- ReAct does not parse tool calls out of assistant text. If your provider returns tool requests in `content`, ensure AbstractCore normalization produces structured `tool_calls`.
