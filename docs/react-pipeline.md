# ReAct pipeline (implemented)

> Updated: 2026-07-10  
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

### Embedding the cycle in a larger workflow (entity visits)

`create_react_workflow(final_next_node="...")` turns `done`/`max_iterations` into TURN
boundaries instead of run terminals: the final answer is persisted to the durable
transcript exactly as in standalone mode, the output dict is stashed at
`_temp.react_output`, and control hands off to the named seam node. An embedding
workflow (e.g. the entity visit TURN chain: RECALL → this cycle → ELECTIONS → COMMIT →
FORM → ANSWER → PARK) merges this adapter's nodes with its own seam nodes in one
`WorkflowSpec`. Per-turn re-entry goes through `reset_react_turn(run.vars)` (fresh
iteration budget + clean `_temp`; transcript and `scratchpad.cycles` untouched) and
enters at `reason` — never `init` (init seeds the task message; visit messages arrive
from the embedding workflow's wait/resume). Tests:
`tests/test_react_visit_composition.py`.

## Cycle mechanics (Reason → Parse → Act → Observe)

### 1) Reason (`LLM_CALL`)

Implemented in `reason_node` (`src/abstractagent/adapters/react_runtime.py`):
- builds the base system prompt via `ReActLogic.build_request(...)` (`src/abstractagent/logic/react.py`);
  the system prompt is byte-stable across iterations (prompt-prefix cache stability, 0212) —
  the per-cycle scratchpad rendering was removed; reasoning stays in the transcript instead
- volatile per-call state (`[loop]` counter, plan view) rides only the TRAILING message,
  merging into a trailing user message when adjacency requires it
- sends the durable transcript as provider-safe `messages` (`_sanitize_llm_messages`:
  orphan repair, adjacent-user merges, metadata dropped — only role/content/tool_call_id/tool_calls survive)
- includes tool schemas (`_runtime.tool_specs`) so the model can emit structured tool calls
- entity dress rides configuration: `_runtime.system_prompt` (prelude head, full replacement),
  `_runtime.allowed_tools` (deny-by-default when explicitly empty), `_limits.max_iterations`
  (per-turn budget), `_runtime.prompt_cache_binding` → LLM_CALL params (cache identity in the
  params lane, never prompt bytes). Tests: `tests/test_react_entity_dress_conformance.py`

### 2) Parse (tool calls vs final answer)

Implemented in `parse_node`:
- reads `response["content"]` and `response["tool_calls"]` via `ReActLogic.parse_response(...)`
- appends a new cycle entry to `scratchpad["cycles"]`:
  - `{"i": iteration, "thought": content, "tool_calls": [...], "observations": [...]}` (observations are filled in later)

Important behavior (code reality):
- When tool calls exist, the adapter appends an assistant message carrying BOTH the reasoning
  content and `tool_calls` (context fidelity, 0213) and stores the cycle entry in the scratchpad.
- When tool calls do not exist, the adapter usually treats `content` as the final answer and moves to `done`.
  A followthrough heuristic can instead retry the loop when the message looks like “I will do X next” but emitted no tool calls
  (enabled by default; disable with `_runtime.check_plan=false`).
- The followthrough heuristic is **fence-blind** (frozen visit seam spec §4 line 6): all prose
  checks run on a view with paired ```fenced blocks removed. Fenced content (code samples,
  entity election fences like ```diary/```feel) is quoted material, never an action commitment —
  a fences-only reply is a valid final answer, and election fences survive every final-answer
  path byte-identical and in order (the max-iterations last-resort strip removes tool-call
  markup spans only). Tests: `tests/test_react_election_fence_conformance.py`.

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

#### Act-only tool results (entity visit seam)

Tools declared `act_only` (core's first-class `ToolDefinition.act_only` field — e.g. entity
diary reads under the G1 privacy rule "the book's words never rest outside the book") render
differently in `observe_node`:

- the durable tool message content is an **act-frame reference** — one exact JSON object with
  a lone `$act_only` top-level key (deterministic sorted-key serialization; detected by parse,
  never regex). The runtime's LLM_CALL wrapper dereferences it at send time into a wire copy
  (`abstractruntime/identity/act_only.py`); the words never rest in `context.messages`,
  `scratchpad.cycles`, LLM payloads, or the emit lane.
- the **ref shape is reserved for dereferenceable frames** (`entry_id` present). Act-only
  frames that reference nothing (failures, suppression records) render as labeled non-ref
  record text — inert to the dereference pass, so a historical record can never wedge the run.
- handler-authored refs (`{"$act_only": {...}}`-shaped outputs) are honored regardless of
  local declarations; a declared act-only tool whose handler returns raw output gets loud
  `#FALLBACK` suppression, never silent rendering.

Contract source: frozen visit seam spec (`a2a/threads/0013-visit-seam-spec`, v2 §2 + addenda).
Tests: `tests/test_react_act_only_observation.py` (includes an interop pin that runs the
runtime's shipped parser over this adapter's rendered bytes).

## Max-iterations conclusion

When the iteration cap is reached, `max_iterations_node` performs a tool-free conclusion pass:
- runs one last `LLM_CALL` with a “max iterations reached” directive and a bounded scratchpad view
- completes the run directly with:
  - `output = {answer, report, iterations, messages, scratchpad}`

## Known limitations (by design or not yet wired)

- ReAct does not parse tool calls out of assistant text. If your provider returns tool requests in `content`, ensure AbstractCore normalization produces structured `tool_calls`.
