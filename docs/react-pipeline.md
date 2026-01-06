# AbstractAgent — ReAct Pipeline (Implemented)

> Updated: 2025-12-29  
> Scope: describes what is implemented today (no “future” claims).

This document explains the full end-to-end pipeline of the **ReAct agent** in this monorepo:
- how each iteration is prompted (“Thought → Action → Observation”),
- how tool calls are generated and normalized,
- who executes tools and who persists the durable scratchpad,
- what is injected back into the next prompt to prevent repetition and give the agent autonomy over time.

It also cross-checks our design against common ReAct “scratchpad” best practices from public references.

## Responsibilities (must stay separated)

### AbstractCore (LLM + tool-call normalization)
- Owns **tool schema format**, **tool-call detection/parsing**, and **syntax normalization/rewriting** across providers and models.
- Emits structured tool calls in the LLM response (`tool_calls`) and a cleaned assistant `content`.
- AbstractAgent expects this normalization to be done upstream: it does **not** “re-parse assistant text” for tool calls.

### AbstractRuntime (execution + durability)
- Executes effects:
  - `EffectType.LLM_CALL` (calls AbstractCore),
  - `EffectType.TOOL_CALLS` (runs tools through the configured `ToolExecutor`),
  - plus “built-in” effects like `ASK_USER`, `MEMORY_QUERY`, `VARS_QUERY`, etc.
- Persists durable state (`run.vars`) and the durable execution ledger.
- Records **durable node traces** under `run.vars["_runtime"]["node_traces"]` (this is the persisted scratchpad source of truth).

### AbstractAgent (agent semantics)
- Defines the ReAct loop prompting/parsing logic (`logic/react.py`).
- Maps that logic onto AbstractRuntime effects (`adapters/react_runtime.py`).
- Does not store its own parallel scratchpad format; it consumes runtime state.

### Hosts (AbstractCode / AbstractFlow)
- Provide UX, choose providers/models/toolsets, and start/resume runs.
- Should not implement their own tool-call parsing; they should rely on AbstractCore + AbstractRuntime.

## Data model (RunState.vars)

ReAct follows the standard runtime namespaces:
- `context`: user-facing state
  - `context.task`: the user request
  - `context.messages`: transcript as a list of `{role, content, timestamp, metadata}`
- `scratchpad`: agent loop state (iteration counters, plan text, flags)
- `_runtime`: runtime-owned metadata + durable traces
  - `_runtime.tool_specs`, `_runtime.allowed_tools`, `_runtime.toolset_id`, `_runtime.inbox`, …
  - `_runtime.node_traces`: durable effect traces (the persisted scratchpad source)
- `_temp`: step-local values (`llm_response`, `pending_tool_calls`, `tool_results`, …)
- `_limits`: canonical limits (`max_iterations`, `max_tokens`, `max_history_messages`, …)

## ReAct cycle (Reason → Parse → Act → Observe)

The runtime workflow is created by:
- `abstractagent/src/abstractagent/agents/react.py` (`ReactAgent`)
- `abstractagent/src/abstractagent/adapters/react_runtime.py` (`create_react_workflow`)

At a high level, each iteration follows:

1) **Reason** (`EffectType.LLM_CALL`)
   - The adapter selects the active message window:
     - `ActiveContextPolicy.select_active_messages_for_llm_from_run(run)`
   - It builds the LLM request using the pure logic layer:
     - `ReActLogic.build_request(...)` (`abstractagent/src/abstractagent/logic/react.py`)
   - It calls AbstractCore with:
     - the **system prompt** (agent rules; stable across iterations)
     - the **prompt** (task + history + runtime scratchpad)
     - the serialized tool specs (`ToolDefinition.to_dict()`), filtered by allowlist
     - optional provider/model overrides from `_runtime`

2) **Parse** (pure; no effect)
   - The adapter reads the LLM response dict produced by AbstractCore.
   - It parses:
     - `content` (assistant text)
     - `tool_calls` (structured tool call requests)
   - Implementation:
     - `ReActLogic.parse_response(...)` (`abstractagent/src/abstractagent/logic/react.py`)
   - If there is assistant text, it is appended to `context.messages` as a normal assistant message.
   - If there are tool calls, they are moved into `_temp.pending_tool_calls`.

3) **Act** (`EffectType.TOOL_CALLS` or built-in effects)
   - The adapter enforces the durable allowlist (`_runtime.allowed_tools`) before executing anything.
   - Some “schema-only tools” are mapped to native runtime effects:
     - `ask_user` → `EffectType.ASK_USER`
     - `recall_memory` → `EffectType.MEMORY_QUERY`
     - `inspect_vars` → `EffectType.VARS_QUERY`
     - `remember` → `EffectType.MEMORY_TAG`
     - `compact_memory` → `EffectType.MEMORY_COMPACT`
   - All other tools are executed via `EffectType.TOOL_CALLS`.
   - Execution is performed by AbstractRuntime’s configured `ToolExecutor` (e.g. `MappingToolExecutor`).

4) **Observe** (pure; no effect)
   - The adapter reads `_temp.tool_results` and appends tool observations to `context.messages` as `role="tool"` messages.
   - Each tool observation is formatted by the logic layer:
     - `ReActLogic.format_observation(name, output, success)`
   - Tool messages store metadata (`name`, `call_id`, `success`) to keep them machine-readable for hosts and audits.
   - The loop returns to **Reason** until:
     - no tool calls are produced, or
     - max iterations is hit, or
     - an explicit wait occurs (`ASK_USER`).

### Finalization (optional synthesis pass)

If at least one tool was executed (`scratchpad.used_tools=True`), the adapter forces a final LLM pass:
- `finalize` (`EffectType.LLM_CALL`, tool-free)
- Goal: produce a clean user-facing answer instead of echoing tool transcript lines.

This finalization step is separate from the runtime-owned scratchpad and may apply its own prompt-size controls.

## How “Thoughts” are generated (and persisted)

ReAct’s “Thought” in this framework is **not a hidden chain-of-thought channel**.

- The model produces normal assistant text in `response["content"]`.
- The prompt instructs the model to:
  - write 1–3 short lines explaining what it will do before calling a tool
  - continue from tool results instead of repeating calls
- That assistant text is appended to `context.messages` as the assistant message for the iteration.

In other words: the “thought” is simply the assistant’s own stated reasoning/explanation, stored durably as part of the transcript.

## Tool-call generation and execution (who does what)

### 1) Tool-call generation (LLM)
The LLM is asked to use the available tool schemas and produce tool calls when needed.

### 2) Tool-call detection/normalization (AbstractCore)
AbstractCore returns a normalized response dict that includes:
- `content`: assistant text
- `tool_calls`: a structured list (name/arguments/call_id), independent of provider/model syntax

AbstractAgent expects `tool_calls` to be present when the model is requesting tools.

### 3) Tool execution (AbstractRuntime)
The adapter converts the tool calls into a `TOOL_CALLS` effect:
- payload: `{"tool_calls": [...], "allowed_tools": [...]}`  
and the runtime executes them through its configured `ToolExecutor`.

### 4) Tool results back to the agent
The runtime returns a JSON-safe tool results structure to the adapter (stored under `_temp.tool_results`).
The adapter then appends these results to `context.messages` as `role="tool"` observations.

## Scratchpad injection (runtime-owned; full, non-truncated entries)

### Why we need a scratchpad beyond “history”
Many ReAct implementations maintain an **agent scratchpad** (a running log of Thought/Action/Observation) because:
- the next reasoning step must be grounded in prior tool outputs,
- tool messages may be filtered/compacted by host policies over time,
- noisy tool outputs can cause the model to “forget” what happened and repeat actions.

### Our scratchpad source of truth
We inject a scratchpad into every ReAct prompt derived from:
- `run.vars["_runtime"]["node_traces"]` (durably persisted by AbstractRuntime)

This avoids any parallel scratchpad persistence formats owned by the agent or host.

### What is injected
Implementation: `ReActLogic._format_runtime_scratchpad(...)` (`abstractagent/src/abstractagent/logic/react.py`)

For each recorded `tool_calls` effect in the runtime traces, we inject a block that includes:
- **Thought**: the preceding completed `llm_call` content (as stored in the trace)
- **Action**: the tool call payload (`tool_calls`) as JSON
- **Observation**: the tool result returned by the runtime as JSON
- plus a timestamp + status header

Important properties:
- The injected scratchpad is **runtime-owned** and **durable** across resumes.
- The agent does **not** truncate scratchpad entries; it renders trace entries as-is.
- The only information loss is whatever a tool decides to return (tool-specific output policy).

### Tool messages in History are rendered as Observations
To reduce “tool syntax confusion”, we render `role="tool"` history messages as:
- `observation[tool_name] (success|error): ...`
instead of `tool: ...` transcript lines.

Implementation: `ReActLogic._format_history_message(...)` (`abstractagent/src/abstractagent/logic/react.py`)

## Best-practice cross-check (online references)

Common ReAct practice is to interleave **Thought → Action → Observation** and feed the running log back as the next-step scratchpad:
- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models” (2022): https://arxiv.org/abs/2210.03629
- Prompting Guide ReAct examples show explicit `Thought N`, `Action N`, `Observation N` loops: https://www.promptingguide.ai/techniques/react

LangChain’s classic agent scratchpad formatting mirrors this:
- It appends each tool action log, then `Observation: ...`, then a trailing `Thought: ` prefix so the model continues from the observation:
  - https://raw.githubusercontent.com/langchain-ai/langchain/7dec2d399b3e012136843d168f804e7c958bb4a7/libs/langchain/langchain/agents/format_scratchpad/log.py

**Conclusion:** Our ReAct implementation follows these widely used patterns:
- explicit “Observation” surfaces for tool outputs (both in history rendering and in the injected scratchpad)
- durable scratchpad that carries tool calls + tool results across iterations
- the loop structure is aligned with Thought/Action/Observation interleaving (even though “Thought” is plain assistant text, not hidden)
