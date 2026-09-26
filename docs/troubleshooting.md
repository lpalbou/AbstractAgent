# Troubleshooting

Symptom-first fixes for common AbstractAgent problems. For concepts and limits, see
[`docs/faq.md`](faq.md); for setup, see [`docs/getting-started.md`](getting-started.md).

## “TOOL_CALLS requires a ToolExecutor”

**Cause:** `EffectType.TOOL_CALLS` is executed by the runtime’s configured `ToolExecutor`, and the
`Runtime` you created has none.

**Fix:** pass a tool executor when you build the `Runtime` yourself (recommended:
`MappingToolExecutor.from_tools([...])`). The factory helpers `create_react_agent(...)` and
`create_codeact_agent(...)` already wire `MappingToolExecutor.from_tools(...)` for you
(`src/abstractagent/agents/react.py`, `src/abstractagent/agents/codeact.py`).

## `save_state(...)` raises, or a run cannot be resumed after a restart

**Cause:** the runtime uses an in-memory `RunStore`, so there is no durable state to resume from.
`BaseAgent.save_state(...)` refuses to save a handle for `InMemoryRunStore`.

**Fix:** use persistent stores, for example
`abstractruntime.storage.json_files.JsonFileRunStore` and
`abstractruntime.storage.json_files.JsonlLedgerStore`. See [`docs/persistence.md`](persistence.md).

## On Ollama, tool calls stop and the run burns through `max_iterations`

**Cause:** the Ollama server's default context window (about 4k tokens) silently truncates the
oldest content first, which is the system prompt and the tool instructions.

**Fix:** raise the context window to the model's maximum, per call with
`create_react_agent(..., llm_kwargs={"num_ctx": <model max>})` or server-side with
`OLLAMA_CONTEXT_LENGTH=<model max> ollama serve`. See
[the context-window FAQ](faq.md#what-happens-when-a-long-run-exceeds-the-models-context-window).

## A long run fails with HTTP 400 on an OpenAI-compatible server

**Cause:** the loops send the full transcript every cycle; when it outgrows the model window, the
server rejects the call.

**Fix:** use a model with a larger window, or keep tasks shorter. Subscribe to the
`context_warning` step (`on_step` or loop hooks) to see the approach before the limit; see
[`docs/faq.md`](faq.md#what-happens-when-a-long-run-exceeds-the-models-context-window) and
[`docs/hooks.md`](hooks.md).

## Streaming is on, but no live tokens arrive (or a delegated child stops streaming)

**Check:**
- `_runtime.stream` is the boolean `True`. Strings such as `"true"` are treated as unset.
- Your host installs a live-delta sink and runs an AbstractRuntime release with live token
  streaming (AbstractGateway does this for its clients).
- A delegated child inherits the parent's value: if the parent set `stream: False`, children stay
  off too.

See `stream` under host-level `_runtime` slots in [`docs/api.md`](api.md).

## The agent retries when the model says “I will do X” but calls no tools

This is the announced-tool check, not a failure: the model is shown its reply verbatim and asked once to call the
tools or answer. If a run then stops with `stop_reason.code = "no_tool_call"`, the model announced tools (or wrote
tool calls that could not run) twice. Retry, check that the tool it asked for is enabled, or try another model or
thinking off. Disable the announcement check with `_runtime.check_plan=false`
(see [`docs/agents.md`](agents.md#replies-that-announce-tools-without-calling-them-all-three-loops)).
