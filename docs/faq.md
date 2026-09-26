# FAQ

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/agents.md`](agents.md)
- [`docs/tools.md`](tools.md)
- [`docs/persistence.md`](persistence.md)
- [`docs/architecture.md`](architecture.md)

## What is AbstractAgent (vs AbstractRuntime / AbstractCore / AbstractFramework)?

- **AbstractAgent** (this package) provides agent *patterns* and workflows: ReAct / CodeAct / MemAct.
  See `src/abstractagent/agents/*` and `src/abstractagent/adapters/*_runtime.py`.
- **AbstractRuntime** executes durable workflows (`WorkflowSpec`) and persists run state + a ledger.
  AbstractAgent adapters emit runtime effects like `EffectType.LLM_CALL` and `EffectType.TOOL_CALLS`.
- **AbstractCore** defines tool schemas and performs provider/model integration for LLM calls and tool-call normalization.
- **AbstractFramework** is the ecosystem umbrella that groups these packages (overview): https://github.com/lpalbou/AbstractFramework
  - AbstractCore: https://github.com/lpalbou/abstractcore
  - AbstractRuntime: https://github.com/lpalbou/abstractruntime

Architecture overview: [`docs/architecture.md`](architecture.md)

## Which agent should I use?

- Use **ReAct** if you want a tool-first loop that decides when to call tools and stops when the model emits **no tool calls**.
  Entry points: `create_react_agent()` / `ReactAgent` in `src/abstractagent/agents/react.py`.
- Use **CodeAct** if the task is primarily Python-centric: code runs via `execute_python` tool calls; on prompted-tools models, fenced ` ```python ... ``` ` blocks are also extracted and executed (capability-conditional — see `codeact_fenced_fallback` in docs/agents.md).
  Entry points: `create_codeact_agent()` / `CodeActAgent` in `src/abstractagent/agents/codeact.py`.
- Use **MemAct** if you need runtime-owned “Active Memory” in addition to the transcript/tool loop.
  Entry points: `create_memact_agent()` / `MemActAgent` in `src/abstractagent/agents/memact.py`.

## How do I configure provider/model (and server base URL)?

All factory helpers accept `provider`, `model`, and `llm_kwargs` and forward them to
`abstractruntime.integrations.abstractcore.create_local_runtime` (see `src/abstractagent/agents/react.py`,
`src/abstractagent/agents/codeact.py`, `src/abstractagent/agents/memact.py`).

Example:

```python
from abstractagent import create_react_agent

agent = create_react_agent(
    provider="lmstudio",
    model="qwen/qwen3-next-80b",
    llm_kwargs={"base_url": "http://localhost:1234/v1"},
)
```

## How do I run an agent unattended (no human watching)?

Use the packaged recipe — it excludes `ask_user` (which would park the run on a
human wait) and sets the no-questions directive. Raw-workflow hosts (the `run_vars`
you pass to `runtime.start(...)`):

```python
from abstractagent.agents.unattended import unattended_runtime_overrides

run_vars["_runtime"].update(unattended_runtime_overrides(base_allowlist=my_tools))
```

Facade users get both halves through `start(...)`:

```python
from abstractagent.agents.unattended import UNATTENDED_DIRECTIVE, unattended_allowlist

agent.start(task, allowed_tools=unattended_allowlist(my_tools),
            system_prompt_extra=UNATTENDED_DIRECTIVE)
```

## Can a delegated sub-agent run on a different model?

Only if the HOST grants it: set `_runtime.delegate_substrates = {"name": {"provider": ..., "model": ..., "description": ..., "thinking": ...}}`
and the model may pass `substrate="name"` to `delegate_agent` (granted names +
descriptions are rendered into the system prompt so the model can see what it
may ask for). Unknown names fail as loud tool errors; raw provider/model strings
in tool args are structurally impossible (a model choosing its own substrate
would be self-escalation). The palette AUTHORITY never propagates to
grandchildren (each level needs its own grant), though the resolved
provider/model do cascade to the child's own delegations like any inherited
substrate. Without a palette, children inherit the parent's
provider/model/temperature/seed/thinking/output-cap.

## How do I add my own tools?

Pass tool callables to the factory helper or agent constructor.
Tools are ordinary Python functions decorated with `@tool` from `abstractcore.tools`.

See:
- `src/abstractagent/tools/__init__.py` (`ALL_TOOLS`)
- [`docs/tools.md`](tools.md)

## How do tool allowlists work?

All adapters maintain an effective allowlist under `vars["_runtime"]["allowed_tools"]` and include it in tool execution effects
as `payload["allowed_tools"]`.

Implementation (source of truth):
- ReAct: `src/abstractagent/adapters/react_runtime.py` (TOOL_CALLS payload includes `allowed_tools`)
- CodeAct: `src/abstractagent/adapters/codeact_runtime.py` (TOOL_CALLS payload includes `allowed_tools`, including inline code execution)
- MemAct: `src/abstractagent/adapters/memact_runtime.py` (TOOL_CALLS payload includes `allowed_tools`)

## How do I use `open_attachment` (attachments)?

`open_attachment` is a **runtime-owned** tool for reading session attachments with bounded output.

Source of truth:
- Tool schema: `OPEN_ATTACHMENT_TOOL` in `src/abstractagent/logic/builtins.py`
- Execution: `abstractruntime.integrations.abstractcore.session_attachments.execute_open_attachment`
  (via the runtime’s AbstractCore `TOOL_CALLS` handler)

Checklist:
- Use the factory helpers (`create_react_agent`, `create_codeact_agent`, `create_memact_agent`) or wire your `Runtime`
  with `abstractruntime.integrations.abstractcore.effect_handlers.build_effect_handlers` and an `ArtifactStore`.
- If you pass `allowed_tools=[...]`, include `"open_attachment"` or it will be blocked by the allowlist.
- Ensure the attachment exists for the current `session_id` (many hosts populate this from uploads; the runtime can also
  register some `read_file` outputs as attachments).

Tool-call shape (as the model would invoke it):
- `open_attachment(artifact_id="…", start_line=1, end_line=200)` (preferred)
- `open_attachment(handle="@path/to/file.ext", start_line=1, end_line=200)` (fallback)

Common errors:
- `ArtifactStore is not available` → your runtime has no `artifact_store` configured.
- `attachment not found` → the `artifact_id`/`handle` is wrong, or the attachment was not registered for this session.

Details: [`docs/tools.md`](tools.md)

## How does pause/resume work?

`BaseAgent.save_state(path)` saves a small JSON file with identifiers (run/workflow/actor/session).
The durable run state lives in the runtime `RunStore`, so persistence requires a persistent store.

Source of truth: `src/abstractagent/agents/base.py` (`save_state`, `load_state`, `attach`)

Guide: [`docs/persistence.md`](persistence.md)

## Why can’t I resume after restarting my process?

If your runtime uses an in-memory `RunStore`, there is no durable data to resume from.
This is enforced by `BaseAgent.save_state(...)` (it raises for `InMemoryRunStore`).

Use persistent stores, e.g.:
- `abstractruntime.storage.json_files.JsonFileRunStore`
- `abstractruntime.storage.json_files.JsonlLedgerStore`

Example: [`docs/getting-started.md`](getting-started.md)

## How do I control iterations / limits?

- Per-agent defaults are set at agent construction (`max_iterations`, etc.) and seeded into `vars["_limits"]` in `start(...)`.
  See: `src/abstractagent/agents/react.py`, `src/abstractagent/agents/codeact.py`, `src/abstractagent/agents/memact.py`.
- `ReactAgent` and `CodeActAgent` also expose `update_limits(...)` to change limits mid-run.

## What happens when a long run exceeds the model's context window?

**Nothing compacts or trims by default.** ReAct deliberately sends the full transcript
every cycle (that is its design; MemAct's memory blocks ride *on top of* the full transcript unless
spans are archived). When the transcript outgrows the model window:

- **OpenAI-compatible servers** (LM Studio, vLLM, …) return a deterministic 400 — the run FAILS
  loudly mid-loop.
- **Ollama** (the factory default) is the trap: its server default context (~4k) **silently
  truncates from the oldest content first** — which is the system prompt and tool instructions.
  The symptom is not an error but an inexplicably degraded agent (tool calls stop, retries churn,
  the run burns to `max_iterations`). Remedies, either lane:
  - per-call: `create_react_agent(..., llm_kwargs={"num_ctx": <model max>})` — AbstractCore
    forwards it to Ollama's `options.num_ctx` (invalid values raise loudly; absence sends
    nothing, so Modelfile defaults stand);
  - server-side: `OLLAMA_CONTEXT_LENGTH=<model max> ollama serve` or a Modelfile
    `PARAMETER num_ctx <model max>`.

  House rule either way: always the MODEL'S MAXIMUM available context unless you explicitly
  choose otherwise — any fixed number below the model's window is a hidden ceiling that
  silently breaks workflows needing more.

The loops emit a one-shot `context_warning` step (with `#FALLBACK` marker) when
estimated usage crosses `warn_tokens_pct` (default 80%) of the `max_tokens` accounting ceiling
(a second one fires if usage crosses the ceiling itself) — subscribe via `on_step`/hooks.
Scope: the accounting uses the SERVER-REPORTED input tokens of the last call, so on an
Ollama server that is already truncating, reported usage plateaus at the serving window and the
warning may never fire — it catches the approach on providers that report true prompt usage
against the model window (OpenAI-compatible servers), not the Ollama silent-truncation cliff
itself.

## How do I set temperature / seed?

All agents accept per-run sampling controls via `start(...)`:
- `ReactAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/react.py`)
- `CodeActAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/codeact.py`)
- `MemActAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/memact.py`)

Adapters merge these values into LLM params via `runtime_llm_params(...)` in `src/abstractagent/adapters/generation_params.py`.

## Why does the agent “retry” when the model says “I will do X” but calls no tools?

A reply that only announces tool use ("Let me verify …") or carries tool-call markup that could not run is not an
answer. ReAct, CodeAct and MemAct re-prompt once, quoting the model's reply verbatim; if the second reply fails the
same way, the step ends with a visible error and the reply is never published. The announcement check can be
disabled with `_runtime.check_plan=false`; the tool-markup check has its own switch, `_runtime.check_unrunnable_calls` (see [`docs/agents.md`](agents.md#replies-that-announce-tools-without-calling-them-all-three-loops)).

Source of truth: `src/abstractagent/adapters/announced_calls.py` and the `parse` nodes of the three adapters.

## Is `execute_python` safe?

`execute_python` runs code in a local subprocess with a timeout (development-only; not a hardened sandbox).

Source of truth:
- `src/abstractagent/tools/code_execution.py`
- `src/abstractagent/sandbox/local.py`

## Is there a CLI?

This repo installs `react-agent`, but it is deprecated and prints a migration hint.

Source of truth:
- `pyproject.toml` (`[project.scripts]`)
- `src/abstractagent/repl.py`

## Does a delegated sub-agent stream its replies?

Yes, when the parent run streams. `delegate_agent` children in ReAct, CodeAct and MemAct inherit the
parent's `_runtime.stream` (and `speculation`, `thinking`, sampling and the other per-run controls), so a
live view keeps receiving tokens while the agent delegates. An explicit `stream: False` is inherited too.
See [`docs/api.md`](api.md) and the delegation section of [`docs/architecture.md`](architecture.md).

## Where are error fixes?

Symptom-first fixes (for example “TOOL_CALLS requires a ToolExecutor”) live in
[`docs/troubleshooting.md`](troubleshooting.md).
