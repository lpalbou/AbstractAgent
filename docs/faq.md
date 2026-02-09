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
- Use **CodeAct** if the task is primarily Python-centric: fenced ` ```python ... ``` ` blocks are executed via `execute_python`.
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

## How do I set temperature / seed?

All agents accept per-run sampling controls via `start(...)`:
- `ReactAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/react.py`)
- `CodeActAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/codeact.py`)
- `MemActAgent.start(..., temperature=..., seed=...)` (`src/abstractagent/agents/memact.py`)

Adapters merge these values into LLM params via `runtime_llm_params(...)` in `src/abstractagent/adapters/generation_params.py`.

## Why does ReAct “retry” when the model says “I will do X” but calls no tools?

The ReAct adapter includes a followthrough heuristic that retries the loop when the model claims it will take an action
but emits no tool calls. It is enabled by default and can be disabled with `_runtime.check_plan=false`.

Source of truth: `src/abstractagent/adapters/react_runtime.py` (`_looks_like_deferred_action`, `check_plan` behavior)

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

## Troubleshooting: “TOOL_CALLS requires a ToolExecutor”

`EffectType.TOOL_CALLS` is executed by the runtime’s configured `ToolExecutor`. If you create a `Runtime` manually,
ensure you provide a tool executor (recommended: `MappingToolExecutor.from_tools([...])`).

Factory helpers already wire this for you:
- `create_react_agent(...)` and `create_codeact_agent(...)` use `MappingToolExecutor.from_tools(...)`
  (see `src/abstractagent/agents/react.py`, `src/abstractagent/agents/codeact.py`).
