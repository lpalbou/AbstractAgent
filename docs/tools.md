# Tools

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/agents.md`](agents.md)
- [`docs/architecture.md`](architecture.md)

AbstractAgent relies on **AbstractCore** tool definitions and **AbstractRuntime** tool execution.

## What ships by default

`abstractagent.tools.ALL_TOOLS` is the default tool bundle (see `src/abstractagent/tools/__init__.py`).

It contains:
- File tools (from `abstractcore.tools.common_tools`): `list_files`, `read_file`, `search_files`, `write_file`, `edit_file`, …
- Web tools (from `abstractcore.tools.common_tools`): `web_search`, `fetch_url`
- System tools (from `abstractcore.tools.common_tools`): `execute_command`
- Agent-specific tools (implemented here):
  - `execute_python` (`src/abstractagent/tools/code_execution.py`)
  - `self_improve` (`src/abstractagent/tools/self_improve.py`)

## Adding your own tools

Tools are plain Python callables decorated with `@tool` from `abstractcore.tools`.

```python
from abstractagent import create_react_agent
from abstractcore.tools import tool

@tool(name="my_tool", description="Example tool")
def my_tool(query: str) -> str:
    return f"Echo: {query}"

agent = create_react_agent(tools=[my_tool])
```

## Tool allowlists (`allowed_tools`)

All workflows compute an effective allowlist and keep it under `vars["_runtime"]["allowed_tools"]`.

- If you pass `allowed_tools=[...]` to `agent.start(...)`, the workflow uses that allowlist (filtered to known tools).
- Tool execution payloads include `allowed_tools` so the runtime/tool-executor can enforce it.

Implementation pointers:
- ReAct allowlist + payload wiring: `src/abstractagent/adapters/react_runtime.py` (`_effective_allowlist`, `EffectType.TOOL_CALLS` payload)
- CodeAct allowlist + payload wiring: `src/abstractagent/adapters/codeact_runtime.py`
- MemAct allowlist + payload wiring: `src/abstractagent/adapters/memact_runtime.py`

## Built-in “schema-only” tools (runtime effects)

Agents always include a set of tool *definitions* that are not implemented as Python callables in this package.
The adapters translate these tool calls into **native runtime effects**:

- `ask_user` → `EffectType.ASK_USER`
- `recall_memory` → `EffectType.MEMORY_QUERY`
- `inspect_vars` → `EffectType.VARS_QUERY`
- `remember` → `EffectType.MEMORY_TAG`
- `remember_note` → `EffectType.MEMORY_NOTE`
- `compact_memory` → `EffectType.MEMORY_COMPACT`
- `delegate_agent` → `EffectType.START_SUBWORKFLOW`

Tool definitions live in `src/abstractagent/logic/builtins.py`.

### Important limitation: `open_attachment`

`open_attachment` is currently included as a tool *definition* (`src/abstractagent/logic/builtins.py`) so models can reference it,
but it is **not mapped to a runtime effect** by the adapters today.

If you need attachment reading, you must:
- provide your own callable tool named `open_attachment`, or
- add adapter support in this repo (mapping it to a runtime/host capability).

## Safety notes

- `execute_python` runs a local subprocess with a timeout (`src/abstractagent/sandbox/local.py`); it is not a hardened sandbox.
- `execute_command` can run arbitrary shell commands (depending on your tool executor / host policy).
