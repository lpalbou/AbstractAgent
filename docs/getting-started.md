# Getting Started

This guide is the main entrypoint after [`README.md`](../README.md).

Next reading:
- [`docs/agents.md`](agents.md) (agent types + API)
- [`docs/api.md`](api.md) (public API surface)
- [`docs/tools.md`](tools.md) (default tools + allowlists)
- [`docs/persistence.md`](persistence.md) (pause/resume across restarts)
- [`docs/faq.md`](faq.md) (common questions)
- [`docs/architecture.md`](architecture.md) (diagrams + runtime model)

## Ecosystem context

AbstractAgent is part of the **AbstractFramework** ecosystem:
- AbstractFramework: https://github.com/lpalbou/AbstractFramework
- AbstractCore (providers + tool schemas): https://github.com/lpalbou/abstractcore
- AbstractRuntime (durable workflows + storage/ledger): https://github.com/lpalbou/abstractruntime

This repository documents **what is implemented in `abstractagent`** (agents + adapters + tools). For provider/runtime details,
refer to AbstractCore/AbstractRuntime documentation.

## Requirements

- Python `>=3.10` (see `pyproject.toml`)
- A configured LLM provider supported by `abstractcore` via `abstractruntime`

## Install

From PyPI:

```bash
pip install abstractagent
```

From source (development):

```bash
pip install -e .
```

With dev dependencies:

```bash
pip install -e ".[dev]"
```

Tip: the repository may be ahead of the latest published PyPI release. To verify what you installed:

```bash
python -c "import importlib.metadata as md; print(md.version('abstractagent'))"
```

## Quick start (ReAct)

```python
from abstractagent import create_react_agent

agent = create_react_agent(provider="ollama", model="qwen3:1.7b-q4_K_M")
agent.start("List the files in the current directory")
state = agent.run_to_completion()
print(state.output["answer"])
```

Where this goes in code:
- `create_react_agent(...)`: `src/abstractagent/agents/react.py`
- Local runtime factory: `abstractruntime.integrations.abstractcore.create_local_runtime`

## Pick an agent

- **ReAct** (`create_react_agent`): tool-first loop. Default tools: `abstractagent.tools.ALL_TOOLS`.
- **CodeAct** (`create_codeact_agent`): executes Python (tool call or fenced code). Default tools: `[execute_python]`.
- **MemAct**: adds runtime-owned Active Memory. Default tools: `abstractagent.tools.ALL_TOOLS`.
  Import: `from abstractagent.agents.memact import create_memact_agent` (MemAct is not re-exported at the package top-level; see
  [`docs/api.md`](api.md)).

If you want file/web/system tools in CodeAct, pass an explicit tool list:

```python
from abstractagent import create_codeact_agent
from abstractagent.tools import ALL_TOOLS

agent = create_codeact_agent(tools=ALL_TOOLS)
```

## Provider configuration (`llm_kwargs`)

Factory helpers accept `llm_kwargs` and forward them to `abstractruntime.integrations.abstractcore.create_local_runtime`,
which passes them to the underlying AbstractCore client.

Example (LMStudio/OpenAI-compatible server):

```python
from abstractagent import create_react_agent

agent = create_react_agent(
    provider="lmstudio",
    model="qwen/qwen3-next-80b",
    llm_kwargs={"base_url": "http://localhost:1234/v1"},
)
```

## Choose tools (allowlist)

You can restrict which tools may execute per run:

```python
agent = create_react_agent()
agent.start(
    "Search for TODOs and summarize them",
    allowed_tools=["list_files", "search_files", "read_file"],
)
state = agent.run_to_completion()
```

Tool details: [`docs/tools.md`](tools.md)

## Add a custom tool

```python
from abstractagent import create_react_agent
from abstractcore.tools import tool

@tool(name="my_tool", description="Example tool")
def my_tool(query: str) -> str:
    return f"Echo: {query}"

agent = create_react_agent(tools=[my_tool])
```

## Persistence (resume across restarts)

By default, the factory helpers use an in-memory runtime store, so `save_state()` cannot resume across process restarts.
For persistence, pass a persistent `RunStore` and `LedgerStore`.

```python
from abstractagent import create_react_agent
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

run_store = JsonFileRunStore(".runs")
ledger_store = JsonlLedgerStore(".runs")

agent = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent.start("Long running task")
agent.save_state("agent_state.json")

# ... later / after restart ...

agent2 = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent2.load_state("agent_state.json")
state = agent2.run_to_completion()
print(state.output["answer"])
```

More details: [`docs/persistence.md`](persistence.md)

## Known limitations / gotchas

- `open_attachment` is a runtime-owned tool (executed by the runtime’s AbstractCore integration). If you pass `allowed_tools`,
  include `"open_attachment"` (see [`docs/tools.md`](tools.md) and [`docs/faq.md`](faq.md)).
- `execute_python` uses a local subprocess with a timeout; it is not a hardened sandbox (`src/abstractagent/sandbox/local.py`).
- ReAct’s “plan/review” flags are stored in `_runtime` but not applied by the current ReAct adapter (`src/abstractagent/agents/react.py`, `src/abstractagent/adapters/react_runtime.py`).
