# AbstractAgent

Agent patterns (ReAct / CodeAct / MemAct) built on **AbstractRuntime** (durable execution) and **AbstractCore** (tools + LLM integration).

AbstractAgent is part of the **AbstractFramework** ecosystem:
- AbstractFramework (ecosystem overview): https://github.com/lpalbou/AbstractFramework
- AbstractCore (providers + tool schemas): https://github.com/lpalbou/abstractcore
- AbstractRuntime (durable workflows + storage/ledger): https://github.com/lpalbou/abstractruntime

Start here: [`docs/getting-started.md`](docs/getting-started.md) (then [`docs/README.md`](docs/README.md) for the full index)

## How it fits (high level)

```mermaid
flowchart LR
  Host[Your app / service] --> Agent[AbstractAgent<br/>ReAct / CodeAct / MemAct]
  Agent --> RT[AbstractRuntime<br/>WorkflowSpec + Effects]
  RT --> Core[AbstractCore<br/>LLM + tool-call normalization]
  RT --> Stores[RunStore + LedgerStore]
  Agent --> Tools[Tool callables<br/>(abstractcore common_tools + agent tools)]
```

## Documentation

- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- API reference: [`docs/api.md`](docs/api.md)
- FAQ / troubleshooting: [`docs/faq.md`](docs/faq.md)
- Architecture (diagrams): [`docs/architecture.md`](docs/architecture.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)
- Acknowledgements: [`ACKNOWLEDMENTS.md`](ACKNOWLEDMENTS.md)

## What you get

- **ReAct**: tool-first Reason → Act → Observe loop
- **CodeAct**: executes Python (tool call or fenced ` ```python``` ` blocks)
- **MemAct**: memory-enhanced agent using runtime-owned Active Memory
- **Durable runs**: pause/resume via `run_id` + runtime stores
- **Tool control**: explicit tool bundles + per-run allowlists
- **Observability**: durable ledger of LLM calls, tool calls, and waits

Where this lives in code (source of truth):
- Agents: `src/abstractagent/agents/*`
- Workflows/adapters: `src/abstractagent/adapters/*_runtime.py`
- Prompting/parsing logic (runtime-agnostic): `src/abstractagent/logic/*`
- Default tool bundle: `src/abstractagent/tools/__init__.py`

## Requirements

- Python `>=3.10` (see `pyproject.toml`)

## Installation

From source (development):

```bash
pip install -e .
```

With dev dependencies:

```bash
pip install -e ".[dev]"
```

From PyPI:

```bash
pip install abstractagent
```

Native Python hardware profile cascades are available for deployment manifests:
`abstractagent[apple]`, `abstractagent[gpu]`, `abstractagent[all-apple]`, and
`abstractagent[all-gpu]`. These delegate to the matching AbstractCore and
AbstractRuntime profiles; AbstractAgent itself remains provider/runtime agnostic.

Note: the repository may be ahead of the latest published PyPI release. To verify what you installed:

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

## Persistence (resume across restarts)

By default, the factory helpers use an in-memory runtime store. For resume across process restarts,
pass a persistent `RunStore`/`LedgerStore` (example below uses JSON files).

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

More details: [`docs/persistence.md`](docs/persistence.md)

## CLI

This repository still installs a `react-agent` entrypoint, but it is **deprecated** and only prints a migration hint
(see `src/abstractagent/repl.py` and `pyproject.toml`).

Interactive UX lives in **AbstractCode**.

## License

MIT (see `LICENSE`).
