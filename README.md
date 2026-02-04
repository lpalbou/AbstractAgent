# AbstractAgent

Agent patterns (ReAct / CodeAct / MemAct) built on **AbstractRuntime** (durable execution) and **AbstractCore** (tools + LLM integration).

Start here: [`docs/getting-started.md`](docs/getting-started.md) (then [`docs/README.md`](docs/README.md) for the full index)

## What you get

- **ReAct**: tool-first Reason → Act → Observe loop
- **CodeAct**: executes Python (tool call or fenced ` ```python``` ` blocks)
- **MemAct**: memory-enhanced agent using runtime-owned Active Memory
- **Durable runs**: pause/resume via `run_id` + runtime stores
- **Tool control**: explicit tool bundles + per-run allowlists
- **Observability**: durable ledger of LLM calls, tool calls, and waits

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
