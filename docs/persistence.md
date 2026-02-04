# State & Persistence (Pause / Resume)

Related:
- [`docs/getting-started.md`](getting-started.md)
- [`docs/agents.md`](agents.md)

AbstractAgent agents run on **AbstractRuntime**, which owns durable run state (`RunStore`) and the execution ledger (`LedgerStore`).

## What `save_state()` actually saves

`BaseAgent.save_state(path)` (see `src/abstractagent/agents/base.py`) stores only:
- `run_id`
- `workflow_id`
- `actor_id`
- `session_id`

The full durable state lives in the runtime’s `RunStore`. This is why a persistent `RunStore` is required
to resume across process restarts.

## Using a persistent store

Example using JSON files:

```python
from abstractagent import create_react_agent
from abstractruntime.storage.json_files import JsonFileRunStore, JsonlLedgerStore

run_store = JsonFileRunStore(".runs")
ledger_store = JsonlLedgerStore(".runs")

agent = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent.start("Do something that takes time")
agent.save_state("agent_state.json")
```

Later:

```python
agent2 = create_react_agent(run_store=run_store, ledger_store=ledger_store)
agent2.load_state("agent_state.json")
state = agent2.run_to_completion()
```

## Attaching without a state file

If you already have a `run_id`:

```python
agent.attach(run_id)
```

This validates the `workflow_id` (and actor/session when available) to prevent accidental cross-agent attachment.

## Ledger access

All agents expose:

```python
entries = agent.get_ledger()
```

This delegates to `Runtime.get_ledger(...)` and returns structured effect records (LLM calls, tool calls, waits, etc).
