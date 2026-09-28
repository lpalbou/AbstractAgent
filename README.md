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
- Loop hooks (listen/steer/capture): [`docs/hooks.md`](docs/hooks.md)
- FAQ: [`docs/faq.md`](docs/faq.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Architecture (diagrams): [`docs/architecture.md`](docs/architecture.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)
- Acknowledgements: [`ACKNOWLEDMENTS.md`](ACKNOWLEDMENTS.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## What you get

- **ReAct**: tool-first Reason → Act → Observe loop
- **CodeAct**: executes Python (tool calls; on prompted-tools models also fenced ` ```python``` ` blocks)
- **MemAct** (experimental): memory-enhanced agent using runtime-owned Active Memory — distinct architecture, less production mileage; import from `abstractagent.agents.memact` (deliberately not top-level)
- **Durable runs**: pause/resume via `run_id` + runtime stores
- **Tool control**: explicit tool bundles + per-run allowlists (unresolvable
  grant names are recorded loudly, never dropped silently)
- **Loop hooks**: listen/steer/capture on the running loop via `LoopHooks`
  (see [`docs/hooks.md`](docs/hooks.md)); the flat `on_step` callback remains
- **Generation controls**: temperature, seed, media policy, prompt-cache
  identity, streaming, Core `thinking` and `speculation` (MTP) are normalized
  before LLM calls and inherited by delegated children
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
`abstractagent[apple]` and `abstractagent[gpu]`. These delegate to the matching
AbstractCore and AbstractRuntime profiles; AbstractAgent itself remains
provider/runtime agnostic.

AbstractAgent requires `abstractcore[tools]>=2.13.41` and
`AbstractRuntime>=0.7.0` (the runtime release with the 50k-token history
window, which ReAct applies to its requests when a host sets
`_runtime.history_window_tokens`, as the entity visit does; 0.5.0 brought live
token streaming, on top of turn grounding, in-flight cancellation and
speculation inheritance). AbstractRuntime 0.7.0 itself requires AbstractCore
2.18.0 or newer.

Note: the repository may be ahead of the latest published PyPI release. To verify what you installed:

```bash
python -c "import importlib.metadata as md; print(md.version('abstractagent'))"
```

## Quick start (ReAct)

```python
from abstractagent import create_react_agent

# provider/model resolve from your AbstractCore config defaults
# (`abstractcore --config`) when omitted; pass them explicitly to pin.
agent = create_react_agent(provider="ollama", model="qwen3:4b")
agent.start("List the files in the current directory")
state = agent.run_to_completion()
print(state.output["answer"])
```

Tip: these loops send the full transcript plus ~19 tool schemas every cycle —
prefer a tool-capable model. On Ollama, raise the context window to the model's
maximum available context or the server silently truncates from the oldest
content first: per-call `llm_kwargs={"num_ctx": <model max>}` or server-side `OLLAMA_CONTEXT_LENGTH=<model max> ollama serve`.
House rule: maximum available context unless you explicitly choose otherwise —
a fixed lower number is a hidden ceiling (see [`docs/faq.md`](docs/faq.md)).

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
