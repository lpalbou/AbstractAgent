# AbstractAgent Documentation

This folder documents the `abstractagent` Python package in this repository.

AbstractAgent is part of the **AbstractFramework** ecosystem:
- AbstractFramework: https://github.com/lpalbou/AbstractFramework
- AbstractCore (providers + tool schemas): https://github.com/lpalbou/abstractcore
- AbstractRuntime (durable workflows + storage/ledger): https://github.com/lpalbou/abstractruntime

If you are new to the stack, start with [`docs/getting-started.md`](getting-started.md) and the overview diagram in
[`docs/architecture.md`](architecture.md).

## Start here

- **Getting started**: [`docs/getting-started.md`](getting-started.md)
- **FAQ**: [`docs/faq.md`](faq.md)
- **Troubleshooting**: [`docs/troubleshooting.md`](troubleshooting.md)
- **API reference**: [`docs/api.md`](api.md)
- **Agents (ReAct / CodeAct / MemAct)**: [`docs/agents.md`](agents.md)
- **Tools and allowlists**: [`docs/tools.md`](tools.md)
- **Loop hooks (listen / steer / capture)**: [`docs/hooks.md`](hooks.md)
- **Skills attachment + system-prompt slots**: [`docs/skills-attachment.md`](skills-attachment.md)
- **State + persistence (pause/resume)**: [`docs/persistence.md`](persistence.md)
- **Queued work (task after task)**: `reset_react_task` — see the "Task reset" section in [`docs/api.md`](api.md)

## Deep dives

- **Architecture (with diagrams)**: [`docs/architecture.md`](architecture.md) — layering, the ReAct graph, delegated sub-agents and the `_runtime` controls they inherit
- **ReAct pipeline (implemented)**: [`docs/react-pipeline.md`](react-pipeline.md)

## Development

- **Running tests / local dev**: [`docs/development.md`](development.md)

`docs/quickstart.md` is a redirect stub to the getting-started guide, kept for older links.

## Project docs

- Changelog: [`CHANGELOG.md`](../CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- Security: [`SECURITY.md`](../SECURITY.md)
- Acknowledgements: [`ACKNOWLEDMENTS.md`](../ACKNOWLEDMENTS.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md)
- License: [`LICENSE`](../LICENSE)
