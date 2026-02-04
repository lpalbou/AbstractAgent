# Contributing

Thanks for taking the time to contribute. This project aims to keep the agent layer small, reliable, and durable-runtime friendly.

## How to contribute

- **Bug reports**: include reproduction steps, expected vs actual behavior, and logs/tool outputs when relevant.
- **Feature requests**: describe the user problem first, then propose the API/UX.
- **Pull requests**: keep changes focused, add/adjust tests when behavior changes, and update docs when user-facing behavior changes.

Security issues: please follow [`SECURITY.md`](SECURITY.md) (responsible disclosure).

## Development setup

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Documentation updates

Entry points:
- [`README.md`](README.md)
- [`docs/getting-started.md`](docs/getting-started.md)
- [`docs/README.md`](docs/README.md) (doc index)

If you change behavior, update the relevant doc(s) and ensure they remain anchored in code evidence:
use file paths and symbol names (e.g., `src/abstractagent/adapters/react_runtime.py:create_react_workflow`).

## Design principles (project conventions)

- **Logic vs runtime separation**: keep prompting/parsing logic in `src/abstractagent/logic/*` free of runtime imports.
- **Durability**: do not store tool callables in `RunState.vars`. Prefer host-held tool executors (e.g. `MappingToolExecutor`).
- **Truthfulness**: do not claim actions without tool outputs; tests should enforce important contracts.

