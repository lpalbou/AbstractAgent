# Acknowledgements

Thanks to the maintainers and contributors of the libraries this package depends on.

AbstractAgent is part of the **AbstractFramework** ecosystem:
- AbstractFramework: https://github.com/lpalbou/AbstractFramework
- AbstractCore: https://github.com/lpalbou/abstractcore
- AbstractRuntime: https://github.com/lpalbou/abstractruntime

## Runtime dependencies

Declared in `pyproject.toml` (`[project].dependencies`) and required to use `abstractagent`:

- **AbstractCore** (`abstractcore[tools]`): provider/model integration + canonical tool definitions
  - used via `abstractcore.tools.common_tools` in `src/abstractagent/tools/__init__.py`
- **AbstractRuntime** (`abstractruntime`): durable execution engine (effects, waits, storage, ledger)
  - used throughout agent wrappers and adapters (e.g. `src/abstractagent/agents/base.py`, `src/abstractagent/adapters/react_runtime.py`)
  - the runtime’s AbstractCore integration implements runtime-owned tooling like `open_attachment`
    (`abstractruntime.integrations.abstractcore.session_attachments.execute_open_attachment`)

## Development / testing

Declared in `pyproject.toml` (`[project.optional-dependencies].dev`):

- **pytest**: test runner used by the suite in `tests/`

## Packaging

Declared in `pyproject.toml` (`[build-system].requires`):

- **setuptools** and **wheel**: build backend and wheel support

Conceptually, the ReAct agent pattern is inspired by:

- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models” (2022)
