# Changelog

All notable changes to `abstractagent` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-02-04

### Added

- New user-facing docs entrypoints and references:
  - `docs/getting-started.md`, `docs/api.md`, `docs/faq.md`, `docs/README.md`
  - `CONTRIBUTING.md`, `SECURITY.md`, `ACKNOWLEDMENTS.md`
- LLM repo maps: `llms.txt`, `llms-full.txt`

### Changed

- Documentation refresh to match current code behavior (including `open_attachment` runtime-owned attachment reading).
- ReAct “plan-only followthrough” retry heuristic is enabled by default (disable with `_runtime.check_plan=false`).

### Fixed

- `create_memact_agent(...)` correctly wires `tool_executor=` into `create_local_runtime(...)`.
- CodeAct fenced-code execution now includes `allowed_tools` in the `TOOL_CALLS` effect payload (consistent allowlist enforcement).
- `manual_agent_demo.py` is self-contained (no missing `abstractagent.ui` dependency).

## [0.3.0] - 2026-01-06

### Added

- MemAct agent pattern (agent + logic + runtime adapter):
  - `src/abstractagent/agents/memact.py`
  - `src/abstractagent/logic/memact.py`
  - `src/abstractagent/adapters/memact_runtime.py`
- Manual LMStudio evaluation harness: `src/abstractagent/scripts/lmstudio_tool_eval.py`

## [0.2.0] - 2025-12-17

### Added

- Initial agent patterns and durable workflow adapters:
  - ReAct: `src/abstractagent/agents/react.py`, `src/abstractagent/adapters/react_runtime.py`
  - CodeAct: `src/abstractagent/agents/codeact.py`, `src/abstractagent/adapters/codeact_runtime.py`
- Common agent API: `src/abstractagent/agents/base.py`
