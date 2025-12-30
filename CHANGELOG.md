# Changelog

All notable changes to AbstractAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Documentation: clarify that the interactive REPL moved to **AbstractCode**; `react-agent`/`python -m abstractagent.repl` are deprecated shims.
- Observability: `on_step` tool-observation previews now include up to **1000 characters** (was 150), with an explicit truncation marker for larger outputs. Full tool outputs are still preserved in the agent message history.
- ReAct prompt construction now **truncates large tool-call arguments** (e.g. `write_file` content) when building the runtime scratchpad, preventing prompt bloat and reducing stalls on local providers.

### Fixed
- Limits: ReAct/CodeAct now treat `_limits.max_tokens` as a **context/budget** limit and use `_limits.max_output_tokens` (if set) to cap OpenAI-style `max_tokens` (output). This prevents invalid LMStudio requests like `max_tokens=262144` which can return HTTP 400.

## [0.2.0] - 2025-12-17

### Features

#### Agent Types

- **ReactAgent**: Reason-Act-Observe agent implementing the ReAct pattern with iterative tool calling
  - Configurable max iterations and token limits
  - Real-time step visibility via `on_step` callback
  - Built-in `ask_user` tool for human-in-the-loop interactions

- **CodeActAgent**: Code-first agent that executes Python code snippets as its primary action
  - Parses Python code from fenced markdown blocks (`\`\`\`python`)
  - Native `execute_python` tool integration
  - Sandbox interface for safe code execution

- **BaseAgent**: Abstract base class providing common functionality for all agent types
  - Runtime integration with AbstractRuntime
  - State management (save/load/attach)
  - Cancellation support
  - Ledger access for auditability
  - Async message injection mid-run

#### Factory Functions

- `create_react_agent()`: One-liner factory to create a ReactAgent with sensible defaults
- `create_codeact_agent()`: One-liner factory to create a CodeActAgent
- Support for custom LLM providers (ollama, anthropic, openai, etc.) via AbstractCore

#### State Persistence & Durability

- Save/load agent state to JSON files for resume across process restarts
- File-backed stores (`JsonFileRunStore`, `JsonlLedgerStore`) for durable execution
- Attach to existing runs by run_id
- Clear state after completion

#### Interactive REPL

- Full-featured async REPL (`python -m abstractagent.repl`)
- Real-time step visibility with timestamps and color coding
- Pause/resume capability with Ctrl+C
- Commands: `status`, `history`, `tools`, `resume`, `help`, `quit`
- State file persistence for resuming across sessions

#### Human-in-the-Loop

- Built-in `ask_user` tool for agents to ask questions
- Interactive question UI with:
  - Multiple choice options
  - Free text input
  - Combined choice + free text
- Async and sync versions for flexible integration

#### Tools

File operations (from AbstractCore):
- `list_files(path)`: List files and directories
- `read_file(path)`: Read file contents
- `search_files(pattern, path)`: Search for files matching glob patterns
- `write_file(path, content)`: Write content to files
- `edit_file(path, old_text, new_text)`: Edit files with search/replace

Web tools (from AbstractCore):
- `web_search(query)`: Search the web
- `fetch_url(url)`: Fetch URL contents

System tools:
- `execute_command(command)`: Execute shell commands with safety restrictions
- `execute_python(code)`: Execute Python code in a sandbox

Agent-specific tools:
- `self_improve(suggestion, target, category, tags)`: Log improvement suggestions to JSONL for later review

#### Limit Management

- Configurable limits: `max_iterations`, `max_tokens`, `max_history_messages`
- Runtime limit updates via `update_limits()`
- Limit status queries via `get_limit_status()`
- Warning thresholds (80% by default) for proactive notification

#### Actor & Session Identity

- ActorFingerprint integration for identifying agent runs
- Session-based message continuity
- Actor/session ID validation on attach/load

#### Ledger & Auditability

- All tool calls recorded via AbstractRuntime's ledger
- Ledger access via `get_ledger()` method
- JSONL ledger store for persistent audit trails

#### Sandbox Interface

- `Sandbox` protocol for code execution environments
- `ExecutionResult` dataclass with stdout, stderr, exit_code, duration
- `LocalSandbox` implementation for subprocess-based execution

### Architecture

```
AbstractAgent
     |
     +-- Uses AbstractRuntime for durable execution
     |   - Workflows survive crashes
     |   - Pause/resume capability
     |   - Ledger tracks all actions
     |
     +-- Uses AbstractCore for LLM/tools
         - Provider-agnostic LLM calls
         - Tool registration and execution
         - Tool call parsing for all model architectures
```

### Dependencies

- abstractcore
- abstractruntime
- Python >= 3.10

[0.2.0]: https://github.com/abstractframework/abstractagent/releases/tag/v0.2.0
