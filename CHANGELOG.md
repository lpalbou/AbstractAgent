# Changelog

All notable changes to AbstractAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-01-06

### Added

- **MemAct Agent**: New agent type implementing Memory-augmented Actor pattern
  - Complete implementation: `agents/memact.py`, `logic/memact.py`, `adapters/memact_runtime.py`
  - Uses Active Memory system (MY PERSONA, RELATIONSHIPS, CURRENT TASKS, etc.)
  - Schema-only `active_memory_delta` tool executed by AbstractRuntime
  - Dedicated tests: `test_memact_workflow.py`, `test_memact_session_memory.py`

- **Verifier Policy** (ReAct/CodeAct): Evidence-based structured output for loop reliability
  - Can return `next_tool_calls` when response is incomplete (avoids brittle "continuation text" heuristics)
  - Omits full "current answer" when tool outputs exist (prevents feeding huge code dumps back to model)
  - Includes recent `ask_user` prompts and user answers in verifier context
  - Tests: `test_react_runtime_verifier_policy.py`, `test_codeact_runtime_verifier_policy.py`

- **Tool Call Queue Management**: `_temp.pending_tool_calls` treated as queue
  - Interleaving schema-only tools with normal tools never drops calls
  - Prevents re-asking already-answered questions
  - Test: `test_toolcall_queue_and_review_prompt.py`

- **Character-Level Truncation** (opt-in): Context budgeting for LLM-visible message payloads
  - Keeps durable run history intact while truncating LLM-facing messages
  - Default: **off** (`_limits.max_message_chars=-1`, `_limits.max_tool_message_chars=-1`)
  - Configurable per-run limits

- **Tool Loop Guard**: Evidence-based deduplication and retry logic
  - Prevents infinite loops from repeated identical tool calls
  - Configurable retry limits and backoff
  - Test: `test_react_runtime_tool_loop_guard.py`

- **LMStudio Tool Eval Script** (`scripts/lmstudio_tool_eval.py`): Tool calling evaluation harness
  - 426 lines of evaluation logic for testing tool calling capabilities
  - Useful for benchmarking LMStudio models

- **React Pipeline Documentation** (`docs/react-pipeline.md`): Complete 190-line implementation guide
  - Explains Reason → Parse → Act → Observe cycle
  - Clarifies responsibilities: AbstractCore (LLM), AbstractRuntime (execution), AbstractAgent (semantics)
  - Documents data model (`RunState.vars` namespaces)
  - Cross-references ReAct scratchpad best practices

### Changed

- **Prompt Role Separation**: User-role content now contains **only the user's request**
  - Internal state (Active Memory, iteration info, guidance/plan) rendered into **system prompt**
  - Reduces instruction confusion and tool-call hallucinations
  - Prevents user-role pollution on native-tool providers

- **Tool Observation Formatting**: Multiple improvements
  - Previews now include up to **1000 characters** (was 150) with explicit truncation marker
  - Prefers tool-supplied `rendered` string for structured outputs (preserves provenance without polluting LLM history)
  - History formatting uses natural-language lines like `Tool <name> succeeded/failed: ...` instead of bracketed `observation[...]` markers

- **System Prompts Enhanced**:
  - Explicit **output-token budget hint** in ReAct system prompt
  - Guidance to **chunk large tool arguments** (e.g., file contents) across multiple tool calls
  - Prevents `max_tokens` truncation loops

- **Memory Tools Enhanced**: Richer built-in memory tool schemas
  - `recall_memory`: Supports `tags_mode`, `usernames`, `locations` (metadata-first filtering)
  - `remember_note`: Supports optional `location` field
  - Aligns with AbstractRuntime 0.4.0 memory query enhancements

- **Active Memory Integration**: `active_memory_delta` now schema-only tool
  - Executed by AbstractRuntime (not parsed by LLM provider)
  - Prevents native-tool providers (LMStudio) from mis-parsing inline memory JSON as tool calls
  - Module alias tools also schema-only

- **Message Sanitization**: Comprehensive content sanitization for LLM interactions
  - Cleans tool observation transcripts
  - Handles character limits and truncation markers
  - Preserves full outputs in durable history
  - Test: `test_runtime_parse_sanitizes_observation_transcripts.py`

- **Final Answer Extraction**: Enhanced verification and extraction logic
  - Better handling of incomplete responses
  - Improved parsing robustness
  - Recovery mechanisms for empty responses

- **Documentation Updates**:
  - Clarified interactive REPL moved to **AbstractCode** (deprecated shims: `react-agent`, `python -m abstractagent.repl`)
  - Enhanced architecture documentation with memory management details

### Fixed

- **Token Limits Handling**: Proper distinction between context and output limits
  - `_limits.max_tokens` treated as **context/budget** limit
  - `_limits.max_output_tokens` (if set) caps OpenAI-style `max_tokens` (output)
  - Prevents invalid LMStudio requests like `max_tokens=262144` (HTTP 400)
  - Test: `test_runtime_config_max_output_tokens_fallback.py`

- **Native Tool Handling**: Omit visible `Tools (session)` Active Memory block for native-tool models
  - Works even when workflow doesn't know provider name
  - Prevents conflicts with hidden tool grammars on OpenAI-compatible servers (LMStudio)
  - Test: `test_native_tool_handling_policy.py`

- **Tool Call Deduplication**: Enhanced retry logic prevents infinite loops
  - Evidence-based detection of repeated identical calls
  - Configurable retry limits
  - Test: `test_react_runtime_tool_loop_guard.py`

- **Context Preservation**: Improved message handling across tool cycles
  - Queue management preserves call order
  - No dropped tool calls during interleaving
  - Test: `test_toolcall_queue_and_review_prompt.py`

- **Empty Response Recovery**: Robust handling of empty LLM responses
  - Recovery mechanisms in CodeAct and ReAct workflows
  - Prevents workflow failures from transient LLM issues

### Testing

- **17 new/modified test files** covering:
  - MemAct agent functionality (2 tests)
  - Verifier policy (2 tests)
  - Tool loop guard and deduplication
  - Tool call queue management
  - Message sanitization and truncation
  - Plan review modes
  - Native tool handling
  - Context policy adoption
  - Runtime provider/model seeding
  - Prompt deobservation
  - Builtin tool specs

### Statistics

- **33 commits** improving agent reliability, memory systems, and prompt engineering
- **38 files changed**: 5,091 insertions, 487 deletions
- **5,578 total lines changed** across the codebase
- **3 new agent modules**: MemAct adapter, agent, and logic
- **1 new script**: LMStudio tool evaluation harness (426 lines)
- **1 new document**: React pipeline architecture (190 lines)

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
