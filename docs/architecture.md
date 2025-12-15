# Architecture Overview

This document explains the architecture of the AbstractCore → AbstractRuntime → AbstractAgent stack.

## Quick Answers

**Q: Are tool definitions properly injected into the LLM prompt?**
Yes. The flow is:
```
@tool decorator → ToolDefinition → to_dict() → LLM_CALL payload → UniversalToolHandler.format_tools_prompt() → LLM prompt
```
Tool execution is durable: tool *specs* are persisted; tool callables are executed via a host-configured `ToolExecutor`.

**Q: Can I have multiple agents?**
Yes. Three patterns:
1. **Same runtime, different agents** - Shared ledger/storage, independent runs
2. **Different runtimes** - Completely isolated
3. **AbstractFlow orchestration** - Coordinated multi-agent workflows

## Package Ownership

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AbstractAgent                                   │
│  Owns: Agent patterns, tool implementations, user interaction               │
│  - ReactAgent, create_react_agent()                                         │
│  - Tool implementations: list_files, read_file, search_files, etc.          │
│  - REPL and UI components                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ uses
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AbstractRuntime                                  │
│  Owns: Durable execution, effect system, storage                            │
│  - Runtime (tick/resume/cancel)                                             │
│  - Effect handlers (LLM_CALL, TOOL_CALLS, ASK_USER, etc.)                   │
│  - RunStore, LedgerStore                                                    │
│  - Workflow orchestration                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ uses
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AbstractCore                                    │
│  Owns: LLM abstraction, tool primitives, model architecture detection       │
│  - create_llm() - provider-agnostic LLM creation                            │
│  - @tool decorator, ToolDefinition, ToolCall, ToolResult                    │
│  - UniversalToolHandler - prompt formatting for all models                  │
│  - ToolRegistry (optional global registry)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tool Flow: From Agent to LLM Prompt

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. TOOL DEFINITION (AbstractCore)                                            │
│                                                                              │
│    @tool(name="list_files", description="List files in a directory")        │
│    def list_files(path: str = ".") -> str:                                   │
│        ...                                                                   │
│                                                                              │
│    Result: Function has ._tool_definition attribute (ToolDefinition)        │
│    NOTE: This does NOT register to global registry                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. AGENT CREATION (AbstractAgent)                                            │
│                                                                              │
│    agent = ReactAgent(runtime=runtime, tools=[list_files, read_file])       │
│                                                                              │
│    - Tools stored in agent.tools                                            │
│    - Workflow created with tool definitions for LLM prompt                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. WORKFLOW START (AbstractRuntime)                                          │
│                                                                              │
│    agent.start("List files in current directory")                           │
│                                                                              │
│    runtime.start(workflow, vars={"task": task, "_runtime": {"tool_specs": [...], "toolset_id": "ts_..."}}) │
│                                                                              │
│    - Only tool *specs* are persisted (JSON-safe)                            │
│    - Tool callables are held by the host ToolExecutor                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. LLM CALL (AbstractRuntime → AbstractCore)                                 │
│                                                                              │
│    reason_node() returns:                                                   │
│    StepPlan(effect=Effect(                                                  │
│        type=EffectType.LLM_CALL,                                            │
│        payload={"prompt": prompt, "tools": tool_dicts}                      │
│    ))                                                                        │
│                                                                              │
│    tool_dicts = [t.to_dict() for t in tool_definitions]                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. PROMPT FORMATTING (AbstractCore - UniversalToolHandler)                   │
│                                                                              │
│    LocalAbstractCoreLLMClient.generate():                                   │
│        tool_handler = UniversalToolHandler(model)                           │
│        tools_prompt = tool_handler.format_tools_prompt(tool_defs)           │
│        effective_prompt = f"{tools_prompt}\n\nUser request: {prompt}"       │
│                                                                              │
│    - Detects model architecture (qwen, llama, mistral, etc.)                │
│    - Formats tools in architecture-specific syntax                          │
│    - Handles both native API and prompted modes                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. RESPONSE PARSING (AbstractCore - UniversalToolHandler)                    │
│                                                                              │
│    parsed = tool_handler.parse_response(content, mode="prompted")           │
│                                                                              │
│    - Detects tool call syntax in response                                   │
│    - Extracts tool name and arguments                                       │
│    - Returns ToolCallResponse with tool_calls list                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. TOOL EXECUTION (AbstractRuntime effect handler)                           │
│                                                                              │
│    act_node() returns:                                                      │
│    StepPlan(effect=Effect(                                                  │
│        type=EffectType.TOOL_CALLS,                                          │
│        payload={"tool_calls": [...]}                                        │
│    ))                                                                        │
│                                                                              │
│    Effect handler: ToolExecutor only                                        │
│    - executed: executes tools locally                                       │
│    - passthrough/approval_required: returns tool_calls and runtime waits    │
│      until the host resumes with tool results                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Global Registry: Used or Bypassed?

**Current architecture does not require the global registry.**

- Tool execution is performed by a host-configured `ToolExecutor` (explicit, per-runtime/session).
- Run state persists only tool specs (`run.vars["_runtime"]`) so JSON-backed resume works.

The AbstractCore global registry remains available only as a legacy adapter path (via `AbstractCoreToolExecutor`) and is discouraged for durable agent runs.

**When is the global registry used?**

Only if you explicitly configure the runtime with an executor that relies on it (e.g. `AbstractCoreToolExecutor`) and register tools globally. The recommended durable path is `MappingToolExecutor`.

## Runtime Access

The `ReactAgent` exposes the runtime for advanced operations:

```python
agent = ReactAgent(runtime=runtime, tools=[...])

# Direct runtime access
agent.runtime.ledger_store.list(run_id)  # Get ledger entries
agent.runtime.run_store.load(run_id)     # Get run state
agent.runtime.cancel_run(run_id)         # Cancel a run

# Agent convenience methods
agent.get_state()           # Current RunState
agent.is_waiting()          # Check if waiting for input
agent.get_pending_question() # Get question details
agent.resume(response)      # Resume with user response
agent.attach(run_id)        # Attach to existing run
agent.save_state(filepath)  # Persist a run reference (run_id/workflow_id/actor_id/session_id)
agent.load_state(filepath)  # Attach to a persisted run reference
```

### Missing: Cancel Method

The agent exposes `cancel()` via `BaseAgent`:

```python
agent.cancel(reason="User cancelled")
```

## Async Message Passing

**Current state: Implemented (inbox pattern)**

`BaseAgent.inject_message()` appends guidance into `run.vars["_runtime"]["inbox"]`. The ReAct workflow reads and clears the inbox at the start of each reasoning step.

### What exists:
- `WAIT_EVENT` effect - pause until external signal
- `ASK_USER` effect - pause for user input
- `resume()` - inject payload to unblock

### What's missing:
- No way to inject messages while agent is actively running
- No interrupt mechanism (only cancel)
- No "suggestion" or "guidance" channel

### Potential solutions:

1. **Polling-based**: Agent checks `run.vars["_runtime"]["inbox"]` each iteration
   ```python
   def reason_node(run, ctx):
       inbox = (run.vars.get("_runtime") or {}).get("inbox", [])
       if inbox:
           # Incorporate messages into prompt
           ...
   ```

2. **Event-based**: Add `INJECT_MESSAGE` effect type
   ```python
   runtime.inject(run_id, message="Consider using grep instead")
   ```

3. **Callback-based**: Agent calls hook before each LLM call
   ```python
   agent = ReactAgent(
       runtime=runtime,
       tools=[...],
       before_llm_call=lambda state: get_pending_guidance()
   )
   ```

## Ledger Recording

All effects are recorded in the ledger:

```python
# Get ledger for a run
entries = agent.runtime.ledger_store.list(run_id)

# Each entry contains:
{
    "run_id": "...",
    "node_id": "reason",
    "effect_type": "llm_call",
    "status": "completed",
    "started_at": "2024-...",
    "ended_at": "2024-...",
    "result": {...},
    "idempotency_key": "...",
    "attempt": 1
}
```

## Summary: What's Working, What's Missing

### ✅ Working

1. **Tool definition** - `@tool` decorator creates `ToolDefinition`
2. **Tool prompt formatting** - `UniversalToolHandler` formats for any model
3. **Tool execution** - Via host-configured `ToolExecutor` (durable)
4. **Ledger recording** - All effects recorded
5. **Pause/resume** - `ASK_USER`, `WAIT_EVENT`, `WAIT_UNTIL`
6. **State persistence** - `save_state()` / `load_state()`
7. **Cancel** - `agent.cancel(reason)` 
8. **Ledger access** - `agent.get_ledger()`
9. **Async message injection** - `agent.inject_message("guidance")`

### ⚠️ Limitations

1. **Interrupt** - Can cancel but not pause mid-execution
2. **Message injection timing** - Messages are read at start of each reasoning step, not mid-LLM-call

### 🔍 Verified: Global Registry Bypassed

The current architecture does not require the global `ToolRegistry`. Tools flow:
1. Agent constructor → `agent.tools`
2. Workflow start → `run.vars["_runtime"]` stores tool specs/toolset_id (JSON-safe)
3. TOOL_CALLS handler → `ToolExecutor.execute(...)` (explicit execution boundary)

This is intentional - it avoids hidden global state and makes pause/persist/resume reliable.

## New Agent Methods

```python
agent = ReactAgent(runtime=runtime, tools=[...])

# Cancel a running agent
agent.cancel(reason="User requested stop")

# Get ledger entries
ledger = agent.get_ledger()
for entry in ledger:
    print(f"{entry['effect_type']}: {entry['status']}")

# Inject guidance while agent is running (read on next iteration)
agent.inject_message("Consider using grep instead of reading the whole file")
```

The `inject_message()` method stores messages in `run.vars["_runtime"]["inbox"]`. The agent reads and clears the inbox at the start of each reasoning step, incorporating the guidance into the LLM prompt.

## Multi-Agent Patterns

### Pattern 1: Shared Runtime, Different Agents

Multiple agents can share the same runtime. Each agent has its own run_id but shares storage.

```python
from abstractruntime.integrations.abstractcore import create_local_runtime
from abstractagent import ReactAgent

# One runtime, shared storage
runtime = create_local_runtime(provider="ollama", model="qwen3:4b")

# Multiple agents with different tools
researcher = ReactAgent(runtime=runtime, tools=[search_web, read_url])
coder = ReactAgent(runtime=runtime, tools=[read_file, write_file, execute_command])
reviewer = ReactAgent(runtime=runtime, tools=[read_file, run_tests])

# Each agent has independent runs
researcher.start("Research best practices for Python async")
coder.start("Implement the async handler")

# But they share the same ledger store
all_entries = runtime.ledger_store.list_all()  # See all agent activity
```

**Benefits:**
- Unified audit trail in ledger
- Shared storage backend
- Lower resource usage

**Considerations:**
- Agents don't automatically communicate
- Need external coordination for handoffs

### Pattern 2: Isolated Runtimes

Each agent gets its own runtime for complete isolation.

```python
runtime_a = create_local_runtime(provider="ollama", model="qwen3:4b")
runtime_b = create_local_runtime(provider="openai", model="gpt-4")

agent_a = ReactAgent(runtime=runtime_a, tools=[...])
agent_b = ReactAgent(runtime=runtime_b, tools=[...])
```

**Benefits:**
- Complete isolation
- Different LLM providers per agent
- Independent failure domains

### Pattern 3: AbstractFlow Orchestration (Future)

AbstractFlow will provide higher-level orchestration for multi-agent workflows:

```python
# Conceptual - AbstractFlow design
from abstractflow import Flow, parallel, sequence

flow = Flow(
    name="code_review_pipeline",
    steps=[
        # Run in parallel
        parallel(
            agent=researcher, task="Research the codebase structure",
            agent=security_scanner, task="Scan for vulnerabilities",
        ),
        # Then sequence
        sequence(
            agent=coder, task="Implement changes based on research",
            agent=reviewer, task="Review the implementation",
        ),
    ],
    # Coordination
    on_agent_complete=lambda agent, result: notify_next(result),
    shared_context=True,  # Agents can see each other's outputs
)

flow.run()
```

**AbstractFlow responsibilities:**
- Agent lifecycle management
- Inter-agent communication
- Shared context/memory
- Failure handling and retries
- Progress tracking

**AbstractRuntime responsibilities:**
- Individual agent execution
- Effect handling (LLM, tools)
- Durability and persistence
- Ledger recording

### Pattern 4: Manual Coordination

Without AbstractFlow, coordinate agents manually:

```python
# Sequential handoff
researcher.start("Research the problem")
research_result = researcher.run_to_completion()

coder.start(f"Implement solution based on: {research_result.output['answer']}")
code_result = coder.run_to_completion()

reviewer.start(f"Review this code: {code_result.output['answer']}")
review_result = reviewer.run_to_completion()
```

```python
# Parallel with threading
import threading

def run_agent(agent, task, results, key):
    agent.start(task)
    results[key] = agent.run_to_completion()

results = {}
threads = [
    threading.Thread(target=run_agent, args=(researcher, "Research X", results, "research")),
    threading.Thread(target=run_agent, args=(scanner, "Scan Y", results, "scan")),
]

for t in threads:
    t.start()
for t in threads:
    t.join()

# Combine results
combined = f"Research: {results['research'].output}\nScan: {results['scan'].output}"
```

## BaseAgent: Creating Custom Agents

All agents inherit from `BaseAgent` to get common functionality:

```python
from abstractagent import BaseAgent
from abstractruntime import WorkflowSpec, StepPlan, Effect, EffectType

class CodeActAgent(BaseAgent):
    """CodeAct agent that executes code directly."""
    
    def _create_workflow(self) -> WorkflowSpec:
        # Define your workflow nodes
        def plan_node(run, ctx):
            return StepPlan(
                node_id="plan",
                effect=Effect(type=EffectType.LLM_CALL, payload={...}),
                next_node="execute",
            )
        
        def execute_node(run, ctx):
            # Execute code
            ...
        
        return WorkflowSpec(
            workflow_id="codeact_agent",
            entry_node="plan",
            nodes={"plan": plan_node, "execute": execute_node, ...},
        )
    
    def start(self, task: str) -> str:
        self._current_run_id = self.runtime.start(
            workflow=self.workflow,
            vars={"task": task},
        )
        return self._current_run_id
    
    def step(self) -> RunState:
        return self.runtime.tick(workflow=self.workflow, run_id=self._current_run_id)

# CodeActAgent automatically gets:
# - run_to_completion(), get_state(), is_waiting(), is_running(), is_complete()
# - cancel(), get_ledger(), inject_message()
# - save_state(), load_state(), attach()
# - get_output(), get_error()
```

## Agent Type Comparison

| Feature | ReAct | CodeAct (planned) |
|---------|-------|-------------------|
| Reasoning | Explicit reason step | Implicit in code |
| Actions | Tool calls | Code execution |
| Observation | Tool results | Execution output |
| Iteration | Reason→Act→Observe loop | Plan→Execute→Verify loop |
| Best for | General tasks | Code-heavy tasks |

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER CODE                                          │
│                                                                              │
│  agent = ReactAgent(runtime, tools=[list_files, read_file])                 │
│  agent.start("List files")                                                  │
│  agent.inject_message("Focus on .py files")  ←── async guidance             │
│  state = agent.run_to_completion()                                          │
│  ledger = agent.get_ledger()                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ReactAgent (BaseAgent)                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Workflow: init → reason → parse → act → observe → reason → ... → done│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  run.vars["_runtime"] = {"tool_specs": [...], "toolset_id": "ts_..."}       │
│  run.vars["_runtime"]["inbox"] = [{"content": "Focus on .py files"}]  ←── messages │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AbstractRuntime                                    │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   RunStore   │    │ LedgerStore  │    │   Runtime    │                   │
│  │  (run state) │    │ (audit log)  │    │  (tick/resume)│                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
│  Effect Handlers:                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  LLM_CALL    │    │ TOOL_CALLS   │    │  ASK_USER    │                   │
│  │  handler     │    │  handler     │    │  handler     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                                               │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ TOOL_CALLS handler → ToolExecutor.execute(tool_calls)                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AbstractCore                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ UniversalToolHandler.format_tools_prompt(tool_defs)                  │   │
│  │                                                                      │   │
│  │ Output:                                                              │   │
│  │ "You have access to the following tools:                             │   │
│  │  **list_files**: List files in a directory                           │   │
│  │  **read_file**: Read file contents                                   │   │
│  │  To use a tool: <|tool_call|>{"name": "...", "arguments": {...}}     │   │
│  │  </|tool_call|>"                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ create_llm(provider, model).generate(prompt_with_tools)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ UniversalToolHandler.parse_response(content) → ToolCallResponse      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM Provider (Ollama/OpenAI/etc)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

1. **No callables in RunState** - Tools are executed via `ToolExecutor`; only specs are persisted

2. **BaseAgent provides common functionality** - All agent types inherit cancel, ledger, inject_message, etc.

3. **Runtime is reusable** - Multiple agents can share one runtime for unified storage

4. **AbstractFlow (future) handles orchestration** - Runtime handles execution, Flow handles coordination

5. **Async messages via inbox** - `inject_message()` stores in `_runtime.inbox`, read at next reasoning step
