# AbstractAgent

Agent implementations using AbstractRuntime and AbstractCore.

## Features

- **ReAct Agent**: Reason-Act-Observe loop with tool calling
- **Host UX in AbstractCode**: the interactive terminal shell lives in **AbstractCode**; AbstractAgent stays focused on agent patterns/workflows
- **Pause/Resume**: Durable agent state with interrupt/resume capability
- **Ask User**: Agent can ask questions with multiple choice + free text
- **Ledger Recording**: All tool calls recorded for auditability

## Installation

```bash
pip install -e .
```

## Quick Start

### Simple (Factory)

```python
from abstractagent import create_react_agent

# One-liner agent creation
agent = create_react_agent()
agent.start("List the files in the current directory")
state = agent.run_to_completion()
print(state.output["answer"])
```

### With Custom Tools

```python
from abstractagent import create_react_agent
from abstractcore.tools import tool

@tool(name="my_tool", description="My custom tool")
def my_tool(query: str) -> str:
    """My custom tool."""
    return f"Result for {query}"

agent = create_react_agent(tools=[my_tool])
```

### Full Control

```python
from abstractruntime.integrations.abstractcore import create_local_runtime
from abstractagent import ReactAgent, list_files, read_file

# Create runtime
runtime = create_local_runtime(
    provider="ollama",
    model="qwen3:4b-instruct-2507-q4_K_M",
)

# Create agent with specific tools
agent = ReactAgent(
    runtime=runtime,
    tools=[list_files, read_file],
)

agent.start("List the files in the current directory")
state = agent.run_to_completion()
print(state.output["answer"])
```

## State Persistence

Resume agents across process restarts:

```python
agent = create_react_agent()
agent.start("Long running task")

# Save state before exit
agent.save_state("agent_state.json")

# ... process restarts ...

# Load and resume
agent = create_react_agent()
agent.load_state("agent_state.json")
state = agent.run_to_completion()

# Cleanup
agent.clear_state("agent_state.json")
```

## Interactive Shell (AbstractCode)

```bash
# The interactive REPL moved to AbstractCode (host UX).
abstractcode --agent react --provider ollama --model qwen3:4b-instruct-2507-q4_K_M
```

## Architecture

```
AbstractAgent
     │
     ├── Uses AbstractRuntime for durable execution
     │   - Workflows survive crashes
     │   - Pause/resume capability
     │   - Ledger tracks all actions (LLM calls, tool calls)
     │
     └── Uses AbstractCore for LLM/tools
         - Provider-agnostic LLM calls
         - Tool registration and execution
         - Tool call parsing for all model architectures
```

## Available Tools

- Default tool callables are re-exported from AbstractCore in `abstractagent.tools` (file ops, web tools, `execute_command`), plus:
  - `execute_python(code, timeout_s=...)`
  - `self_improve(suggestion, ...)`
- The agent also exposes schema-only built-ins (`ask_user`, `recall_memory`, `inspect_vars`, `remember`, `compact_memory`) which are translated into **Runtime effects** by the workflow adapters (durable; no callable persistence).
