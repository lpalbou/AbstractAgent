# AbstractAgent

Agent implementations using AbstractRuntime and AbstractCore.

## Features

- **ReAct Agent**: Reason-Act-Observe loop with tool calling
- **Async REPL**: Interactive agent with real-time step visibility
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

## REPL Usage

```bash
# Start the ReAct agent REPL
python -m abstractagent.repl --provider ollama --model qwen3:4b-instruct-2507-q4_K_M
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

- `list_files(path)` - List files and directories
- `read_file(path)` - Read file contents
- `search_files(pattern, path)` - Search for files matching a glob pattern
- `execute_command(command)` - Execute a shell command (with safety restrictions)
- `ask_user(question, choices)` - Ask the user a question (built-in)
