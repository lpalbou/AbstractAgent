# AbstractAgent

Agent implementations using AbstractRuntime and AbstractCore.

## Features

- **ReAct Agent**: Reason-Act-Observe loop with tool calling
- **Async REPL**: Interactive agent with real-time step visibility
- **Pause/Resume**: Test runtime durability by pausing and resuming agents

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Start the ReAct agent REPL
react-agent --provider ollama --model qwen3:1.7b-q4_K_M
```

## Architecture

```
AbstractAgent
     │
     ├── Uses AbstractRuntime for durable execution
     │   - Workflows survive crashes
     │   - Pause/resume capability
     │   - Ledger tracks all actions
     │
     └── Uses AbstractCore for LLM/tools
         - Provider-agnostic LLM calls
         - Tool registration and execution
```
