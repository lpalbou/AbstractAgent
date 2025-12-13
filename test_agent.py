"""Test script for the ReAct agent."""

import sys
sys.path.insert(0, "src")

from abstractcore.tools import ToolRegistry
from abstractruntime import (
    RunStatus,
    InMemoryRunStore,
    InMemoryLedgerStore,
)
from abstractruntime.integrations.abstractcore import create_local_runtime

from abstractagent.agents.react import ReactAgent
from abstractagent.tools import ALL_TOOLS


def on_step(step: str, data: dict) -> None:
    """Print step information."""
    if step == "init":
        print(f"🚀 Starting: {data.get('task', '')[:50]}")
    elif step == "reason":
        print(f"🤔 Reasoning (iteration {data.get('iteration', '?')})...")
    elif step == "parse":
        print(f"📋 Action: {data.get('action_type', 'unknown')}")
    elif step == "act":
        print(f"🔧 Tool: {data.get('tool', '')}({data.get('args', {})})")
    elif step == "observe":
        print(f"👁️ Result: {data.get('result', '')[:80]}...")
    elif step == "done":
        print(f"✅ ANSWER: {data.get('answer', '')}")


def main():
    # Setup
    tool_registry = ToolRegistry()
    for tool in ALL_TOOLS:
        tool_registry.register(tool)
    
    runtime = create_local_runtime(
        provider="ollama",
        model="qwen3:1.7b-q4_K_M",
    )
    
    agent = ReactAgent(
        runtime=runtime,
        tool_registry=tool_registry,
        on_step=on_step,
    )
    
    # Test task
    task = "List the files in the current directory"
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")
    
    agent.start(task)
    state = agent.run_to_completion()
    
    print(f"\n{'='*60}")
    print(f"Final Status: {state.status.value}")
    print(f"Iterations: {state.output.get('iterations', '?')}")
    print(f"Answer: {state.output.get('answer', 'N/A')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
