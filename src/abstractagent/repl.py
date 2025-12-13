"""Async REPL for the ReAct agent.

Provides an interactive interface with:
- Real-time step visibility
- Pause/resume capability
- Command history
"""

import asyncio
import sys
import argparse
from typing import Dict, Any, Optional
from datetime import datetime

from abstractcore.tools import ToolRegistry
from abstractruntime import (
    RunStatus,
    InMemoryRunStore,
    InMemoryLedgerStore,
    RetryPolicy,
)
from abstractruntime.integrations.abstractcore import create_local_runtime

from .agents.react import ReactAgent
from .tools import ALL_TOOLS


class AgentREPL:
    """Interactive REPL for the ReAct agent."""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.agent: Optional[ReactAgent] = None
        self.paused = False
        self._step_queue: asyncio.Queue = asyncio.Queue()
        
        # Setup tool registry
        self.tool_registry = ToolRegistry()
        for tool in ALL_TOOLS:
            self.tool_registry.register(tool)
        
        # Setup runtime
        self.run_store = InMemoryRunStore()
        self.ledger_store = InMemoryLedgerStore()
        self.runtime = create_local_runtime(
            provider=provider,
            model=model,
            run_store=self.run_store,
            ledger_store=self.ledger_store,
            effect_policy=RetryPolicy(llm_max_attempts=2),
        )
    
    def on_step(self, step: str, data: Dict[str, Any]) -> None:
        """Callback for agent steps - adds to queue for async display."""
        asyncio.get_event_loop().call_soon_threadsafe(
            self._step_queue.put_nowait,
            (step, data)
        )
    
    def print_step(self, step: str, data: Dict[str, Any]) -> None:
        """Print a step to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if step == "init":
            print(f"\n[{timestamp}] 🚀 Starting agent with task: {data.get('task', '')[:50]}...")
        elif step == "reason":
            print(f"[{timestamp}] 🤔 Reasoning (iteration {data.get('iteration', '?')})...")
        elif step == "parse":
            action = data.get('action_type', 'unknown')
            if action == "tool":
                print(f"[{timestamp}] 📋 Decided to use a tool")
            elif action == "answer":
                print(f"[{timestamp}] ✅ Found answer")
            else:
                print(f"[{timestamp}] 💭 Thinking...")
        elif step == "act":
            tool = data.get('tool', 'unknown')
            args = data.get('args', {})
            print(f"[{timestamp}] 🔧 Calling tool: {tool}({args})")
        elif step == "observe":
            result = data.get('result', '')[:100]
            print(f"[{timestamp}] 👁️ Observed: {result}...")
        elif step == "done":
            answer = data.get('answer', '')
            print(f"\n[{timestamp}] ✅ ANSWER: {answer}")
        elif step == "pause":
            print(f"[{timestamp}] ⏸️ Agent paused. Type 'resume' to continue.")
        elif step == "max_iterations":
            print(f"[{timestamp}] ⚠️ Max iterations ({data.get('iterations', '?')}) reached")
    
    async def run_agent_async(self, task: str) -> None:
        """Run the agent asynchronously with step visibility."""
        self.agent = ReactAgent(
            runtime=self.runtime,
            tool_registry=self.tool_registry,
            on_step=self.print_step,  # Direct print for simplicity
        )
        
        self.agent.start(task)
        self.paused = False
        
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        try:
            while True:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                
                state = self.agent.step()
                
                if state.status == RunStatus.COMPLETED:
                    print(f"\n{'='*60}")
                    print(f"Completed in {state.output.get('iterations', '?')} iterations")
                    print(f"{'='*60}\n")
                    break
                elif state.status == RunStatus.WAITING:
                    print("\n⏸️ Agent is waiting. Type 'resume' to continue or 'cancel' to stop.")
                    break
                elif state.status == RunStatus.FAILED:
                    print(f"\n❌ Agent failed: {state.error}")
                    break
                
                # Small delay to allow for interrupt
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            print("\n⏹️ Agent interrupted")
    
    def pause(self) -> None:
        """Pause the agent."""
        self.paused = True
        print("⏸️ Pausing agent...")
    
    def resume(self) -> None:
        """Resume the agent."""
        if self.agent and self.agent.get_state():
            state = self.agent.get_state()
            if state.status == RunStatus.WAITING:
                print("▶️ Resuming agent...")
                self.agent.resume()
            else:
                self.paused = False
                print("▶️ Continuing agent...")
    
    def show_status(self) -> None:
        """Show current agent status."""
        if not self.agent:
            print("No active agent")
            return
        
        state = self.agent.get_state()
        if not state:
            print("No active run")
            return
        
        print(f"\nAgent Status:")
        print(f"  Run ID: {self.agent.run_id}")
        print(f"  Status: {state.status.value}")
        print(f"  Current Node: {state.current_node}")
        print(f"  Iteration: {state.vars.get('iteration', 0)}")
        
        if state.status == RunStatus.WAITING:
            print(f"  Waiting for: {state.waiting.wait_key}")
        
        history = state.vars.get('history', [])
        if history:
            print(f"  History entries: {len(history)}")
    
    def show_history(self) -> None:
        """Show agent conversation history."""
        if not self.agent:
            print("No active agent")
            return
        
        state = self.agent.get_state()
        if not state:
            print("No active run")
            return
        
        history = state.vars.get('history', [])
        if not history:
            print("No history yet")
            return
        
        print("\nConversation History:")
        print("-" * 40)
        for i, entry in enumerate(history):
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')[:200]
            print(f"[{i+1}] {role}: {content}")
            if len(entry.get('content', '')) > 200:
                print("    ...")
        print("-" * 40)
    
    def show_help(self) -> None:
        """Show help message."""
        print("""
ReAct Agent REPL Commands:
  <task>     - Start agent with a task (e.g., "list files in current directory")
  pause      - Pause the running agent
  resume     - Resume a paused agent
  status     - Show current agent status
  history    - Show conversation history
  tools      - List available tools
  help       - Show this help message
  quit/exit  - Exit the REPL

Examples:
  > list the python files in this directory
  > what is in the README.md file?
  > search for all markdown files
""")
    
    def show_tools(self) -> None:
        """Show available tools."""
        print("\nAvailable Tools:")
        for tool in self.tool_registry.list_tools():
            params = ", ".join(f"{k}" for k in tool.parameters.keys())
            print(f"  {tool.name}({params})")
            print(f"    {tool.description}")
        print()
    
    async def repl_loop(self) -> None:
        """Main REPL loop."""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ReAct Agent REPL                          ║
║                                                              ║
║  Provider: {self.provider:<20} Model: {self.model:<15}  ║
║                                                              ║
║  Type 'help' for commands, or enter a task to start.        ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        self.show_tools()
        
        agent_task: Optional[asyncio.Task] = None
        
        while True:
            try:
                # Get input
                if sys.stdin.isatty():
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\n> ").strip()
                    )
                else:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    user_input = line.strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                cmd = user_input.lower()
                
                if cmd in ('quit', 'exit', 'q'):
                    if agent_task and not agent_task.done():
                        agent_task.cancel()
                        try:
                            await agent_task
                        except asyncio.CancelledError:
                            pass
                    print("Goodbye!")
                    break
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd == 'tools':
                    self.show_tools()
                
                elif cmd == 'status':
                    self.show_status()
                
                elif cmd == 'history':
                    self.show_history()
                
                elif cmd == 'pause':
                    self.pause()
                
                elif cmd == 'resume':
                    self.resume()
                
                else:
                    # Treat as a task
                    if agent_task and not agent_task.done():
                        print("Agent is already running. Use 'pause' or wait for completion.")
                        continue
                    
                    agent_task = asyncio.create_task(self.run_agent_async(user_input))
                    await agent_task
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                if agent_task and not agent_task.done():
                    agent_task.cancel()
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        pass
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Entry point for the REPL."""
    parser = argparse.ArgumentParser(description="ReAct Agent REPL")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="gemma3:1b-it-q4_K_M", help="Model name")
    args = parser.parse_args()
    
    repl = AgentREPL(provider=args.provider, model=args.model)
    asyncio.run(repl.repl_loop())


if __name__ == "__main__":
    main()
