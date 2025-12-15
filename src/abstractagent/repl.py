"""Async REPL for the ReAct agent.

Provides an interactive interface with:
- Real-time step visibility
- Pause/resume capability
- Interactive question handling
- Run persistence for resume across sessions
"""

import asyncio
import sys
import argparse
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from abstractcore.tools import ToolDefinition
from abstractruntime import (
    RunStatus,
    InMemoryRunStore,
    InMemoryLedgerStore,
    JsonFileRunStore,
    JsonlLedgerStore,
)
from abstractruntime.integrations.abstractcore import MappingToolExecutor, create_local_runtime

from .agents.react import ReactAgent
from .tools import ALL_TOOLS
from .ui.question import get_user_response_async, Colors, _c


class AgentREPL:
    """Interactive REPL for the ReAct agent."""
    
    def __init__(self, provider: str, model: str, state_file: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.state_file = state_file
        self.agent: Optional[ReactAgent] = None
        self._interrupted = False
        self._tools = list(ALL_TOOLS)
        
        # Setup runtime with persistence. If a state file is provided, use
        # file-backed stores so runs can resume across process restarts.
        if self.state_file:
            base_dir = Path(self.state_file).expanduser().resolve()
            store_dir = base_dir.with_name(base_dir.name + ".d")
            self.run_store = JsonFileRunStore(store_dir)
            self.ledger_store = JsonlLedgerStore(store_dir)
        else:
            self.run_store = InMemoryRunStore()
            self.ledger_store = InMemoryLedgerStore()

        self.runtime = create_local_runtime(
            provider=provider,
            model=model,
            run_store=self.run_store,
            ledger_store=self.ledger_store,
            tool_executor=MappingToolExecutor.from_tools(self._tools),
        )

        self.agent = ReactAgent(
            runtime=self.runtime,
            tools=self._tools,
            on_step=self.print_step,
        )
    
    def print_step(self, step: str, data: Dict[str, Any]) -> None:
        """Print a step to the console with formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        ts = _c(f"[{timestamp}]", Colors.DIM)
        
        if step == "init":
            task = data.get('task', '')[:60]
            print(f"\n{ts} {_c('Starting:', Colors.CYAN, Colors.BOLD)} {task}")
        elif step == "reason":
            iteration = data.get('iteration', '?')
            print(f"{ts} {_c(f'Thinking (step {iteration})...', Colors.YELLOW)}")
        elif step == "parse":
            has_tools = data.get('has_tool_calls', False)
            if has_tools:
                print(f"{ts} {_c('Decided to use tools', Colors.BLUE)}")
        elif step == "act":
            tool = data.get('tool', 'unknown')
            args = data.get('args', {})
            args_str = json.dumps(args) if args else ""
            if len(args_str) > 50:
                args_str = args_str[:47] + "..."
            print(f"{ts} {_c('Tool:', Colors.GREEN)} {tool}({args_str})")
        elif step == "observe":
            result = data.get('result', '')[:80]
            print(f"{ts} {_c('Result:', Colors.DIM)} {result}")
        elif step == "ask_user":
            print(f"{ts} {_c('Agent has a question...', Colors.MAGENTA, Colors.BOLD)}")
        elif step == "user_response":
            response = data.get('response', '')[:50]
            print(f"{ts} {_c('You answered:', Colors.MAGENTA)} {response}")
        elif step == "done":
            answer = data.get('answer', '')
            print(f"\n{ts} {_c('ANSWER:', Colors.GREEN, Colors.BOLD)}")
            print(_c("─" * 60, Colors.DIM))
            print(answer)
            print(_c("─" * 60, Colors.DIM))
        elif step == "max_iterations":
            print(f"{ts} {_c('Max iterations reached', Colors.YELLOW)}")
    
    async def handle_waiting_state(self) -> bool:
        """Handle agent waiting state (questions).
        
        Returns True if handled and should continue, False to stop.
        """
        if not self.agent or not self.agent.is_waiting():
            return False
        
        question = self.agent.get_pending_question()
        if not question:
            return False
        
        # Get user response via UI
        response = await get_user_response_async(
            prompt=question.get("prompt", "Please respond:"),
            choices=question.get("choices"),
            allow_free_text=question.get("allow_free_text", True),
        )
        
        if not response:
            print(_c("No response provided. Agent paused.", Colors.YELLOW))
            return False
        
        # Resume agent with response
        self.agent.resume(response)
        return True
    
    async def run_agent_async(self, task: str) -> None:
        """Run the agent asynchronously with step visibility."""
        self.agent.start(task)
        if self.state_file:
            self.agent.save_state(self.state_file)
        self._interrupted = False
        
        print(f"\n{_c('═' * 60, Colors.CYAN)}")
        print(f"{_c('Task:', Colors.BOLD)} {task}")
        print(f"{_c('═' * 60, Colors.CYAN)}")
        
        try:
            while not self._interrupted:
                state = self.agent.step()
                
                if state.status == RunStatus.COMPLETED:
                    print(f"\n{_c('═' * 60, Colors.GREEN)}")
                    print(f"{_c('Completed', Colors.GREEN, Colors.BOLD)} in {state.output.get('iterations', '?')} steps")
                    print(f"{_c('═' * 60, Colors.GREEN)}")
                    if self.state_file:
                        self.agent.clear_state(self.state_file)
                    break
                    
                elif state.status == RunStatus.WAITING:
                    # Handle question
                    handled = await self.handle_waiting_state()
                    if not handled:
                        print(f"\n{_c('Agent paused.', Colors.YELLOW)} Type 'resume' to continue.")
                        break
                    # After handling, continue the loop to process next step
                    continue
                    
                elif state.status == RunStatus.FAILED:
                    print(f"\n{_c('Failed:', Colors.YELLOW)} {state.error}")
                    if self.state_file:
                        self.agent.clear_state(self.state_file)
                    break
                
                # Small delay for interrupt handling
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            print(f"\n{_c('Interrupted', Colors.YELLOW)}")
            self._interrupted = True
    
    def interrupt(self) -> None:
        """Interrupt the running agent."""
        self._interrupted = True
        print(f"\n{_c('Interrupting...', Colors.YELLOW)} (state preserved)")
    
    async def resume_agent(self) -> None:
        """Resume a paused agent."""
        if not self.agent:
            print("No agent to resume. Start a new task.")
            return
        
        state = self.agent.get_state()
        if not state:
            print("No active run.")
            return
        
        if state.status == RunStatus.WAITING:
            handled = await self.handle_waiting_state()
            if handled:
                # Continue running after handling
                await self._continue_running()
        elif state.status == RunStatus.RUNNING:
            self._interrupted = False
            await self._continue_running()
        else:
            print(f"Agent is {state.status.value}, cannot resume.")
    
    async def _continue_running(self) -> None:
        """Continue running the agent after resume."""
        try:
            while not self._interrupted:
                state = self.agent.step()
                
                if state.status == RunStatus.COMPLETED:
                    print(f"\n{_c('═' * 60, Colors.GREEN)}")
                    print(f"{_c('Completed', Colors.GREEN, Colors.BOLD)} in {state.output.get('iterations', '?')} steps")
                    print(f"{_c('═' * 60, Colors.GREEN)}")
                    break
                elif state.status == RunStatus.WAITING:
                    handled = await self.handle_waiting_state()
                    if not handled:
                        print(f"\n{_c('Agent paused.', Colors.YELLOW)} Type 'resume' to continue.")
                        break
                    continue
                elif state.status == RunStatus.FAILED:
                    print(f"\n{_c('Failed:', Colors.YELLOW)} {state.error}")
                    break
                
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self._interrupted = True
    
    def show_status(self) -> None:
        """Show current agent status."""
        if not self.agent:
            print("No active agent")
            return
        
        state = self.agent.get_state()
        if not state:
            print("No active run")
            return
        
        print(f"\n{_c('Agent Status', Colors.CYAN, Colors.BOLD)}")
        print(_c("─" * 40, Colors.DIM))
        print(f"  Run ID:    {state.run_id[:16]}...")
        print(f"  Status:    {_c(state.status.value, Colors.GREEN if state.status == RunStatus.COMPLETED else Colors.YELLOW)}")
        print(f"  Node:      {state.current_node}")
        print(f"  Iteration: {state.vars.get('iteration', 0)}")
        
        if state.status == RunStatus.WAITING and state.waiting:
            print(f"\n  {_c('Waiting for:', Colors.MAGENTA)} {state.waiting.reason.value}")
            if state.waiting.prompt:
                print(f"  {_c('Question:', Colors.MAGENTA)} {state.waiting.prompt[:50]}...")
        
        print(_c("─" * 40, Colors.DIM))
    
    def show_history(self) -> None:
        """Show agent conversation history."""
        if not self.agent:
            print("No active agent")
            return
        
        state = self.agent.get_state()
        if not state:
            print("No active run")
            return
        
        history = state.vars.get('messages', [])
        if not history:
            print("No history yet")
            return
        
        print(f"\n{_c('Conversation History', Colors.CYAN, Colors.BOLD)}")
        print(_c("─" * 60, Colors.DIM))
        
        for i, entry in enumerate(history):
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            
            if role == "assistant":
                role_color = Colors.GREEN
            elif role == "tool":
                role_color = Colors.BLUE
            elif role == "user":
                role_color = Colors.MAGENTA
            else:
                role_color = Colors.DIM
            
            # Truncate long content
            if len(content) > 100:
                content = content[:97] + "..."
            
            print(f"[{i+1}] {_c(role, role_color, Colors.BOLD)}: {content}")
        
        print(_c("─" * 60, Colors.DIM))
    
    def show_help(self) -> None:
        """Show help message."""
        print(f"""
{_c('ReAct Agent REPL', Colors.CYAN, Colors.BOLD)}
{_c('─' * 40, Colors.DIM)}

{_c('Commands:', Colors.BOLD)}
  <task>      Start agent with a task
  resume      Resume a paused/interrupted agent
  status      Show current agent status
  history     Show conversation history
  tools       List available tools
  help        Show this help
  quit        Exit

{_c('During execution:', Colors.BOLD)}
  Ctrl+C      Interrupt (preserves state)

{_c('Examples:', Colors.DIM)}
  > list the python files in this directory
  > what is in the README.md file?
  > search for TODO comments in the code
""")
    
    def show_tools(self) -> None:
        """Show available tools."""
        print(f"\n{_c('Available Tools', Colors.CYAN, Colors.BOLD)}")
        print(_c("─" * 50, Colors.DIM))
        
        for tool in self._tools:
            tool_def = getattr(tool, "_tool_definition", None)
            if tool_def is None:
                tool_def = ToolDefinition.from_function(tool)

            params = ", ".join(str(k) for k in (tool_def.parameters or {}).keys())
            print(f"  {_c(tool_def.name, Colors.GREEN)}({params})")
            print(f"    {_c(tool_def.description, Colors.DIM)}")
        
        # Show built-in ask_user
        print(f"  {_c('ask_user', Colors.MAGENTA)}(question, choices?)")
        print(f"    {_c('Ask the user a question', Colors.DIM)}")
        
        print(_c("─" * 50, Colors.DIM))
    
    async def repl_loop(self) -> None:
        """Main REPL loop."""
        # Header
        print(f"""
{_c('╔' + '═' * 58 + '╗', Colors.CYAN)}
{_c('║', Colors.CYAN)}  {_c('ReAct Agent REPL', Colors.BOLD)}                                   {_c('║', Colors.CYAN)}
{_c('║', Colors.CYAN)}                                                          {_c('║', Colors.CYAN)}
{_c('║', Colors.CYAN)}  Provider: {self.provider:<15} Model: {self.model:<17} {_c('║', Colors.CYAN)}
{_c('║', Colors.CYAN)}                                                          {_c('║', Colors.CYAN)}
{_c('║', Colors.CYAN)}  Type {_c("'help'", Colors.GREEN)} for commands, or enter a task.          {_c('║', Colors.CYAN)}
{_c('╚' + '═' * 58 + '╝', Colors.CYAN)}
""")
        
        self.show_tools()

        if self.state_file:
            try:
                loaded = self.agent.load_state(self.state_file)
                if loaded is not None:
                    print(f"\n{_c('Loaded saved run.', Colors.CYAN)} Type 'status' or 'resume'.")
            except Exception as e:
                print(f"{_c('State load failed:', Colors.YELLOW)} {e}")
        
        agent_task: Optional[asyncio.Task] = None
        
        while True:
            try:
                # Get input
                if sys.stdin.isatty():
                    prompt = f"\n{_c('>', Colors.CYAN, Colors.BOLD)} "
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(prompt).strip()
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
                    print(_c("Goodbye!", Colors.CYAN))
                    break
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd == 'tools':
                    self.show_tools()
                
                elif cmd == 'status':
                    self.show_status()
                
                elif cmd == 'history':
                    self.show_history()
                
                elif cmd == 'resume':
                    await self.resume_agent()
                
                else:
                    # Treat as a task
                    if agent_task and not agent_task.done():
                        print(_c("Agent is running. Use Ctrl+C to interrupt.", Colors.YELLOW))
                        continue
                    
                    agent_task = asyncio.create_task(self.run_agent_async(user_input))
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        pass
                    
            except KeyboardInterrupt:
                print()
                if agent_task and not agent_task.done():
                    self.interrupt()
                    agent_task.cancel()
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        pass
                else:
                    print(_c("Type 'quit' to exit.", Colors.DIM))
            except EOFError:
                break
            except Exception as e:
                print(f"{_c('Error:', Colors.YELLOW)} {e}")


def main():
    """Entry point for the REPL."""
    parser = argparse.ArgumentParser(description="ReAct Agent REPL")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="qwen3:1.7b-q4_K_M", help="Model name")
    parser.add_argument("--state-file", help="File to persist agent state")
    args = parser.parse_args()
    
    repl = AgentREPL(
        provider=args.provider,
        model=args.model,
        state_file=args.state_file,
    )
    asyncio.run(repl.repl_loop())


if __name__ == "__main__":
    main()
