"""ReAct Agent implementation using AbstractRuntime and AbstractCore.

ReAct = Reason + Act + Observe loop.

The agent uses AbstractCore's tool handling:
- Tools are passed to LLM via AbstractCore's generate(tools=...)
- AbstractCore handles prompt formatting for the model
- AbstractCore parses tool calls from responses
"""

from typing import Any, Dict, List, Optional, Callable

from abstractruntime import (
    Runtime,
    RunState,
    RunStatus,
    StepPlan,
    Effect,
    EffectType,
    WorkflowSpec,
)
from abstractcore.tools import ToolRegistry, ToolCall


MAX_ITERATIONS = 10


def create_react_workflow(
    tool_registry: ToolRegistry,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> WorkflowSpec:
    """Create a ReAct agent workflow using AbstractCore's tool handling.
    
    Args:
        tool_registry: Registry containing available tools
        on_step: Optional callback for step visibility (step_name, data)
    
    Returns:
        WorkflowSpec for the ReAct agent
    """
    
    tools = tool_registry.list_tools()
    
    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)
    
    def init_node(run: RunState, ctx) -> StepPlan:
        """Initialize the agent state."""
        run.vars["iteration"] = 0
        run.vars["messages"] = []
        emit("init", {"task": run.vars.get("task", "")})
        return StepPlan(node_id="init", next_node="reason")
    
    def reason_node(run: RunState, ctx) -> StepPlan:
        """Call LLM with tools - AbstractCore handles formatting."""
        iteration = run.vars.get("iteration", 0)
        
        if iteration >= MAX_ITERATIONS:
            return StepPlan(node_id="reason", next_node="max_iterations")
        
        run.vars["iteration"] = iteration + 1
        
        task = run.vars.get("task", "")
        messages = run.vars.get("messages", [])
        
        # Build messages for the LLM
        if not messages:
            # First turn - just the task
            prompt = f"Task: {task}\n\nUse the available tools to complete this task. When done, provide your final answer."
        else:
            # Subsequent turns - include history
            history_text = "\n".join([
                f"{m['role']}: {m['content']}" for m in messages[-6:]
            ])
            prompt = f"Task: {task}\n\nHistory:\n{history_text}\n\nContinue working on the task. Use tools or provide final answer."
        
        emit("reason", {"iteration": iteration + 1})
        
        # Convert tools to format AbstractCore expects
        tool_defs = [t.to_dict() for t in tools]
        
        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt,
                    "tools": tool_defs,  # AbstractCore handles tool formatting
                },
                result_key="llm_response",
            ),
            next_node="parse",
        )
    
    def parse_node(run: RunState, ctx) -> StepPlan:
        """Parse LLM response - check for tool calls or final answer."""
        response = run.vars.get("llm_response", {})
        content = response.get("content", "")
        tool_calls = response.get("tool_calls") or []
        
        # Add assistant message to history
        messages = run.vars.get("messages", [])
        messages.append({"role": "assistant", "content": content})
        run.vars["messages"] = messages
        
        emit("parse", {
            "has_tool_calls": len(tool_calls) > 0,
            "content_preview": content[:100] if content else "(no content)",
        })
        
        if tool_calls:
            # Store tool calls for execution
            run.vars["pending_tool_calls"] = tool_calls
            return StepPlan(node_id="parse", next_node="act")
        else:
            # No tool calls - treat content as final answer
            run.vars["final_answer"] = content
            return StepPlan(node_id="parse", next_node="done")
    
    def act_node(run: RunState, ctx) -> StepPlan:
        """Execute pending tool calls."""
        tool_calls = run.vars.get("pending_tool_calls", [])
        messages = run.vars.get("messages", [])
        
        results = []
        for tc in tool_calls:
            # Handle both dict and ToolCall formats
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                call_id = tc.get("call_id", "1")
            else:
                name = tc.name
                args = tc.arguments
                call_id = tc.call_id or "1"
            
            emit("act", {"tool": name, "args": args})
            
            # Execute via registry
            tool_call = ToolCall(name=name, arguments=args, call_id=call_id)
            result = tool_registry.execute_tool(tool_call)
            
            if result.success:
                output = str(result.output)
            else:
                output = f"Error: {result.error}"
            
            results.append({"tool": name, "result": output})
            emit("observe", {"tool": name, "result": output[:150]})
        
        # Add tool results to messages
        for r in results:
            messages.append({
                "role": "tool",
                "content": f"[{r['tool']}]: {r['result']}"
            })
        run.vars["messages"] = messages
        
        # Clear pending and continue reasoning
        run.vars["pending_tool_calls"] = []
        return StepPlan(node_id="act", next_node="reason")
    
    def done_node(run: RunState, ctx) -> StepPlan:
        """Complete with final answer."""
        answer = run.vars.get("final_answer", "No answer provided")
        emit("done", {"answer": answer})
        
        return StepPlan(
            node_id="done",
            complete_output={
                "answer": answer,
                "iterations": run.vars.get("iteration", 0),
                "messages": run.vars.get("messages", []),
            },
        )
    
    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        """Handle max iterations reached."""
        emit("max_iterations", {"iterations": MAX_ITERATIONS})
        
        # Use last content as answer
        messages = run.vars.get("messages", [])
        last_content = messages[-1]["content"] if messages else "Max iterations reached"
        
        return StepPlan(
            node_id="max_iterations",
            complete_output={
                "answer": last_content,
                "iterations": MAX_ITERATIONS,
                "messages": messages,
            },
        )
    
    return WorkflowSpec(
        workflow_id="react_agent",
        entry_node="init",
        nodes={
            "init": init_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )


class ReactAgent:
    """ReAct agent with pause/resume capability."""
    
    def __init__(
        self,
        runtime: Runtime,
        tool_registry: ToolRegistry,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.runtime = runtime
        self.tool_registry = tool_registry
        self.on_step = on_step
        self.workflow = create_react_workflow(tool_registry, on_step)
        self._current_run_id: Optional[str] = None
    
    def start(self, task: str) -> str:
        """Start a new agent run with a task."""
        self._current_run_id = self.runtime.start(
            workflow=self.workflow,
            vars={"task": task},
        )
        return self._current_run_id
    
    def step(self) -> RunState:
        """Execute one step of the agent."""
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        
        return self.runtime.tick(
            workflow=self.workflow,
            run_id=self._current_run_id,
        )
    
    def run_to_completion(self) -> RunState:
        """Run the agent until completion or waiting."""
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        
        state = self.step()
        while state.status == RunStatus.RUNNING:
            state = self.step()
        
        return state
    
    def get_state(self) -> Optional[RunState]:
        """Get current agent state."""
        if not self._current_run_id:
            return None
        return self.runtime.get_state(self._current_run_id)
    
    @property
    def run_id(self) -> Optional[str]:
        return self._current_run_id
