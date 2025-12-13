"""ReAct Agent implementation using AbstractRuntime and AbstractCore.

ReAct = Reason + Act + Observe loop.

The agent uses AbstractCore's tool handling:
- Tools are passed to LLM via AbstractCore's generate(tools=...)
- AbstractCore handles prompt formatting for the model
- AbstractCore parses tool calls from responses

Special tool: ask_user
- Agent can ask the user questions with optional choices
- Triggers ASK_USER effect for durable pause/resume
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
from abstractcore.tools import ToolRegistry, ToolCall, ToolDefinition


MAX_ITERATIONS = 10

# Built-in ask_user tool definition
ASK_USER_TOOL = ToolDefinition(
    name="ask_user",
    description="Ask the user a question when you need clarification or input. Use this when the task is ambiguous or you need the user to make a choice.",
    parameters={
        "question": {
            "type": "string",
            "description": "The question to ask the user (required)",
        },
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of choices for the user to pick from",
        },
    },
    when_to_use="When the task is ambiguous or you need user input to proceed",
)


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
    
    # Include ask_user tool alongside registered tools
    tools = tool_registry.list_tools() + [ASK_USER_TOOL]
    
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
            
            # Special handling for ask_user - triggers ASK_USER effect
            if name == "ask_user":
                question = args.get("question", "Please provide input:")
                choices = args.get("choices", [])
                
                # Store remaining tool calls for after resume
                remaining = tool_calls[tool_calls.index(tc) + 1:] if isinstance(tool_calls, list) else []
                run.vars["pending_tool_calls"] = remaining
                
                emit("ask_user", {"question": question, "choices": choices})
                
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.ASK_USER,
                        payload={
                            "prompt": question,
                            "choices": choices if choices else None,
                            "allow_free_text": True,
                        },
                        result_key="user_response",
                    ),
                    next_node="handle_user_response",
                )
            
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
    
    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        """Handle user response after ask_user."""
        user_response = run.vars.get("user_response", {})
        response_text = user_response.get("response", "")
        
        emit("user_response", {"response": response_text})
        
        # Add user response to messages
        messages = run.vars.get("messages", [])
        messages.append({
            "role": "user",
            "content": f"[User response]: {response_text}"
        })
        run.vars["messages"] = messages
        
        # Continue with any remaining tool calls or back to reasoning
        if run.vars.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")
    
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
            "handle_user_response": handle_user_response_node,
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
    
    def is_waiting(self) -> bool:
        """Check if agent is waiting for user input."""
        state = self.get_state()
        return state is not None and state.status == RunStatus.WAITING
    
    def get_pending_question(self) -> Optional[Dict[str, Any]]:
        """Get pending question if agent is waiting for user input.
        
        Returns dict with: prompt, choices (optional), allow_free_text
        """
        state = self.get_state()
        if not state or state.status != RunStatus.WAITING or not state.waiting:
            return None
        
        return {
            "prompt": state.waiting.prompt,
            "choices": state.waiting.choices,
            "allow_free_text": state.waiting.allow_free_text,
            "wait_key": state.waiting.wait_key,
        }
    
    def resume(self, response: str) -> RunState:
        """Resume agent with user response.
        
        Args:
            response: User's answer to the pending question
            
        Returns:
            Updated RunState after resuming
        """
        if not self._current_run_id:
            raise RuntimeError("No active run.")
        
        state = self.get_state()
        if not state or state.status != RunStatus.WAITING:
            raise RuntimeError("Agent is not waiting for input.")
        
        wait_key = state.waiting.wait_key if state.waiting else None
        
        return self.runtime.resume(
            workflow=self.workflow,
            run_id=self._current_run_id,
            wait_key=wait_key,
            payload={"response": response},
        )
    
    def attach(self, run_id: str) -> RunState:
        """Attach to an existing run for resume.
        
        Args:
            run_id: ID of the run to attach to
            
        Returns:
            Current RunState
        """
        state = self.runtime.get_state(run_id)
        self._current_run_id = run_id
        return state
    
    @property
    def run_id(self) -> Optional[str]:
        return self._current_run_id
