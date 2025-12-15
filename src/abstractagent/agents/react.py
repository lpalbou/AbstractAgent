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

import hashlib
import json
from typing import Any, Dict, List, Optional, Callable, Union

from abstractruntime import (
    Runtime,
    RunState,
    RunStatus,
    StepPlan,
    Effect,
    EffectType,
    WorkflowSpec,
)
from abstractcore.tools import ToolCall, ToolDefinition

from .base import BaseAgent


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

def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_react_workflow(
    tools: List[Callable],
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> WorkflowSpec:
    """Create a ReAct agent workflow using AbstractCore's tool handling.
    
    Args:
        tools: List of tool functions decorated with @tool
        on_step: Optional callback for step visibility (step_name, data)
    
    Returns:
        WorkflowSpec for the ReAct agent
    """
    
    # Convert tool functions to ToolDefinitions for LLM prompt
    tool_defs = []
    for t in tools:
        if hasattr(t, '_tool_definition'):
            tool_defs.append(t._tool_definition)
        else:
            # Fallback: create definition from function
            tool_defs.append(ToolDefinition.from_function(t))
    
    # Include ask_user tool
    tool_defs.append(ASK_USER_TOOL)
    
    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)
    
    def init_node(run: RunState, ctx) -> StepPlan:
        """Initialize the agent state."""
        run.vars["iteration"] = 0
        messages = run.vars.get("messages")
        if not isinstance(messages, list):
            messages = []

        task = str(run.vars.get("task", "") or "")
        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append({"role": "user", "content": task})

        run.vars["messages"] = messages
        emit("init", {"task": task})
        return StepPlan(node_id="init", next_node="reason")
    
    def reason_node(run: RunState, ctx) -> StepPlan:
        """Call LLM with tools - AbstractCore handles formatting."""
        iteration = run.vars.get("iteration", 0)
        
        if iteration >= MAX_ITERATIONS:
            return StepPlan(node_id="reason", next_node="max_iterations")
        
        run.vars["iteration"] = iteration + 1
        
        task = run.vars.get("task", "")
        messages = run.vars.get("messages", [])
        if not isinstance(messages, list):
            messages = []
            run.vars["messages"] = messages
        
        # Check for injected messages (async guidance from user)
        inbox = run.vars.get("_inbox", [])
        inbox_text = ""
        if inbox:
            inbox_messages = [m.get("content", "") for m in inbox]
            inbox_text = "\n\n[User guidance]: " + " | ".join(inbox_messages)
            run.vars["_inbox"] = []  # Clear inbox after reading
        
        # Build messages for the LLM
        if len(messages) <= 1:
            prompt = (
                f"Task: {task}\n\n"
                "Use the available tools to complete this task. When done, provide your final answer."
            )
        else:
            history_text = "\n".join(
                [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages[-12:]]
            )
            prompt = (
                "Continue the conversation and work on the user's latest request.\n\n"
                f"History:\n{history_text}\n\n"
                "Use tools or provide a final answer."
            )
        
        # Append any injected guidance
        if inbox_text:
            prompt += inbox_text
        
        emit("reason", {"iteration": iteration + 1, "has_guidance": bool(inbox_text)})
        
        # Convert tools to format AbstractCore expects
        tool_dicts = [t.to_dict() for t in tool_defs]
        
        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload={
                    "prompt": prompt,
                    "tools": tool_dicts,  # AbstractCore handles tool formatting
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
        """Execute pending tool calls via TOOL_CALLS effect.
        
        This uses the runtime's effect system so tool calls are:
        - Recorded in the ledger
        - Subject to retry/idempotency policies
        - Consistent with the effect-based architecture
        """
        tool_calls = run.vars.get("pending_tool_calls", [])
        
        if not tool_calls:
            return StepPlan(node_id="act", next_node="reason")
        
        # Check for ask_user - handle specially with ASK_USER effect
        for i, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = tc.get("arguments", {})
            else:
                name = tc.name
                args = tc.arguments
            
            if name == "ask_user":
                question = args.get("question", "Please provide input:")
                choices = args.get("choices", [])
                
                # Store remaining tool calls for after resume
                run.vars["pending_tool_calls"] = tool_calls[i + 1:]
                
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
        
        # Emit act events for visibility
        for tc in tool_calls:
            if isinstance(tc, dict):
                emit("act", {"tool": tc.get("name", ""), "args": tc.get("arguments", {})})
            else:
                emit("act", {"tool": tc.name, "args": tc.arguments})
        
        # Use TOOL_CALLS effect for ledger recording
        # Format tool calls for the effect
        formatted_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                formatted_calls.append({
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                    "call_id": tc.get("call_id", "1"),
                })
            else:
                formatted_calls.append({
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.call_id or "1",
                })
        
        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls},
                result_key="tool_results",
            ),
            next_node="observe",
        )
    
    def observe_node(run: RunState, ctx) -> StepPlan:
        """Process tool results and add to conversation history."""
        tool_results = run.vars.get("tool_results", {})
        results = tool_results.get("results", [])
        messages = run.vars.get("messages", [])
        
        for r in results:
            name = r.get("name", "tool")
            if r.get("success"):
                output = r.get("output", "")
            else:
                output = f"Error: {r.get('error', 'unknown error')}"
            
            emit("observe", {"tool": name, "result": str(output)[:150]})
            
            messages.append({
                "role": "tool",
                "content": f"[{name}]: {output}"
            })
        
        run.vars["messages"] = messages
        run.vars["pending_tool_calls"] = []
        
        return StepPlan(node_id="observe", next_node="reason")
    
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
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )


class ReactAgent(BaseAgent):
    """ReAct agent with pause/resume capability.
    
    Inherits common functionality from BaseAgent:
    - run_to_completion(), get_state(), is_waiting(), is_running(), is_complete()
    - get_pending_question(), resume(), attach()
    - save_state(), load_state(), clear_state()
    - cancel(), get_ledger(), inject_message()
    - get_output(), get_error()
    """
    
    def _create_workflow(self) -> WorkflowSpec:
        """Create the ReAct workflow."""
        return create_react_workflow(self.tools, self.on_step)
    
    def start(self, task: str) -> str:
        """Start a new agent run with a task."""
        actor_id = self._ensure_actor_id()
        seeded_messages = [dict(m) for m in (self.session_messages or [])]
        tool_defs: List[ToolDefinition] = []
        for t in self.tools:
            tool_defs.append(getattr(t, "_tool_definition", None) or ToolDefinition.from_function(t))
        tool_defs.append(ASK_USER_TOOL)

        tool_specs = [t.to_dict() for t in tool_defs]
        toolset_id = _compute_toolset_id(tool_specs)

        self._current_run_id = self.runtime.start(
            workflow=self.workflow,
            vars={
                "task": task,
                "messages": seeded_messages,
                "_runtime": {
                    "tool_specs": tool_specs,
                    "toolset_id": toolset_id,
                },
            },
            actor_id=actor_id,
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


def create_react_agent(
    provider: str = "ollama",
    model: str = "qwen3:4b-instruct-2507-q4_K_M",
    tools: Optional[List[Callable]] = None,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ReactAgent:
    """Create a ReAct agent with minimal configuration.
    
    Args:
        provider: LLM provider (default: "ollama")
        model: Model name (default: "qwen3:4b-instruct-2507-q4_K_M")
        tools: List of tool functions decorated with @tool (default: ALL_TOOLS)
        on_step: Optional callback for step visibility
        
    Returns:
        Configured ReactAgent ready to use
        
    Example:
        agent = create_react_agent()
        agent.start("List files in current directory")
        state = agent.run_to_completion()
        print(state.output["answer"])
    """
    from abstractruntime.integrations.abstractcore import create_local_runtime
    from abstractruntime.integrations.abstractcore import MappingToolExecutor
    from ..tools import ALL_TOOLS
    
    # Use default tools if none provided
    if tools is None:
        tools = list(ALL_TOOLS)

    # Create runtime with an explicit tool executor (durable: no callables in RunState)
    runtime = create_local_runtime(
        provider=provider,
        model=model,
        tool_executor=MappingToolExecutor.from_tools(tools),
    )
    
    # Create and return agent with tools
    return ReactAgent(
        runtime=runtime,
        tools=tools,
        on_step=on_step,
    )
