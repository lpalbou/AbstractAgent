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
from abstractruntime.core.vars import ensure_namespaces
from abstractcore.tools import ToolCall, ToolDefinition

from .base import BaseAgent


DEFAULT_MAX_ITERATIONS = 20
# Legacy alias (kept for compatibility with older code/docs).
MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS

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


def _new_message(
    ctx: Any,
    *,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp: Optional[str] = None
    now_iso = getattr(ctx, "now_iso", None)
    if callable(now_iso):
        timestamp = str(now_iso())
    if not timestamp:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def _ensure_react_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place.

    Returns: (context, scratchpad, runtime_ns, temp)
    """
    ensure_namespaces(run.vars)
    context = run.vars["context"]
    scratchpad = run.vars["scratchpad"]
    runtime_ns = run.vars["_runtime"]
    temp = run.vars["_temp"]

    # Legacy (flat) -> namespaced migration. This keeps resume working for older persisted runs.
    if "task" in run.vars and "task" not in context:
        context["task"] = run.vars.pop("task")
    if "messages" in run.vars and "messages" not in context:
        context["messages"] = run.vars.pop("messages")
    if "iteration" in run.vars and "iteration" not in scratchpad:
        scratchpad["iteration"] = run.vars.pop("iteration")
    if "max_iterations" in run.vars and "max_iterations" not in scratchpad:
        scratchpad["max_iterations"] = run.vars.pop("max_iterations")
    if "_inbox" in run.vars and "inbox" not in runtime_ns:
        runtime_ns["inbox"] = run.vars.pop("_inbox")

    for key in ("llm_response", "tool_results", "pending_tool_calls", "user_response", "final_answer"):
        if key in run.vars and key not in temp:
            temp[key] = run.vars.pop(key)

    if not isinstance(context.get("messages"), list):
        context["messages"] = []
    if not isinstance(runtime_ns.get("inbox"), list):
        runtime_ns["inbox"] = []

    iteration = scratchpad.get("iteration")
    if not isinstance(iteration, int):
        try:
            scratchpad["iteration"] = int(iteration or 0)
        except (TypeError, ValueError):
            scratchpad["iteration"] = 0

    max_iterations = scratchpad.get("max_iterations")
    if max_iterations is None:
        scratchpad["max_iterations"] = DEFAULT_MAX_ITERATIONS
    elif not isinstance(max_iterations, int):
        try:
            scratchpad["max_iterations"] = int(max_iterations)
        except (TypeError, ValueError):
            scratchpad["max_iterations"] = DEFAULT_MAX_ITERATIONS

    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    return context, scratchpad, runtime_ns, temp


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
        context, scratchpad, _, _ = _ensure_react_vars(run)
        scratchpad["iteration"] = 0

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]

        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        emit("init", {"task": task})
        return StepPlan(node_id="init", next_node="reason")
    
    def reason_node(run: RunState, ctx) -> StepPlan:
        """Call LLM with tools - AbstractCore handles formatting."""
        context, scratchpad, runtime_ns, _ = _ensure_react_vars(run)
        iteration = int(scratchpad.get("iteration", 0) or 0)
        max_iterations = int(scratchpad.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
        if max_iterations < 1:
            max_iterations = 1
        
        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")
        
        scratchpad["iteration"] = iteration + 1
        
        task = str(context.get("task", "") or "")
        messages = context["messages"]
        
        # Check for injected messages (async guidance from user)
        inbox = runtime_ns.get("inbox", [])
        inbox_text = ""
        if inbox:
            inbox_messages = [m.get("content", "") for m in inbox]
            inbox_text = "\n\n[User guidance]: " + " | ".join(inbox_messages)
            runtime_ns["inbox"] = []  # Clear inbox after reading
        
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
                "You have access to the conversation history below as context.\n"
                "Do not claim you have no memory of it; it is provided to you here.\n\n"
                "Continue the conversation and work on the user's latest request.\n\n"
                f"History:\n{history_text}\n\n"
                "Use tools or provide a final answer."
            )
        
        # Append any injected guidance
        if inbox_text:
            prompt += inbox_text
        
        emit(
            "reason",
            {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(inbox_text)},
        )
        
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
                result_key="_temp.llm_response",
            ),
            next_node="parse",
        )
    
    def parse_node(run: RunState, ctx) -> StepPlan:
        """Parse LLM response - check for tool calls or final answer."""
        context, _, _, temp = _ensure_react_vars(run)
        response = temp.get("llm_response", {})
        if not isinstance(response, dict):
            response = {}
        content = response.get("content", "")
        tool_calls = response.get("tool_calls") or []
        
        # Add assistant message to history
        context["messages"].append(_new_message(ctx, role="assistant", content=content))
        
        emit("parse", {
            "has_tool_calls": len(tool_calls) > 0,
            "content_preview": content[:100] if content else "(no content)",
        })
        temp.pop("llm_response", None)
        
        if tool_calls:
            # Store tool calls for execution
            temp["pending_tool_calls"] = tool_calls
            return StepPlan(node_id="parse", next_node="act")
        else:
            # No tool calls - treat content as final answer
            temp["final_answer"] = content
            return StepPlan(node_id="parse", next_node="done")
    
    def act_node(run: RunState, ctx) -> StepPlan:
        """Execute pending tool calls via TOOL_CALLS effect.
        
        This uses the runtime's effect system so tool calls are:
        - Recorded in the ledger
        - Subject to retry/idempotency policies
        - Consistent with the effect-based architecture
        """
        _, _, _, temp = _ensure_react_vars(run)
        tool_calls = temp.get("pending_tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        
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
                temp["pending_tool_calls"] = tool_calls[i + 1:]
                
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
                        result_key="_temp.user_response",
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
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )
    
    def observe_node(run: RunState, ctx) -> StepPlan:
        """Process tool results and add to conversation history."""
        context, _, _, temp = _ensure_react_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}
        results = tool_results.get("results", [])
        messages = context["messages"]
        
        for r in results:
            name = r.get("name", "tool")
            if r.get("success"):
                output = r.get("output", "")
            else:
                output = f"Error: {r.get('error', 'unknown error')}"
            
            emit("observe", {"tool": name, "result": str(output)[:150]})
            
            messages.append(
                _new_message(
                    ctx,
                    role="tool",
                    content=f"[{name}]: {output}",
                    metadata={
                        "name": name,
                        "call_id": r.get("call_id"),
                        "success": bool(r.get("success")),
                    },
                )
            )
        
        temp.pop("tool_results", None)
        temp["pending_tool_calls"] = []
        
        return StepPlan(node_id="observe", next_node="reason")
    
    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        """Handle user response after ask_user."""
        context, _, _, temp = _ensure_react_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = user_response.get("response", "")
        
        emit("user_response", {"response": response_text})
        
        # Add user response to messages
        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)
        
        # Continue with any remaining tool calls or back to reasoning
        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")
    
    def done_node(run: RunState, ctx) -> StepPlan:
        """Complete with final answer."""
        context, scratchpad, _, temp = _ensure_react_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer})
        
        return StepPlan(
            node_id="done",
            complete_output={
                "answer": answer,
                "iterations": int(scratchpad.get("iteration", 0) or 0),
                "messages": list(context.get("messages") or []),
            },
        )
    
    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        """Handle max iterations reached."""
        context, scratchpad, _, _ = _ensure_react_vars(run)
        max_iterations = int(scratchpad.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations})
        
        # Use last content as answer
        messages = list(context.get("messages") or [])
        last_content = messages[-1]["content"] if messages else "Max iterations reached"
        
        return StepPlan(
            node_id="max_iterations",
            complete_output={
                "answer": last_content,
                "iterations": max_iterations,
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

    def __init__(
        self,
        *,
        runtime: Runtime,
        tools: Optional[List[Callable]] = None,
        on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        actor_id: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        super().__init__(runtime=runtime, tools=tools, on_step=on_step, actor_id=actor_id)
        self.max_iterations = int(max_iterations)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
    
    def start(self, task: str) -> str:
        """Start a new agent run with a task."""
        actor_id = self._ensure_actor_id()
        session_id = self._ensure_session_id()
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
                "context": {"task": task, "messages": seeded_messages},
                "scratchpad": {"iteration": 0, "max_iterations": self.max_iterations},
                "_runtime": {
                    "tool_specs": tool_specs,
                    "toolset_id": toolset_id,
                    "inbox": [],
                },
                "_temp": {},
            },
            actor_id=actor_id,
            session_id=session_id,
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
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
        max_iterations=max_iterations,
    )
