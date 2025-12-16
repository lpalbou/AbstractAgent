"""ReAct logic (pure; no runtime imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest


class ReActLogic:
    def __init__(
        self,
        *,
        tools: List[ToolDefinition],
        max_history_messages: int = -1,
        max_tokens: Optional[int] = None,
    ):
        self._tools = list(tools)
        self._max_history_messages = int(max_history_messages)
        # -1 means unlimited (send all messages), otherwise must be >= 1
        if self._max_history_messages != -1 and self._max_history_messages < 1:
            self._max_history_messages = 1
        self._max_tokens = max_tokens

    @property
    def tools(self) -> List[ToolDefinition]:
        return list(self._tools)

    def build_request(
        self,
        *,
        task: str,
        messages: List[Dict[str, Any]],
        guidance: str = "",
        iteration: int = 1,
        max_iterations: int = 20,
        vars: Optional[Dict[str, Any]] = None,
    ) -> LLMRequest:
        """Build an LLM request for the ReAct agent.

        Args:
            task: The task to perform
            messages: Conversation history
            guidance: Optional guidance text to inject
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            vars: Optional run.vars dict. If provided, limits are read from
                  vars["_limits"] (canonical) with fallback to instance defaults.
        """
        task = str(task or "")
        guidance = str(guidance or "").strip()

        # Get limits from vars if available, else use instance defaults
        limits = (vars or {}).get("_limits", {})
        max_history = int(limits.get("max_history_messages", self._max_history_messages) or self._max_history_messages)
        max_tokens = limits.get("max_tokens", self._max_tokens)
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        if len(messages) <= 1:
            prompt = (
                f"Task: {task}\n\n"
                "Use the available tools to complete this task. When done, provide your final answer."
            )
        else:
            # -1 means unlimited (use all messages)
            if max_history == -1:
                history = messages
            else:
                history = messages[-max_history:]
            history_text = "\n".join(
                [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history]
            )
            prompt = (
                "You have access to the conversation history below as context.\n"
                "Do not claim you have no memory of it; it is provided to you here.\n\n"
                f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
                f"History:\n{history_text}\n\n"
                "Continue the conversation and work on the user's latest request.\n"
                "Use tools when needed, or provide a final answer."
            )

        if guidance:
            prompt += "\n\n[User guidance]: " + guidance

        return LLMRequest(prompt=prompt, tools=self.tools, max_tokens=max_tokens)

    def parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        if not isinstance(response, dict):
            return "", []

        content = response.get("content")
        content = "" if content is None else str(content)

        tool_calls_raw = response.get("tool_calls") or []
        tool_calls: List[ToolCall] = []
        if isinstance(tool_calls_raw, list):
            for tc in tool_calls_raw:
                if isinstance(tc, ToolCall):
                    tool_calls.append(tc)
                    continue
                if isinstance(tc, dict):
                    name = str(tc.get("name", "") or "")
                    args = tc.get("arguments", {})
                    call_id = tc.get("call_id")
                    if isinstance(args, dict):
                        tool_calls.append(ToolCall(name=name, arguments=dict(args), call_id=call_id))

        # FALLBACK: Parse from content if no native tool calls
        # Handles <|tool_call|>, <function_call>, ```tool_code, etc.
        if not tool_calls and content:
            from abstractcore.tools.parser import parse_tool_calls, detect_tool_calls
            if detect_tool_calls(content):
                # Pass model name for architecture-specific parsing
                model_name = response.get("model")
                tool_calls = parse_tool_calls(content, model_name=model_name)

        return content, tool_calls

    def format_observation(self, *, name: str, output: str, success: bool) -> str:
        if success:
            return f"[{name}]: {output}"
        return f"[{name}]: Error: {output}"

