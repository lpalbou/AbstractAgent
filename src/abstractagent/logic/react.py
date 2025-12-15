"""ReAct logic (pure; no runtime imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest


class ReActLogic:
    def __init__(self, *, tools: List[ToolDefinition], max_history_messages: int = 12):
        self._tools = list(tools)
        self._max_history_messages = int(max_history_messages)
        if self._max_history_messages < 1:
            self._max_history_messages = 1

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
    ) -> LLMRequest:
        task = str(task or "")
        guidance = str(guidance or "").strip()

        if len(messages) <= 1:
            prompt = (
                f"Task: {task}\n\n"
                "Use the available tools to complete this task. When done, provide your final answer."
            )
        else:
            history = messages[-self._max_history_messages :]
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

        return LLMRequest(prompt=prompt, tools=self.tools)

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
        return content, tool_calls

    def format_observation(self, *, name: str, output: str, success: bool) -> str:
        if success:
            return f"[{name}]: {output}"
        return f"[{name}]: Error: {output}"

