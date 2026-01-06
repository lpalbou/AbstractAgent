"""MemAct logic (pure; no runtime imports).

MemAct is a memory-enhanced agent (Letta-like) that relies on a separate, runtime-owned
Active Memory system. This logic layer stays conventional:
- tool calling is the only way to have an effect
- tool results are appended to chat history by the runtime adapter

The memory system is injected by the MemAct runtime adapter via the system prompt and
updated via a structured JSON envelope at finalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest


class MemActLogic:
    def __init__(
        self,
        *,
        tools: List[ToolDefinition],
        max_history_messages: int = -1,
        max_tokens: Optional[int] = None,
    ):
        self._tools = list(tools)
        self._max_history_messages = int(max_history_messages)
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
        """Build a base LLM request (adapter injects memory blocks separately)."""
        _ = messages  # history is carried via chat messages by the adapter

        task = str(task or "").strip()
        guidance = str(guidance or "").strip()

        limits = (vars or {}).get("_limits", {})
        max_output_tokens = limits.get("max_output_tokens", None)
        if max_output_tokens is not None:
            try:
                max_output_tokens = int(max_output_tokens)
            except Exception:
                max_output_tokens = None

        output_budget_line = ""
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            output_budget_line = f"- Output token limit for this response: {max_output_tokens}.\n"

        system_prompt = (
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            "You are an autonomous MemAct agent.\n"
            "Taking action / having an effect means calling a tool.\n\n"
            "Rules:\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            "- Be autonomous: do not ask the user for confirmation to proceed; keep going until the task is done.\n"
            "- If you need to create/edit files, run commands, fetch URLs, or search, you MUST call an appropriate tool.\n"
            "- Never fabricate tool outputs.\n"
            "- Only ask the user a question when required information is missing.\n"
            f"{output_budget_line}"
        ).strip()

        if guidance:
            system_prompt = (system_prompt + "\n\nGuidance:\n" + guidance).strip()

        return LLMRequest(
            prompt=task,
            system_prompt=system_prompt,
            tools=self.tools,
            max_tokens=max_output_tokens,
        )

    def parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        if not isinstance(response, dict):
            return "", []

        content = response.get("content")
        content = "" if content is None else str(content)
        content = content.lstrip()
        for prefix in ("assistant:", "assistant："):
            if content.lower().startswith(prefix):
                content = content[len(prefix) :].lstrip()
                break

        if not content.strip():
            reasoning = response.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()

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

