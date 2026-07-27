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
        max_history_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        # max_history_messages/max_tokens are DEPRECATED-IGNORED (fable5 B-F9
        # 2026-07-13): they were always accepted-and-ignored here (the dead-knob
        # class) — window policy lives in _limits, output caps in
        # _limits.max_output_tokens. A hard TypeError removal broke shipped
        # abstractcode call sites (regression adversary P0, same day), so the
        # one-release shim warns loudly on MEANINGFUL values and ignores the
        # legacy no-op spellings (None / -1). Removal lands next release.
        if max_history_messages is not None and max_history_messages != -1:
            import warnings

            warnings.warn(
                "max_history_messages on the Logic constructor was never applied and is "
                "deprecated-ignored; window policy lives in _limits (removal next release)",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_tokens is not None:
            import warnings

            warnings.warn(
                "max_tokens on the Logic constructor was never applied and is "
                "deprecated-ignored; use the agent facades' max_output_tokens (removal next release)",
                DeprecationWarning,
                stacklevel=2,
            )
        self._tools = list(tools)

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

        # Cache stability (0212 propagated 2026-07-13): the iteration counter
        # is NOT rendered here — it mutated the system-prompt head every cycle
        # and busted the provider prefix cache; the adapter carries loop
        # position in a trailing volatile message instead.
        _ = iteration
        system_prompt = (
            "You are an autonomous MemAct agent.\n"
            "Taking action / having an effect means calling a tool.\n\n"
            "Rules:\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            "- Be autonomous: do not ask the user for confirmation to proceed; keep going until the task is done.\n"
            "- If you need to create/edit files, run commands, fetch URLs, or search, you MUST call an appropriate tool.\n"
            "- Efficiency: batch independent read-only tool calls into a single turn (multiple tool calls) when possible.\n"
            "  Side-effectful tools: never batch two calls that touch the SAME target (e.g. two edits to one file); calls on DIFFERENT, independent targets may ride one turn — the runtime executes a batch in order. Never batch execute_command, comms sends, or any mcp:: tool.\n"
            "  Only split tool calls across turns when later calls depend on earlier outputs.\n"
            "  For MULTIPLE edits to ONE file in a turn, prefer ONE edit_file diff call — many hunks apply atomically. One call is not a batch.\n"
            "- When context is getting large, use delegate_agent(task, context, tools) to offload an independent subtask with minimal context.\n"
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
