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
        max_tokens = limits.get("max_tokens", self._max_tokens)
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        runtime_ns = (vars or {}).get("_runtime", {})
        scratchpad = (vars or {}).get("scratchpad", {})
        plan_mode = bool(runtime_ns.get("plan_mode")) if isinstance(runtime_ns, dict) else False
        plan_text = scratchpad.get("plan") if isinstance(scratchpad, dict) else None
        plan = str(plan_text).strip() if isinstance(plan_text, str) and plan_text.strip() else ""

        history_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
        if not history_text:
            prompt = (
                f"Task: {task}\n\n"
                "Use the available tools to complete this task. When done, provide your final answer."
            )
        else:
            prompt = (
                "You have access to the conversation history below as context.\n"
                "Do not claim you have no memory of it; it is provided to you here.\n\n"
                f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
                f"History:\n{history_text}\n\n"
                "Continue the conversation and work on the user's latest request.\n"
                "Use tools when needed, or provide a final answer."
            )

        prompt += (
            "\n\nRules:\n"
            "- Be truthful: only claim actions that are supported by tool outputs in History.\n"
            "- Be autonomous: do not ask the user for confirmation to proceed. Keep going until the task is done.\n"
            "- Only ask the user a question when required information is missing.\n"
            "- Before calling a tool, write 1–3 short lines explaining what you will do and why.\n"
            "- After tool results, continue from the new information; do not repeat successful tool calls with the same args.\n"
            "- For file work, prefer file tools (write_file/edit_file) and verify with list_files/read_file.\n"
            "- Do not prefix your messages with role labels like 'assistant:'.\n"
        )

        if plan_mode and plan:
            prompt += (
                "\n\nPlan mode (enabled):\n"
                "- Maintain and update the plan as you work (mark steps done, add/remove steps if needed).\n"
                "- If the plan changes, include a final section at the END of your message:\n"
                "  Plan Update:\n"
                "  <markdown checklist>\n"
                "- Do not stop until the plan is complete.\n\n"
                f"Current plan:\n{plan}\n"
            )

        if guidance:
            prompt += "\n\n[User guidance]: " + guidance

        return LLMRequest(prompt=prompt, tools=self.tools, max_tokens=max_tokens)

    def parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        if not isinstance(response, dict):
            return "", []

        content = response.get("content")
        content = "" if content is None else str(content)
        # Some OSS models echo role labels; strip common prefixes to keep UI/history clean.
        content = content.lstrip()
        for prefix in ("assistant:", "assistant："):
            if content.lower().startswith(prefix):
                content = content[len(prefix) :].lstrip()
                break

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
            from abstractcore.tools.parser import parse_tool_calls, detect_tool_calls, clean_tool_syntax
            if detect_tool_calls(content):
                # Pass model name for architecture-specific parsing
                model_name = response.get("model")
                tool_calls = parse_tool_calls(content, model_name=model_name)
                # Clean tool call syntax from the assistant content so:
                # - UI shows the human-readable "why" (if any)
                # - History doesn't get polluted with tool tags that can cause repeats
                if tool_calls:
                    content = clean_tool_syntax(content, tool_calls)
        elif tool_calls and content:
            # Even when a provider returns native tool call fields, some OSS models also
            # embed tool-call syntax in `content`. Clean it to avoid polluting history/UI.
            from abstractcore.tools.parser import clean_tool_syntax
            content = clean_tool_syntax(content, tool_calls)

        return content, tool_calls

    def format_observation(self, *, name: str, output: str, success: bool) -> str:
        if success:
            return f"[{name}]: {output}"
        return f"[{name}]: Error: {output}"
