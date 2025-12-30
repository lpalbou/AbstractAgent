"""CodeAct logic (pure; no runtime imports).

CodeAct is a ReAct-like loop where the main action is executing Python code
instead of calling many specialized tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)


class CodeActLogic:
    @staticmethod
    def _format_history_message(message: Dict[str, Any]) -> str:
        role = str(message.get("role", "unknown") or "unknown")
        content = message.get("content", "")
        content_str = "" if content is None else str(content)

        if role != "tool":
            return f"{role}: {content_str}"

        meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        name = meta.get("name") if isinstance(meta, dict) else None
        success = meta.get("success") if isinstance(meta, dict) else None

        cleaned = content_str.strip()
        if isinstance(name, str) and name:
            prefix = f"[{name}]:"
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()

        label = "observation"
        if isinstance(name, str) and name:
            label += f"[{name}]"
        if success is True:
            label += " (success)"
        elif success is False:
            label += " (error)"

        return f"{label}: {cleaned}"

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
        """Build an LLM request for the CodeAct agent.

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

        history = messages if messages else []
        history_text = "\n".join([self._format_history_message(m) for m in history])

        prompt = (
            "You are CodeAct: you can solve tasks by writing and executing Python code.\n"
            "Use the tool `execute_python` to run Python snippets. Prefer small, focused scripts.\n"
            "Print any intermediate results you need.\n"
            "Taking action / having an effect means calling `execute_python`. If you want to compute, test, or verify something, you MUST run code via the tool.\n"
            "If you list next steps, immediately start executing them (with `execute_python`) as long as they are within the user's request.\n"
            "Never fabricate tool outputs. Do not output internal transcript markers like `observation[...]`.\n"
            "If the latest History entry is an observation, start by stating what you observed in 1 line.\n"
            "Be autonomous: do not ask the user for confirmation to proceed; keep going until the task is done.\n"
            "Only ask the user a question when required information is missing.\n"
            "When you are confident, provide the final answer without calling tools.\n\n"
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            f"Task: {task}\n\n"
        )
        if history_text:
            prompt += f"History:\n{history_text}\n\n"

        if guidance:
            prompt += f"[User guidance]: {guidance}\n\n"

        if plan_mode and plan:
            prompt += (
                "Plan mode (enabled):\n"
                "- Maintain and update the plan as you work (mark steps done, add/remove steps if needed).\n"
                "- If the plan changes, include a final section at the END of your message:\n"
                "  Plan Update:\n"
                "  <markdown checklist>\n"
                "- Do not stop until the plan is complete.\n\n"
                f"Current plan:\n{plan}\n\n"
            )

        prompt += (
            "If you need to run code, either:\n"
            "- Call `execute_python` with the Python code, or\n"
            "- If tool calling is unavailable, include a fenced ```python code block.\n"
        )

        return LLMRequest(prompt=prompt, tools=self.tools, max_tokens=max_tokens)

    def parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        if not isinstance(response, dict):
            return "", []

        content = response.get("content")
        content = "" if content is None else str(content)

        # Some providers return a separate `reasoning` field. If content is empty,
        # preserve reasoning as the assistant message so iterative loops don't lose context.
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

    def extract_code(self, text: str) -> str | None:
        text = str(text or "")
        m = _CODE_BLOCK_RE.search(text)
        if not m:
            return None
        code = m.group(1).strip("\n")
        return code.strip() or None

    def format_observation(self, *, name: str, output: Any, success: bool) -> str:
        if name != "execute_python":
            out = "" if output is None else str(output)
            return f"[{name}]: {out}" if success else f"[{name}]: Error: {out}"

        if not isinstance(output, dict):
            out = "" if output is None else str(output)
            return f"[execute_python]: {out}" if success else f"[execute_python]: Error: {out}"

        stdout = str(output.get("stdout") or "")
        stderr = str(output.get("stderr") or "")
        exit_code = output.get("exit_code")
        error = output.get("error")

        parts: List[str] = []
        if error:
            parts.append(f"error={error}")
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        if stdout:
            parts.append("stdout:\n" + stdout)
        if stderr:
            parts.append("stderr:\n" + stderr)

        rendered = "\n".join(parts).strip() or "(no output)"
        return f"[execute_python]: {rendered}"
