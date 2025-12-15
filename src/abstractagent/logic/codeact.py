"""CodeAct logic (pure; no runtime imports).

CodeAct is a ReAct-like loop where the main action is executing Python code
instead of calling many specialized tools.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\\s*\\n(.*?)\\n```", re.IGNORECASE | re.DOTALL)


class CodeActLogic:
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

        history = messages[-self._max_history_messages :] if messages else []
        history_text = "\n".join(
            [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history]
        )

        prompt = (
            "You are CodeAct: you can solve tasks by writing and executing Python code.\n"
            "Use the tool `execute_python` to run Python snippets. Prefer small, focused scripts.\n"
            "Print any intermediate results you need.\n"
            "When you are confident, provide the final answer without calling tools.\n\n"
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            f"Task: {task}\n\n"
        )
        if history_text:
            prompt += f"History:\n{history_text}\n\n"

        if guidance:
            prompt += f"[User guidance]: {guidance}\n\n"

        prompt += (
            "If you need to run code, either:\n"
            "- Call `execute_python` with the Python code, or\n"
            "- If tool calling is unavailable, include a fenced ```python code block.\n"
        )

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

