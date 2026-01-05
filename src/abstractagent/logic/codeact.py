"""CodeAct logic (pure; no runtime imports).

This module implements a conventional CodeAct loop:
- the model primarily acts by producing Python code (or calling execute_python)
- tool results are appended to chat history
- the model iterates until it can answer directly

CodeAct is intentionally *not* a memory-enhanced agent.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)


class CodeActLogic:
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

    def add_tools(self, tools: List[ToolDefinition]) -> int:
        if not isinstance(tools, list) or not tools:
            return 0

        existing = {str(t.name) for t in self._tools if getattr(t, "name", None)}
        added = 0
        for t in tools:
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            if name in existing:
                continue
            self._tools.append(t)
            existing.add(name)
            added += 1
        return added

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
        _ = messages  # history is carried out-of-band via chat messages

        task = str(task or "")
        guidance = str(guidance or "").strip()

        limits = (vars or {}).get("_limits", {})
        max_output_tokens = limits.get("max_output_tokens", None)
        if max_output_tokens is not None:
            try:
                max_output_tokens = int(max_output_tokens)
            except Exception:
                max_output_tokens = None

        runtime_ns = (vars or {}).get("_runtime", {})
        scratchpad = (vars or {}).get("scratchpad", {})
        plan_mode = bool(runtime_ns.get("plan_mode")) if isinstance(runtime_ns, dict) else False
        plan_text = scratchpad.get("plan") if isinstance(scratchpad, dict) else None
        plan = str(plan_text).strip() if isinstance(plan_text, str) and plan_text.strip() else ""

        prompt = task.strip()

        output_budget_line = ""
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            output_budget_line = f"- Output token limit for this response: {max_output_tokens}.\n"

        system_prompt = (
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            "You are CodeAct: you solve tasks by writing and executing Python when needed.\n\n"
            "Evidence & action (IMPORTANT):\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            "- If the task requires code execution or file edits, do it now (call a tool or output a fenced ```python``` block).\n"
            "- Do not “announce” actions without executing them.\n\n"
            "Rules:\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            "- Be autonomous: do not ask the user for confirmation to proceed; keep going until the task is done.\n"
            "- If you need to run code, call `execute_python` (preferred) or output a fenced ```python code block.\n"
            "- Never fabricate tool outputs.\n"
            "- Only ask the user a question when required information is missing.\n"
            f"{output_budget_line}"
        ).strip()

        if guidance:
            system_prompt = (system_prompt + "\n\nGuidance:\n" + guidance).strip()

        if plan_mode and plan:
            system_prompt = (system_prompt + "\n\nCurrent plan:\n" + plan).strip()

        if plan_mode:
            system_prompt = (
                system_prompt
                + "\n\nPlan mode:\n"
                "- Maintain and update the plan as you work.\n"
                "- If the plan changes, include a final section at the END of your message:\n"
                "  Plan Update:\n"
                "  <markdown checklist>\n"
            ).strip()

        return LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            tools=self.tools,
            max_tokens=max_output_tokens,
        )

    def parse_response(self, response: Any) -> Tuple[str, List[ToolCall]]:
        if not isinstance(response, dict):
            return "", []

        content = response.get("content")
        content = "" if content is None else str(content)

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
        out = "" if output is None else str(output)
        if success:
            return f"[{name}]: {out}"
        return f"[{name}]: Error: {out}"

