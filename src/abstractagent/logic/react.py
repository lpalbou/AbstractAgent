"""ReAct logic (pure; no runtime imports).

This module implements the classic ReAct loop:
- the model decides whether to call tools
- tool results are appended to chat history
- the model iterates until it can answer directly

ReAct is intentionally *not* a memory-enhanced agent. Long-term memory and
structured memory blocks belong in a separate agent (MemAct).
"""

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

    def add_tools(self, tools: List[ToolDefinition]) -> int:
        """Add tool definitions to this logic instance (deduped by name)."""
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
        """Build an LLM request for the ReAct agent.

        Notes:
        - The user request belongs in the user-role message (prompt), not in the system prompt.
        - Conversation + tool history is provided via `messages` by the runtime adapter.
        """
        _ = messages  # history is carried out-of-band via chat messages

        task = str(task or "")
        guidance = str(guidance or "").strip()

        # Output token cap (provider max_tokens) comes from `_limits.max_output_tokens`.
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
            output_budget_line = (
                f"- Output token limit for this response: {max_output_tokens}.\n"
            )

        system_prompt = (
            f"CYCLE: {int(iteration)}/{int(max_iterations)}\n\n"
            """## MY PERSONA
I am a truthful and highly autonomous ReAct agent powered by the AbstractFramework. I am a creative critical thinker who balances ideas with constructive skepticism and always think of long term consequences of my actions. I strive to be ethical and successful in all my decisions and undertakings. I am precise, clear, concise and provide direct responses avoiding unnecessary verbosity.

## AGENCY / AUTONOMY
- I start by analyzing the intent behind the user request to identify and further clarify the EXPECTED OUTCOMES
- I build a plan of actions to achieve the desired outcome
- DURING each CYCLE:
  - THINK : I evaluate the current state of the conversation, in particular the previous tool executions, and I list the next best action(s) that can be carried out by the available tools
  - ACT : request the execution of the tools(s) you selected in the THINK phase
  - OBSERVE : discuss the results of the tool executions and give a feedback on if I achieved the desired outcome. if not, make a recommendation
- This CYCLE is repeated until I achieve ALL the EXPECTED OUTCOMES and there is no more tools to call
- My PRIMARY GOAL is to achieve the EXPECTED OUTCOMES by calling iteratively the appropriate set of tools, make discoveries and adjust my next steps

CRITICAL: 
- I can not execute a tool by myself, I can only request the execution of tools to my host and then observe the results of those executions to adjust my plan
- I must continue iterating new CYCLES until I achieve ALL the EXPECTED OUTCOMES and there is no more tools to call""").strip()

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
        # Some OSS models echo role labels; strip common prefixes to keep UI/history clean.
        content = content.lstrip()
        for prefix in ("assistant:", "assistant："):
            if content.lower().startswith(prefix):
                content = content[len(prefix) :].lstrip()
                break

        # Some providers return a separate `reasoning` field. If content is empty, fall back
        # to reasoning so iterative loops don't lose context.
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
