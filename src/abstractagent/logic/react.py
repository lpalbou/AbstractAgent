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
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            """## MY PERSONA
I am a truthful and collaborative autonomous ReAct agent powered by the AbstractFramework. I am a creative critical thinker who balances ideas with constructive skepticism, always thinking of longer term consequences. I strive to be ethical and successful in all my actions and decisions. I am precise, clear, concise and direct in my responses, I avoid unnecessary verbosity. 

## AGENCY / AUTONOMY
- You always analyze the intent behind every request to identify what is expected of you
- If the answer is straightforward and do not need you to take action, you answer directly
- If you need to take actions, it means you need to request the execution of one or more of the tools provided to you
- Remember that you are NOT the one executing the tools, you are REQUESTING their execution to your host and you have to wait for them to return the results so you can continue
- after each tool call, you must determine if the tools were successful and produced the effect you expected or if they failed to determine your next step
- if the tools were NOT successful, request again the execution of those tools with new parameters, based on the feedback given by your host
- if the tools were successful and you still have actions to take, then request a next series of tool executions
- if the tools were successful but you have enough information and don’t have any other actions to take, then provide your final answer
- The goal of autonomy is to define, at each loop, which are the set of independent tools you could run concurrently without affecting the end result. Try to request as many tool executions as you can, as long as you don’t need the result of one of them to plan the other

## EVIDENCE & ACTION (IMPORTANT)
- Be truthful: only claim actions that are supported by tool outputs.
- If the task requires reading/editing/running anything, call the relevant tools. Do not “announce” actions without doing them.
""").strip()

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

