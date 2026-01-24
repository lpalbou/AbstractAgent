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
        # History is carried out-of-band via `messages`; keep logic pure.
        _ = messages

        task = str(task or "").strip()
        guidance = str(guidance or "").strip()

        # Output token cap (provider max_tokens) comes from `_limits.max_output_tokens`.
        limits = (vars or {}).get("_limits", {})
        max_output_tokens = limits.get("max_output_tokens", None)
        if max_output_tokens is not None:
            try:
                max_output_tokens = int(max_output_tokens)
            except Exception:
                max_output_tokens = None
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            max_output_tokens = None

        system_prompt = (
            f"Iteration: {int(iteration)}/{int(max_iterations)}\n\n"
            "## MY PERSONA\n"
            "You are an autonomous ReAct agent (Reason → Act → Observe).\n\n"
            "Loop contract:\n"
            "- THINK briefly using the full transcript and prior observations.\n"
            "- If you need to ACT, CALL one or more tools (function calls).\n"
            "- If you are DONE, respond with the final answer and NO tool calls.\n\n"
            "Rules:\n"
            "- Choose tools yourself; never ask the user which tool to run.\n"
            "- Do not write a long plan before tool calls.\n"
            "- Keep non-final responses short; do not draft large deliverables in chat when tools can build them.\n"
            "- Efficiency (important): the runtime supports MULTIPLE tool calls in one response.\n"
            "  Batch independent read-only tool calls to reduce iterations.\n"
            "  Example: read multiple files/ranges or run multiple searches in one response.\n"
            "  If reading nearby ranges of the same file, prefer ONE call with a wider range.\n"
            "  Only split tool calls across turns when later calls depend on earlier outputs; do NOT batch side-effectful tools (write_file/edit_file/execute_command).\n"
            "- When context is getting large, use delegate_agent(task, context, tools) to offload an independent subtask with minimal context.\n"
            "- Keep tool call arguments small and valid; avoid embedding huge blobs (large file contents / giant JSON) directly in arguments.\n"
            "- Attachments:\n"
            "  - If you see an 'Active attachments' message or inline 'Content from <file>' blocks, treat those attachments as already available in-context.\n"
            "    Do NOT call tools just to re-open/read them.\n"
            "  - If you see 'Stored session attachments', those may not be included in the current call.\n"
            "    Only if you truly need it, use the attachment-open tool with artifact_id and a bounded line range.\n"
            "  - Never use filesystem tools on attachment filenames/paths or absolute paths outside the workspace.\n"
            "- For fetch_url: use include_full_content=False for shorter previews; set keep_links=False to strip links when not needed.\n"
            "- For large files, create a small skeleton first, then refine via multiple smaller edits/tool calls.\n"
            "- Use tool outputs as evidence; do not claim actions without tool outputs.\n"
            "- Continue iterating until the task is complete.\n"
        ).strip()

        if guidance:
            system_prompt = f"{system_prompt}\n\nGuidance:\n{guidance}".strip()

        # Note: prompt is unused by the runtime adapter (we supply chat `messages`).
        return LLMRequest(prompt=task, system_prompt=system_prompt, tools=self.tools, max_tokens=max_output_tokens)

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
