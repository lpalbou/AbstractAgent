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

    @staticmethod
    def fenced_fallback_enabled(runtime_ns: Any) -> bool:
        """A1 capability-conditional default (0014 scoped piece, operator
        green light 2026-07-14):         explicit `_runtime.codeact_fenced_fallback`
        wins both ways (bool or the _flag string spellings; an unrecognized
        string falls through to the capability default); otherwise the
        default is NOT `supports_native_tools` (the bit runtime seeds per run
        from model capabilities). Rationale, both directions measured: a
        prompt that teaches the fence to a NATIVE-tools model competes with a
        trained channel and manufactures dead prose replies (the Mnemosyne
        zero-tool class, 2026-07-11); a PROMPTED model without the fence has
        no action channel at all. Absent capability bit -> True (fail-safe:
        extraction on a native model is harmless; a prompted model without
        the fence is armless). ONE resolver — the adapter and the prompt line
        must never disagree (teaching a disabled channel manufactures dead
        replies; extracting an untaught one executes illustrations)."""
        if isinstance(runtime_ns, dict):
            explicit = runtime_ns.get("codeact_fenced_fallback")
            if isinstance(explicit, bool):
                return explicit
            if isinstance(explicit, (int, float)):
                return bool(explicit)
            if isinstance(explicit, str):
                lowered = explicit.strip().lower()
                if lowered in ("1", "true", "yes", "on", "enabled"):
                    return True
                if lowered in ("0", "false", "no", "off", "disabled"):
                    return False
            # Capability signal: both keys are written by the same seeder
            # (runtime start / facade) — accept either (wave-F P3: a host
            # seeding only the descriptive `tool_support: "native"` string
            # got fence ON on a native model, the competing-channel class).
            native = runtime_ns.get("supports_native_tools") is True or (
                str(runtime_ns.get("tool_support") or "").strip().lower() == "native"
            )
            if native:
                # Routed-model distrust (wave-F P1): the seeded bit describes
                # the model it was DERIVED from, not necessarily the model the
                # run is routed to (`_runtime.model` per-run routing). When the
                # seeder stamped `tool_support_model` and it differs from the
                # effective model, the bit is stale — fail toward fence ON:
                # extraction on a native model degrades (competing channel,
                # loud in transcripts); a prompted model without the fence
                # completes SILENTLY with code-as-prose as the "answer".
                seeded_model = str(runtime_ns.get("tool_support_model") or "").strip()
                effective_model = str(runtime_ns.get("model") or "").strip()
                if seeded_model and effective_model and seeded_model.lower() != effective_model.lower():
                    return True
                return False
        return True

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

        # NOTE (cache stability, 0212 propagated 2026-07-13): the iteration
        # counter and the live plan text are deliberately NOT in this system
        # prompt — they mutate per cycle and busted the provider prefix cache
        # on every call (the head of the prompt changed at byte ~12). The
        # adapter carries loop position (and the plan in plan_mode) in a
        # trailing volatile message instead, mirroring ReAct.
        _ = iteration  # retained in the signature for adapters/tests; not rendered here

        # A1 (2026-07-14): the fence is taught ONLY when the fenced fallback
        # is enabled — a prompt teaching a channel the parser has disabled
        # manufactures dead replies, and teaching it to a native-tools model
        # competes with the trained tool_calls channel (the Mnemosyne
        # zero-tool class, 2026-07-11). The capability bit is per-run
        # constant, so the prompt stays byte-stable within a run.
        fenced = self.fenced_fallback_enabled(runtime_ns)
        if fenced:
            action_line = "- If the task requires code execution or file edits, do it now (call a tool or output a fenced ```python``` block).\n"
            run_code_line = "- If you need to run code, call `execute_python` (preferred) or output a fenced ```python code block.\n"
        else:
            action_line = "- If the task requires code execution or file edits, do it now (call a tool).\n"
            run_code_line = "- If you need to run code, call `execute_python`.\n"
        system_prompt = (
            "You are CodeAct: you solve tasks by writing and executing Python when needed.\n\n"
            "Evidence & action (IMPORTANT):\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            f"{action_line}"
            "- Do not “announce” actions without executing them.\n\n"
            "Rules:\n"
            "- Be truthful: only claim actions supported by tool outputs.\n"
            "- Be autonomous: do not ask the user for confirmation to proceed; keep going until the task is done.\n"
            f"{run_code_line}"
            "- Efficiency: batch independent read-only tool calls into a single turn (multiple tool calls) to reduce iterations.\n"
            "  Examples: read_file for multiple files/ranges, search_files with different queries, list_files across folders, analyze_code on multiple targets.\n"
            "  If reading nearby ranges of the same file, prefer ONE call with a wider range.\n"
            "  Only split tool calls across turns when later calls depend on earlier outputs.\n"
            "  Side-effectful tools: never batch two calls that touch the SAME target (e.g. two edits to one file); calls on DIFFERENT, independent targets may ride one turn — the runtime executes a batch in order. Never batch execute_python, execute_command, comms sends, or any mcp:: tool.\n"
            "  For MULTIPLE edits to ONE file in a turn, prefer ONE edit_file diff call — many hunks apply atomically. One call is not a batch.\n"
            "- When context is getting large, use delegate_agent(task, context, tools) to offload an independent subtask with minimal context.\n"
            "- Never fabricate tool outputs.\n"
            "- Only ask the user a question when required information is missing.\n"
            f"{output_budget_line}"
        ).strip()

        if guidance:
            system_prompt = (system_prompt + "\n\nGuidance:\n" + guidance).strip()

        # Plan MODE instructions are byte-stable (safe in the prefix); the live
        # plan TEXT is not — it changes on every plan update and rides the
        # adapter's volatile tail instead (cache stability, see `plan` unused
        # below when plan_mode is on).
        _ = plan
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
