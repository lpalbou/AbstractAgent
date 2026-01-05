"""ReAct logic (pure; no runtime imports)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools import ToolCall, ToolDefinition

from .types import LLMRequest


class ReActLogic:
    @staticmethod
    def _format_history_message(message: Dict[str, Any]) -> str:
        role = str(message.get("role", "unknown") or "unknown")
        content = message.get("content", "")
        content_str = "" if content is None else str(content)

        if role != "tool":
            return f"{role}: {content_str}"

        # Tool messages should be presented as *results*, not as tool-call syntax.
        # Avoid bracketed markers like `observation[tool] ...` which can look like
        # parsable tool-call notation to some models.
        meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        name = meta.get("name") if isinstance(meta, dict) else None
        success = meta.get("success") if isinstance(meta, dict) else None

        cleaned = content_str.strip()
        if isinstance(name, str) and name:
            prefix = f"[{name}]:"
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()

        tool_name = str(name).strip() if isinstance(name, str) and name.strip() else "tool"
        if success is True:
            status = "succeeded"
        elif success is False:
            status = "failed"
        else:
            status = "returned"

        return f"Tool {tool_name} {status}: {cleaned}"

    @staticmethod
    def _format_runtime_scratchpad(vars: Optional[Dict[str, Any]]) -> str:
        """Build a ReAct scratchpad from runtime-owned node traces.

        Source of truth: `run.vars["_runtime"]["node_traces"]` (persisted by AbstractRuntime).
        This avoids creating parallel persistence formats in agents/hosts.
        """
        if not isinstance(vars, dict):
            return ""
        runtime_ns = vars.get("_runtime")
        if not isinstance(runtime_ns, dict):
            return ""
        traces = runtime_ns.get("node_traces")
        if not isinstance(traces, dict) or not traces:
            return ""

        all_steps: List[Dict[str, Any]] = []
        for node_trace in traces.values():
            if not isinstance(node_trace, dict):
                continue
            steps = node_trace.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if isinstance(step, dict):
                    all_steps.append(step)

        if not all_steps:
            return ""

        all_steps.sort(key=lambda d: str(d.get("ts") or ""))

        def _render(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                # Keep scratchpad compact: it can contain large tool arguments/results.
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            except Exception:
                return str(value)

        last_thought: Optional[str] = None
        blocks: List[str] = []

        for step in all_steps:
            effect = step.get("effect")
            if not isinstance(effect, dict):
                continue
            etype = str(effect.get("type") or "")
            status = str(step.get("status") or "")

            if etype == "llm_call" and status == "completed":
                result = step.get("result")
                if isinstance(result, dict):
                    # Prefer a provider-supplied reasoning field when available.
                    reasoning = result.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        last_thought = reasoning
                    else:
                        content = result.get("content")
                        if isinstance(content, str) and content.strip():
                            last_thought = content
                continue

            if etype != "tool_calls":
                continue

            ts = str(step.get("ts") or "")
            header = f"[tool_calls]{' ' + ts if ts else ''} ({status or 'unknown'})"
            payload = effect.get("payload")
            payload_dict = dict(payload) if isinstance(payload, dict) else {}
            tool_calls = payload_dict.get("tool_calls")
            action = _render(tool_calls) if tool_calls is not None else ""

            result = step.get("result")
            observation = _render(result)

            parts: List[str] = [header]
            if isinstance(last_thought, str) and last_thought.strip():
                parts.append("Thought:")
                parts.append(last_thought.strip())
            if action.strip():
                parts.append("Action:")
                parts.append(action)
            if observation.strip():
                parts.append("Observation:")
                parts.append(observation)
            blocks.append("\n".join(parts).strip())

        return "\n\n".join(blocks).strip()

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
        """Add tool definitions to this logic instance (deduped by name).

        This enables hosts (e.g. AbstractCode) to dynamically register schema-only tools
        discovered at runtime (e.g. MCP tools) without rebuilding the workflow.
        """
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
        active_memory: str = "",
        system_memory: str = "",
        iteration: int = 1,
        max_iterations: int = 20,
        vars: Optional[Dict[str, Any]] = None,
    ) -> LLMRequest:
        """Build an LLM request for the ReAct agent.

        Args:
            task: The task to perform
            messages: Conversation history
            guidance: Optional guidance text to inject
            active_memory: Deprecated split for "user prompt memory". Active Memory is internal;
                          callers should render it into `system_memory`.
            system_memory: "System prompt memory" (persona/memory organization/tools).
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            vars: Optional run.vars dict. If provided, limits are read from
                  vars["_limits"] (canonical) with fallback to instance defaults.
        """
        task = str(task or "")
        guidance = str(guidance or "").strip()

        # Get limits from vars if available, else use instance defaults
        limits = (vars or {}).get("_limits", {})
        # IMPORTANT: `_limits.max_tokens` is the *context/budget* limit for the run, not
        # an OpenAI `max_tokens` (output cap). Using it as output tokens can generate
        # invalid provider payloads (e.g. max_tokens=262144) and trigger 400 responses.
        #
        # Output tokens are controlled via `_limits.max_output_tokens` (canonical).
        max_tokens = limits.get("max_output_tokens", None)
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        runtime_ns = (vars or {}).get("_runtime", {})
        scratchpad = (vars or {}).get("scratchpad", {})
        plan_mode = bool(runtime_ns.get("plan_mode")) if isinstance(runtime_ns, dict) else False
        plan_text = scratchpad.get("plan") if isinstance(scratchpad, dict) else None
        plan = str(plan_text).strip() if isinstance(plan_text, str) and plan_text.strip() else ""

        # USER ROLE CONTENT MUST CONTAIN ONLY THE USER'S REQUEST.
        #
        # Rationale:
        # - Mixing internal memory/history (especially tool-call-like syntax) into a user-role message
        #   causes some models to treat it as a new user instruction and can trigger loops.
        prompt = task.strip()

        # Keep long-lived agent rules in a separate system prompt for clarity and stability.
        output_budget_line = ""
        if isinstance(max_tokens, int) and max_tokens > 0:
            output_budget_line = (
                f"- Output token limit for this response: {max_tokens}. Keep any single tool-call payload comfortably below this. "
                "If you need to send large tool arguments "
                "(e.g., file contents), split them across multiple tool calls (e.g., `write_file` with mode='w' then mode='a').\n"
            )
        else:
            output_budget_line = (
                "- If you need to send large tool arguments (e.g., file contents), split them across multiple tool calls "
                "(e.g., `write_file` with mode='w' then mode='a').\n"
            )

        system_prompt = (
            "You are an autonomous ReAct agent.\n"
            "Taking action / having an effect means calling a tool.\n\n"
            "Rules:\n"
            "- Be truthful: only claim actions supported by tool outputs (recorded durably by the host/runtime).\n"
            "- Be autonomous: do not ask the user for confirmation to proceed. Keep going until the task is done.\n"
            "- If you want to create/edit files, run commands, fetch URLs, or search, you MUST call the appropriate tool.\n"
            "- If you list next steps, immediately start executing them (with tools) as long as they are within the user's request.\n"
            "- Never fabricate tool outputs. Tool outcomes are captured into internal memory (Active Memory → Key History).\n"
#            "- Do not output lines that look like internal transcript markers (e.g. `observation[tool] ...`). Those are context-only.\n"
            "- Do not quote internal memory verbatim in your answer; use it silently as context.\n"
            "- Only ask the user a question when required information is missing.\n"
            "- Before calling a tool, write 1–3 short lines explaining what you will do and why.\n"
#            "- After tool results, continue from the new information; do not repeat successful tool calls with the same args.\n"
            "- Use Active Memory (Current Tasks/Context/Key History) as your working memory; avoid repeating the same successful action.\n"
            f"{output_budget_line}"
#            "- For file work, prefer file tools (write_file/edit_file) and verify with list_files/read_file.\n"
#            "- If the user asked you to create/update a file, do it with write_file/edit_file (do not ask for permission).\n"
 #           "- Do not prefix your messages with role labels like 'assistant:'.\n"
            "\n"
        )

        system_memory = str(system_memory or "").strip()
        active_memory = str(active_memory or "").strip()

        # Treat *all* Active Memory as internal/system context (never user-role).
        internal_sections: List[str] = []
        if system_memory:
            internal_sections.append(system_memory)
        if active_memory:
            internal_sections.append(active_memory)

        if internal_sections:
            internal_header = (
                "# MY MEMORY\n"
                "The sections below contains the different components of my memory/state and how to interact with them to achieve my goals.\n"
            ).strip()
            system_prompt = f"{internal_header}\n\n" + "\n\n".join(internal_sections).strip() + "\n\n" + system_prompt
            system_prompt = system_prompt.strip()

        # User guidance is host/user-provided policy, not part of the user request body.
        if guidance:
            system_prompt = (system_prompt + "\n\n" + "Guidance:\n" + guidance).strip()

        if plan_mode and plan:
            system_prompt = (system_prompt + "\n\n" + "Current plan:\n" + plan).strip()

        if plan_mode:
            system_prompt += (
                "\nPlan mode:\n"
                "- Maintain and update the plan as you work.\n"
                "- If the plan changes, include a final section at the END of your message:\n"
                "  Plan Update:\n"
                "  <markdown checklist>\n"
                "- Do not stop until the plan is complete.\n"
            )

        return LLMRequest(prompt=prompt, system_prompt=system_prompt, tools=self.tools, max_tokens=max_tokens)

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

        # Some providers return a separate `reasoning` field (e.g. OSS models via OpenAI-compatible APIs).
        # If the cleaned content is empty, fall back to reasoning so the agent's thought is preserved
        # in history/scratchpad and can prevent repetitive tool loops.
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
