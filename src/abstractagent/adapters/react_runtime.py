"""AbstractRuntime adapter for ReAct-like agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy

from ..logic.react import ReActLogic

def _new_message(
    ctx: Any,
    *,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    timestamp: Optional[str] = None
    now_iso = getattr(ctx, "now_iso", None)
    if callable(now_iso):
        timestamp = str(now_iso())
    if not timestamp:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()

    import uuid

    meta = dict(metadata or {})
    meta.setdefault("message_id", f"msg_{uuid.uuid4().hex}")

    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "metadata": meta,
    }


def ensure_react_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place.

    Returns:
        Tuple of (context, scratchpad, runtime_ns, temp, limits) dicts.
    """
    ensure_namespaces(run.vars)
    limits = ensure_limits(run.vars)
    context = run.vars["context"]
    scratchpad = run.vars["scratchpad"]
    runtime_ns = run.vars["_runtime"]
    temp = run.vars["_temp"]

    if "task" in run.vars and "task" not in context:
        context["task"] = run.vars.pop("task")
    if "messages" in run.vars and "messages" not in context:
        context["messages"] = run.vars.pop("messages")
    if "iteration" in run.vars and "iteration" not in scratchpad:
        scratchpad["iteration"] = run.vars.pop("iteration")
    if "max_iterations" in run.vars and "max_iterations" not in scratchpad:
        scratchpad["max_iterations"] = run.vars.pop("max_iterations")
    if "_inbox" in run.vars and "inbox" not in runtime_ns:
        runtime_ns["inbox"] = run.vars.pop("_inbox")

    for key in ("llm_response", "tool_results", "pending_tool_calls", "user_response", "final_answer"):
        if key in run.vars and key not in temp:
            temp[key] = run.vars.pop(key)

    if not isinstance(context.get("messages"), list):
        context["messages"] = []
    if not isinstance(runtime_ns.get("inbox"), list):
        runtime_ns["inbox"] = []

    iteration = scratchpad.get("iteration")
    if not isinstance(iteration, int):
        try:
            scratchpad["iteration"] = int(iteration or 0)
        except (TypeError, ValueError):
            scratchpad["iteration"] = 0

    max_iterations = scratchpad.get("max_iterations")
    if max_iterations is None:
        scratchpad["max_iterations"] = 25
    elif not isinstance(max_iterations, int):
        try:
            scratchpad["max_iterations"] = int(max_iterations)
        except (TypeError, ValueError):
            scratchpad["max_iterations"] = 25

    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    # Track whether any external tools were actually executed during this run.
    # This is used to reliably trigger a final "synthesis" pass so the agent
    # returns a user-facing answer instead of echoing tool observations.
    used_tools = scratchpad.get("used_tools")
    if not isinstance(used_tools, bool):
        scratchpad["used_tools"] = bool(used_tools) if used_tools is not None else False

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_react_workflow(
    *,
    logic: ReActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    workflow_id: str = "react_agent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> WorkflowSpec:
    """Adapt ReActLogic to an AbstractRuntime workflow."""

    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)

    def _current_tool_defs() -> list[Any]:
        """Return the current tool definitions from the logic (dynamic)."""
        defs = getattr(logic, "tools", None)
        if not isinstance(defs, list):
            try:
                defs = list(defs)  # type: ignore[arg-type]
            except Exception:
                defs = []
        return [t for t in defs if getattr(t, "name", None)]

    def _tool_by_name() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in _current_tool_defs():
            name = getattr(t, "name", None)
            if isinstance(name, str) and name.strip():
                out[name] = t
        return out

    def _default_allowlist() -> list[str]:
        if isinstance(allowed_tools, list):
            allow = [str(t).strip() for t in allowed_tools if isinstance(t, str) and t.strip()]
            return allow if allow else []
        # Default allowlist: all tools currently known to the logic (deduped, order preserved).
        out: list[str] = []
        seen: set[str] = set()
        for t in _current_tool_defs():
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name.strip() or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _normalize_allowlist(raw: Any) -> list[str]:
        items: list[Any]
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, tuple):
            items = list(raw)
        elif isinstance(raw, str):
            items = [raw]
        else:
            items = []

        out: list[str] = []
        seen: set[str] = set()
        current = _tool_by_name()
        for t in items:
            if not isinstance(t, str):
                continue
            name = t.strip()
            if not name:
                continue
            if name in seen:
                continue
            # Only accept tool names known to the workflow's logic (dynamic).
            if name not in current:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _effective_allowlist(runtime_ns: Dict[str, Any]) -> list[str]:
        # Allow runtime vars to override tool selection (Visual Agent tools pin).
        if isinstance(runtime_ns, dict) and "allowed_tools" in runtime_ns:
            normalized = _normalize_allowlist(runtime_ns.get("allowed_tools"))
            runtime_ns["allowed_tools"] = normalized
            return normalized
        return _normalize_allowlist(list(_default_allowlist()))

    def _allowed_tool_defs(allow: list[str]) -> list[Any]:
        out: list[Any] = []
        current = _tool_by_name()
        for name in allow:
            tool = current.get(name)
            if tool is not None:
                out.append(tool)
        return out

    def _system_prompt(runtime_ns: Dict[str, Any]) -> Optional[str]:
        raw = runtime_ns.get("system_prompt") if isinstance(runtime_ns, dict) else None
        if isinstance(raw, str) and raw.strip():
            return raw
        return None

    def _sanitize_llm_messages(messages: Any, *, limits: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Convert runtime-owned message dicts into OpenAI-style {role, content, ...}.

        Runtime messages can include extra metadata fields (`timestamp`, `metadata`) that many providers
        will reject. Keep only the fields the LLM API expects.
        """
        if not isinstance(messages, list) or not messages:
            return []
        # Keep the LLM-visible context bounded even if the durable history contains large
        # tool outputs or code dumps.
        def _limit_int(key: str, default: int) -> int:
            if not isinstance(limits, dict):
                return default
            try:
                return int(limits.get(key, default))
            except Exception:
                return default
        max_message_chars = _limit_int("max_message_chars", -1)
        max_tool_message_chars = _limit_int("max_tool_message_chars", -1)

        def _truncate(text: str, *, max_chars: int) -> str:
            if max_chars <= 0:
                return text
            if len(text) <= max_chars:
                return text
            suffix = f"\n… (truncated, {len(text):,} chars total)"
            keep = max_chars - len(suffix)
            if keep < 200:
                keep = max_chars
                suffix = ""
            return text[:keep].rstrip() + suffix

        out: List[Dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            content = m.get("content")
            if not role or content is None:
                continue
            content_str = str(content)
            if not content_str.strip():
                continue
            limit = max_tool_message_chars if role == "tool" else max_message_chars
            entry: Dict[str, str] = {"role": role, "content": _truncate(content_str, max_chars=limit)}
            if role == "tool":
                meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
                call_id = meta.get("call_id") if isinstance(meta, dict) else None
                if call_id is not None and str(call_id).strip():
                    # OpenAI-compatible servers accept `tool_call_id` for tool messages.
                    entry["tool_call_id"] = str(call_id).strip()
            out.append(entry)
        return out

    def _flag(runtime_ns: Dict[str, Any], key: str, *, default: bool = False) -> bool:
        if not isinstance(runtime_ns, dict) or key not in runtime_ns:
            return bool(default)
        val = runtime_ns.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            lowered = val.strip().lower()
            if lowered in ("1", "true", "yes", "on", "enabled"):
                return True
            if lowered in ("0", "false", "no", "off", "disabled"):
                return False
        return bool(default)

    def _int(runtime_ns: Dict[str, Any], key: str, *, default: int) -> int:
        if not isinstance(runtime_ns, dict) or key not in runtime_ns:
            return int(default)
        val = runtime_ns.get(key)
        try:
            return int(val)  # type: ignore[arg-type]
        except Exception:
            return int(default)

    def _extract_plan_update(content: str) -> Optional[str]:
        """Extract a plan update block from model content (best-effort).

        Convention (prompted in Plan mode): the model appends a final section:

            Plan Update:
            - [ ] ...
            - [x] ...
        """
        if not isinstance(content, str) or not content.strip():
            return None
        import re

        lines = content.splitlines()
        header_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if re.match(r"(?i)^\s*plan\s*update\s*:\s*$", line.strip()):
                header_idx = i
        if header_idx is None:
            return None
        plan_lines = lines[header_idx + 1 :]
        while plan_lines and not plan_lines[0].strip():
            plan_lines.pop(0)
        plan_text = "\n".join(plan_lines).strip()
        if not plan_text:
            return None
        # Require at least one bullet/numbered line to avoid accidental captures.
        if not re.search(r"(?m)^\s*(?:[-*]|\d+\.)\s+", plan_text):
            return None
        return plan_text

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)
        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]

        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        # Ensure toolset metadata is present for audit/debug.
        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)
        runtime_ns.setdefault("inbox", [])

        emit("init", {"task": task})
        if _flag(runtime_ns, "plan_mode", default=False) and not isinstance(scratchpad.get("plan"), str):
            return StepPlan(node_id="init", next_node="plan")
        return StepPlan(node_id="init", next_node="reason")

    def plan_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, _ = ensure_react_vars(run)
        task = str(context.get("task", "") or "")

        allow = _effective_allowlist(runtime_ns)

        prompt = (
            "You are preparing a high-level execution plan for the user's request.\n"
            "Return a concise TODO list (5–12 steps) that is actionable and verifiable.\n"
            "Do not call tools yet. Do not include role prefixes like 'assistant:'.\n\n"
            f"User request:\n{task}\n\n"
            "Plan (markdown checklist):\n"
            "- [ ] ...\n"
        )

        emit("plan_request", {"tools": allow})

        payload: Dict[str, Any] = {"prompt": prompt, "params": {"temperature": 0.2}}
        sys = _system_prompt(runtime_ns)
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        return StepPlan(
            node_id="plan",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.plan_llm_response",
            ),
            next_node="plan_parse",
        )

    def plan_parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_react_vars(run)
        resp = temp.get("plan_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}
        plan_text = resp.get("content")
        plan = "" if plan_text is None else str(plan_text).strip()
        if not plan and isinstance(resp.get("data"), dict):
            plan = json.dumps(resp.get("data"), ensure_ascii=False, indent=2).strip()

        scratchpad["plan"] = plan
        temp.pop("plan_llm_response", None)

        if plan:
            context["messages"].append(_new_message(ctx, role="assistant", content=plan, metadata={"kind": "plan"}))
        emit("plan", {"plan": plan})
        return StepPlan(node_id="plan_parse", next_node="reason")

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)

        # Read from _limits (canonical) with fallback to scratchpad (backward compat)
        if "current_iteration" in limits:
            iteration = int(limits.get("current_iteration", 0) or 0)
            max_iterations = int(limits.get("max_iterations", 25) or 25)
        else:
            # Backward compatibility: use scratchpad
            iteration = int(scratchpad.get("iteration", 0) or 0)
            max_iterations = int(scratchpad.get("max_iterations") or 25)

        if max_iterations < 1:
            max_iterations = 1

        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        # Update both for transition period
        scratchpad["iteration"] = iteration + 1
        limits["current_iteration"] = iteration + 1

        task = str(context.get("task", "") or "")
        messages_view = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)

        # Refresh tool metadata BEFORE rendering Active Memory so token fitting stays accurate
        # (even though we do not render a "Tools (session)" block into Active Memory prompts).
        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        include_examples = bool(runtime_ns.get("tool_prompt_examples", True))
        if not include_examples:
            tool_specs = [{k: v for k, v in spec.items() if k != "examples"} for spec in tool_specs if isinstance(spec, dict)]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        inbox = runtime_ns.get("inbox", [])
        guidance = ""
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = " | ".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []
        req = logic.build_request(
            task=task,
            messages=messages_view,
            guidance=guidance,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        # Provide the selected active-context messages as proper chat messages (sanitized).
        #
        # IMPORTANT: When we send `messages`, do not also send a non-empty `prompt`.
        # Some providers/servers will append `prompt` as an extra user message even when the
        # current request is already present in `messages`, which duplicates user turns and
        # wastes context budget.
        payload: Dict[str, Any] = {"prompt": ""}
        payload["messages"] = _sanitize_llm_messages(messages_view, limits=limits)
        tools_payload = list(tool_specs)
        if tools_payload:
            payload["tools"] = tools_payload
        sys = _system_prompt(runtime_ns) or req.system_prompt
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys
        # Provider/model can be configured statically (create_react_workflow args)
        # or injected dynamically through durable vars in `_runtime` (Visual Agent pins).
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()
        params: Dict[str, Any] = {}
        if req.max_tokens is not None:
            params["max_tokens"] = req.max_tokens
        # Tool calling is formatting-sensitive; bias toward deterministic output when tools are present.
        params["temperature"] = 0.2 if tools_payload else 0.7
        payload["params"] = params

        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.llm_response",
            ),
            next_node="parse",
        )

    def tool_retry_minimal_node(run: RunState, ctx) -> StepPlan:
        """Recovery path when the model fabricates `observation[...]` logs instead of calling tools.

        This intentionally sends a minimal prompt (no History/Scratchpad) to reduce
        long-context contamination and force either a real tool call or a direct answer.
        """
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        task = str(context.get("task", "") or "")

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        include_examples = bool(runtime_ns.get("tool_prompt_examples", True))
        if not include_examples:
            tool_specs = [{k: v for k, v in spec.items() if k != "examples"} for spec in tool_specs if isinstance(spec, dict)]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)
        # Reuse the canonical agent rules from ReActLogic (but do not include history in prompt).
        sys_req = logic.build_request(task=task, messages=[], guidance="", iteration=0, max_iterations=0, vars=run.vars)

        bad_excerpt = str(temp.get("tool_retry_bad_content") or "").strip()
        temp.pop("tool_retry_bad_content", None)
        if len(bad_excerpt) > 240:
            bad_excerpt = bad_excerpt[:240].rstrip() + "…"

        prompt = (
            "Task:\n"
            f"{task}\n\n"
            "Your previous message was invalid: it contained fabricated `observation[...]` tool logs, but no tool was called.\n\n"
            "Now do ONE of the following:\n"
            "1) If you need more information to answer correctly, CALL ONE OR MORE TOOLS now using the required tool call format.\n"
            "2) If you can answer without tools, answer directly WITHOUT mentioning any tool calls or observations.\n\n"
            "Rules:\n"
            "- Do NOT write `observation[` anywhere.\n"
            "- Do NOT fabricate tool results.\n"
            "- If you call tools, output ONLY tool call block(s) (no extra text).\n"
            "- You MAY batch multiple tool calls by repeating the tool-call block once per call (prefer independent calls).\n"
        )
        if bad_excerpt:
            prompt += f"\nBad output excerpt (do not copy):\n{bad_excerpt}\n"

        payload: Dict[str, Any] = {"prompt": prompt}
        if tool_specs:
            payload["tools"] = tool_specs
        sys = _system_prompt(runtime_ns) or sys_req.system_prompt
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys

        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        payload["params"] = {"temperature": 0.2}

        emit("tool_retry_minimal", {"tools": allow, "has_excerpt": bool(bad_excerpt)})
        return StepPlan(
            node_id="tool_retry_minimal",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.llm_response",
            ),
            next_node="parse",
        )

    def empty_response_retry_node(run: RunState, ctx) -> StepPlan:
        """Recovery path when the model returns an empty message (no content, no tool calls).

        This is treated as an invalid agent step. We re-prompt with the original task plus
        recent tool evidence and explicitly require either tool calls or a substantive answer.
        """
        context, scratchpad, runtime_ns, _, _ = ensure_react_vars(run)
        task = str(context.get("task", "") or "")

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        include_examples = bool(runtime_ns.get("tool_prompt_examples", True))
        if not include_examples:
            tool_specs = [{k: v for k, v in spec.items() if k != "examples"} for spec in tool_specs if isinstance(spec, dict)]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        # Include recent tool outputs and user messages as evidence (bounded).
        messages = list(context.get("messages") or [])
        evidence_lines: list[str] = []
        tool_count = 0
        user_count = 0
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "tool" and isinstance(content, str) and content.strip():
                evidence_lines.append(content.strip())
                tool_count += 1
            elif role == "user" and isinstance(content, str) and content.strip():
                # Avoid duplicating the original task.
                if content.strip() != task.strip():
                    evidence_lines.append(content.strip())
                    user_count += 1
            if tool_count >= 6 and user_count >= 2:
                break
        evidence_lines.reverse()
        evidence = "\n\n".join(evidence_lines) if evidence_lines else "(no prior evidence captured)"

        # Build a strong corrective prompt. Prefer tools; allow a direct answer if truly possible.
        prompt = (
            "The previous assistant message was EMPTY (no content and no tool calls). This is invalid.\n"
            "Recover by continuing the task using the evidence below.\n\n"
            f"Task:\n{task}\n\n"
            f"Evidence (recent tool outputs + user messages):\n{evidence}\n\n"
            "Now do EXACTLY ONE of the following:\n"
            "1) CALL one or more tools to make progress (preferred).\n"
            "2) If you already have enough evidence, provide a concise final answer.\n\n"
            "Rules:\n"
            "- Do not output an empty message.\n"
            "- Do not ask the user a question in plain text; use the `ask_user` tool.\n"
            "- If you call tools, include the tool call(s) directly (no preamble).\n"
        )

        payload: Dict[str, Any] = {"prompt": prompt}
        if tool_specs:
            payload["tools"] = list(tool_specs)
        sys = _system_prompt(runtime_ns)
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()
        payload["params"] = {"temperature": 0.2}

        emit("empty_response_retry", {"tools": allow, "evidence": bool(evidence_lines)})
        return StepPlan(
            node_id="empty_response_retry",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.llm_response"),
            next_node="parse",
        )

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)

        def _sanitize_tool_call_content(text: str) -> str:
            """Remove tool-transcript markers from assistant content before persisting to history.

            Some OSS models may include internal transcript artifacts (e.g. fabricated
            `observation[...]` lines) or embed the tool call itself inside the message
            (`Action:` blocks). We keep only the user-facing prose that appears *before*
            such markers so the runtime doesn't persist fabricated logs into context.
            """
            if not isinstance(text, str) or not text.strip():
                return ""
            out_lines: list[str] = []
            for line in text.splitlines():
                lowered = line.lstrip().lower()
                if lowered.startswith("observation["):
                    break
                if lowered.startswith("action:"):
                    break
                if lowered.startswith("<|tool_call|>") or lowered.startswith("<tool_call>"):
                    break
                if lowered.startswith("```tool_call") or lowered.startswith("```tool_code"):
                    break
                out_lines.append(line)
            return "\n".join(out_lines).rstrip()

        def _should_retry_for_missing_tool_call(text: str) -> bool:
            if not isinstance(text, str) or not text.strip():
                return False
            # Some models echo our internal History formatting (e.g. `observation[web_search] (success): ...`)
            # as transcript lines. Treat only *line-start* occurrences as suspicious (avoid false positives
            # in JSON/code blocks), and only use this signal when no tools have actually run yet.
            for line in text.splitlines():
                if line.lstrip().lower().startswith("observation["):
                    return True
            return False

        def _extract_final_answer(text: str) -> tuple[bool, str]:
            """Return (is_explicit_final, stripped_answer)."""
            if not isinstance(text, str) or not text.strip():
                return False, ""
            s = text.lstrip()
            if s.upper().startswith("FINAL:"):
                return True, s[len("FINAL:") :].lstrip()
            return False, text

        emit(
            "parse",
            {
                "has_tool_calls": bool(tool_calls),
                "content": content,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id} for tc in tool_calls],
            },
        )
        temp.pop("llm_response", None)

        # Reset retry counter on any successful tool-call detection.
        if tool_calls:
            scratchpad["tool_retry_count"] = 0
            scratchpad["tool_retry_minimal_used"] = False

        if tool_calls:
            clean = _sanitize_tool_call_content(content)
            if clean.strip():
                context["messages"].append(_new_message(ctx, role="assistant", content=clean))
                if _flag(runtime_ns, "plan_mode", default=False):
                    updated = _extract_plan_update(clean)
                    if isinstance(updated, str) and updated.strip():
                        scratchpad["plan"] = updated.strip()
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            return StepPlan(node_id="parse", next_node="act")

        # Empty response is an invalid step: recover with a bounded retry that carries evidence.
        if not isinstance(content, str) or not content.strip():
            try:
                empty_retries = int(scratchpad.get("empty_response_retry_count") or 0)
            except Exception:
                empty_retries = 0

            if empty_retries < 2:
                scratchpad["empty_response_retry_count"] = empty_retries + 1
                emit("parse_retry_empty_response", {"retries": empty_retries + 1})
                return StepPlan(node_id="parse", next_node="empty_response_retry")

            safe = (
                "I can't proceed: the model repeatedly returned empty outputs (no content, no tool calls).\n"
                "Please retry, reduce context, or switch models."
            )
            context["messages"].append(_new_message(ctx, role="assistant", content=safe, metadata={"kind": "error"}))
            temp["final_answer"] = safe
            temp["pending_tool_calls"] = []
            scratchpad["empty_response_retry_count"] = 0
            return StepPlan(node_id="parse", next_node="maybe_review")

        # If the model appears to have produced a fake "observation[tool]" transcript instead of
        # calling tools, give it one corrective retry before treating the message as final.
        if not bool(scratchpad.get("used_tools")) and _should_retry_for_missing_tool_call(content):
            try:
                retries = int(scratchpad.get("tool_retry_count") or 0)
            except Exception:
                retries = 0
            if retries < 2:
                scratchpad["tool_retry_count"] = retries + 1
                inbox = runtime_ns.get("inbox")
                if not isinstance(inbox, list):
                    inbox = []
                    runtime_ns["inbox"] = inbox
                inbox.append(
                    {
                        "role": "system",
                        "content": (
                            "You wrote an `observation[...]` line, but no tool was actually called.\n"
                            "Do NOT fabricate tool outputs.\n"
                            "If you need to search/fetch/read/write, CALL a tool now using the required tool call format.\n"
                            "Never output `observation[...]` markers; those are context-only."
                        ),
                    }
                )
                emit("parse_retry_missing_tool_call", {"retries": retries + 1})
                return StepPlan(node_id="parse", next_node="reason")

            # If the model still fails after retries, attempt a single minimal-context recovery call
            # instead of accepting a fabricated transcript as the final answer.
            if not bool(scratchpad.get("tool_retry_minimal_used")):
                scratchpad["tool_retry_minimal_used"] = True
                scratchpad["tool_retry_count"] = 0
                temp["tool_retry_bad_content"] = content
                emit("parse_retry_minimal_context", {"retries": retries})
                return StepPlan(node_id="parse", next_node="tool_retry_minimal")

            safe = (
                "I can't proceed safely: the model repeatedly produced fabricated `observation[...]` tool logs instead of calling tools.\n"
                "Please retry, reduce context, or switch models."
            )
            context["messages"].append(_new_message(ctx, role="assistant", content=safe, metadata={"kind": "error"}))
            temp["final_answer"] = safe
            scratchpad["tool_retry_count"] = 0
            return StepPlan(node_id="parse", next_node="maybe_review")

        final_raw = _sanitize_tool_call_content(content)
        if not final_raw.strip():
            final_raw = str(content or "").strip()

        is_final, final_text = _extract_final_answer(final_raw)
        if is_final:
            if final_text:
                context["messages"].append(_new_message(ctx, role="assistant", content=final_text))
                if _flag(runtime_ns, "plan_mode", default=False):
                    updated = _extract_plan_update(final_text)
                    if isinstance(updated, str) and updated.strip():
                        scratchpad["plan"] = updated.strip()
            temp["final_answer"] = final_text or "No answer provided"
            temp["pending_tool_calls"] = []
            scratchpad["tool_retry_count"] = 0
            return StepPlan(node_id="parse", next_node="maybe_review")

        # Default: treat as a normal final answer even if it lacks an explicit FINAL marker.
        final = final_raw
        if final:
            context["messages"].append(_new_message(ctx, role="assistant", content=final))
            if _flag(runtime_ns, "plan_mode", default=False):
                updated = _extract_plan_update(final)
                if isinstance(updated, str) and updated.strip():
                    scratchpad["plan"] = updated.strip()

        temp["final_answer"] = final or "No answer provided"
        temp["pending_tool_calls"] = []
        scratchpad["tool_retry_count"] = 0
        scratchpad["empty_response_retry_count"] = 0
        return StepPlan(node_id="parse", next_node="maybe_review")

    def act_node(run: RunState, ctx) -> StepPlan:
        # Treat `_temp.pending_tool_calls` as a durable queue.
        # This avoids dropping calls when schema-only tools (ask_user/memory/etc.) are interleaved
        # with normal tools, and avoids re-asking the same question due to missing context.
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        raw_queue = temp.get("pending_tool_calls", [])
        if not isinstance(raw_queue, list) or not raw_queue:
            temp["pending_tool_calls"] = []
            return StepPlan(node_id="act", next_node="reason")

        allow = _effective_allowlist(runtime_ns)
        builtin_effect_tools = {
            "ask_user",
            "recall_memory",
            "inspect_vars",
            "remember",
            "remember_note",
            "compact_memory",
        }

        # Normalize queue items and assign stable call_ids once so splitting into batches does not
        # introduce duplicate ids.
        tool_queue: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_queue, start=1):
            if isinstance(item, ToolCall):
                d: Dict[str, Any] = {"name": item.name, "arguments": item.arguments, "call_id": item.call_id}
            elif isinstance(item, dict):
                d = dict(item)
            else:
                continue
            call_id = str(d.get("call_id") or "").strip()
            if not call_id:
                d["call_id"] = str(idx)
            tool_queue.append(d)

        if not tool_queue:
            temp["pending_tool_calls"] = []
            return StepPlan(node_id="act", next_node="reason")

        def _is_builtin(tc: Dict[str, Any]) -> bool:
            name = tc.get("name")
            return isinstance(name, str) and name in builtin_effect_tools

        # Execute one schema-only builtin (if it is next), otherwise execute the longest contiguous
        # prefix of normal tools. Leave the remainder queued for subsequent act/observe cycles.
        if _is_builtin(tool_queue[0]):
            tc = tool_queue[0]
            name = str(tc.get("name") or "").strip()
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}

            # Pop the builtin from the queue.
            temp["pending_tool_calls"] = list(tool_queue[1:])

            if name and name not in allow:
                temp["tool_results"] = {
                    "results": [
                        {
                            "call_id": str(tc.get("call_id") or ""),
                            "name": name,
                            "success": False,
                            "output": None,
                            "error": f"Tool '{name}' is not allowed for this agent",
                        }
                    ]
                }
                emit("act_blocked", {"tool": name})
                return StepPlan(node_id="act", next_node="observe")

            if name == "ask_user":
                question = str(args.get("question") or "Please provide input:")
                choices = args.get("choices")
                choices = list(choices) if isinstance(choices, list) else None

                # Persist the asked question in the durable message history so both the main model
                # and the reviewer can see what was asked (and avoid re-asking).
                msgs = context.get("messages")
                if isinstance(msgs, list):
                    content = f"[Agent question]: {question}"
                    last = msgs[-1] if msgs else None
                    last_role = last.get("role") if isinstance(last, dict) else None
                    last_meta = last.get("metadata") if isinstance(last, dict) else None
                    last_kind = last_meta.get("kind") if isinstance(last_meta, dict) else None
                    last_content = last.get("content") if isinstance(last, dict) else None
                    if not (last_role == "assistant" and last_kind == "ask_user_prompt" and str(last_content or "") == content):
                        msgs.append(_new_message(ctx, role="assistant", content=content, metadata={"kind": "ask_user_prompt"}))

                emit("ask_user", {"question": question, "choices": choices or []})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.ASK_USER,
                        payload={"prompt": question, "choices": choices, "allow_free_text": True},
                        result_key="_temp.user_response",
                    ),
                    next_node="handle_user_response",
                )

            if name == "recall_memory":
                payload = dict(args)
                payload.setdefault("tool_name", "recall_memory")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_query", {"query": payload.get("query"), "span_id": payload.get("span_id")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.MEMORY_QUERY, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
                )

            if name == "inspect_vars":
                payload = dict(args)
                payload.setdefault("tool_name", "inspect_vars")
                payload.setdefault("call_id", tc.get("call_id") or "vars")
                emit("vars_query", {"path": payload.get("path")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.VARS_QUERY, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
                )

            if name == "remember":
                payload = dict(args)
                payload.setdefault("tool_name", "remember")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_tag", {"span_id": payload.get("span_id"), "tags": payload.get("tags")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.MEMORY_TAG, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
                )

            if name == "remember_note":
                payload = dict(args)
                payload.setdefault("tool_name", "remember_note")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_note", {"note": payload.get("note"), "tags": payload.get("tags")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.MEMORY_NOTE, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
                )

            if name == "compact_memory":
                payload = dict(args)
                payload.setdefault("tool_name", "compact_memory")
                payload.setdefault("call_id", tc.get("call_id") or "compact")
                emit(
                    "memory_compact",
                    {
                        "preserve_recent": payload.get("preserve_recent"),
                        "mode": payload.get("compression_mode"),
                        "focus": payload.get("focus"),
                    },
                )
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.MEMORY_COMPACT, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
                )

            # Unknown builtin: continue with the queue (best-effort).
            if temp.get("pending_tool_calls"):
                return StepPlan(node_id="act", next_node="act")
            return StepPlan(node_id="act", next_node="reason")

        # Normal tools: execute contiguous prefix until the next builtin.
        batch: List[Dict[str, Any]] = []
        for tc in tool_queue:
            if _is_builtin(tc):
                break
            batch.append(tc)

        remaining = tool_queue[len(batch) :]
        temp["pending_tool_calls"] = list(remaining)

        # Emit observability events for the batch.
        for tc in batch:
            emit("act", {"tool": tc.get("name", ""), "args": tc.get("arguments", {}), "call_id": str(tc.get("call_id") or "")})

        formatted_calls: List[Dict[str, Any]] = []
        for tc in batch:
            formatted_calls.append(
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", {}), "call_id": str(tc.get("call_id") or "")}
            )

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls, "allowed_tools": list(allow)},
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_react_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []
        if results:
            scratchpad["used_tools"] = True

        # Prefer a tool-supplied human/LLM-friendly rendering when present.
        def _display(v: Any) -> str:
            if isinstance(v, dict):
                rendered = v.get("rendered")
                if isinstance(rendered, str) and rendered.strip():
                    return rendered.strip()
            return "" if v is None else str(v)

        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")
            display = _display(output)
            if not success:
                # Preserve structured outputs for provenance, but show a clean string to the LLM/UI.
                display = _display(output) if isinstance(output, dict) else str(error or output)
            rendered = logic.format_observation(
                name=name,
                output=display,
                success=success,
            )
            emit("observe", {"tool": name, "success": success, "result": rendered})

            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={
                        "name": name,
                        "call_id": r.get("call_id"),
                        "success": success,
                    },
                )
            )

        temp.pop("tool_results", None)
        # Reset verifier/review rounds after executing tools. This enables repeated
        # verify→act→observe cycles without immediately hitting review_max_rounds.
        scratchpad["review_count"] = 0
        pending = temp.get("pending_tool_calls", [])
        if isinstance(pending, list) and pending:
            return StepPlan(node_id="observe", next_node="act")
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def maybe_review_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, runtime_ns, _, _ = ensure_react_vars(run)

        if not _flag(runtime_ns, "review_mode", default=False):
            return StepPlan(node_id="maybe_review", next_node="done")

        max_rounds = _int(runtime_ns, "review_max_rounds", default=1)
        if max_rounds < 0:
            max_rounds = 0
        count = scratchpad.get("review_count")
        try:
            count_int = int(count or 0)
        except Exception:
            count_int = 0

        if count_int >= max_rounds:
            return StepPlan(node_id="maybe_review", next_node="done")

        scratchpad["review_count"] = count_int + 1
        return StepPlan(node_id="maybe_review", next_node="review")

    def review_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)

        task = str(context.get("task", "") or "")
        plan = scratchpad.get("plan")
        plan_text = str(plan).strip() if isinstance(plan, str) and plan.strip() else "(no plan)"

        allow = _effective_allowlist(runtime_ns)

        def _truncate_block(text: str, *, max_chars: int) -> str:
            s = str(text or "")
            if max_chars <= 0:
                return s
            if len(s) <= max_chars:
                return s
            suffix = f"\n… (truncated, {len(s):,} chars total)"
            keep = max_chars - len(suffix)
            if keep < 200:
                keep = max_chars
                suffix = ""
            return s[:keep].rstrip() + suffix

        def _format_allowed_tools() -> str:
            # Prefer the already-computed tool_specs (created in reason_node) to avoid
            # re-materializing tool definitions and to keep formatting stable.
            specs = runtime_ns.get("tool_specs")
            if not isinstance(specs, list) or not specs:
                defs = _allowed_tool_defs(allow)
                specs = [t.to_dict() for t in defs]
            lines: list[str] = []
            for spec in specs:
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name") or "").strip()
                if not name:
                    continue
                params = spec.get("parameters")
                props = params.get("properties", {}) if isinstance(params, dict) else {}
                keys = sorted([k for k in props.keys() if isinstance(k, str)])
                if keys:
                    lines.append(f"- {name}({', '.join(keys)})")
                else:
                    lines.append(f"- {name}()")
            return "\n".join(lines) if lines else "(no tools available)"

        # Include recent tool outputs for evidence-based review.
        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        try:
            tool_limit = int(limits.get("review_max_tool_output_chars", -1))
        except Exception:
            tool_limit = -1
        try:
            answer_limit = int(limits.get("review_max_answer_chars", -1))
        except Exception:
            answer_limit = -1

        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                tool_msgs.append(_truncate_block(content.strip(), max_chars=tool_limit))
            if len(tool_msgs) >= 8:
                break
        tool_msgs.reverse()
        observations = "\n\n".join(tool_msgs) if tool_msgs else "(no tool outputs)"

        # Include recent user messages (especially ask_user responses) so the reviewer can
        # avoid re-asking questions the user already answered.
        try:
            user_limit = int(limits.get("review_max_user_message_chars", -1))
        except Exception:
            user_limit = -1

        user_msgs: list[str] = []
        ask_prompts: list[str] = []
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                if content.strip() != task.strip():
                    user_msgs.append(_truncate_block(content.strip(), max_chars=user_limit))
                    if len(user_msgs) >= 4:
                        break
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
            if not isinstance(meta, dict) or meta.get("kind") != "ask_user_prompt":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                ask_prompts.append(_truncate_block(content.strip(), max_chars=user_limit))
                if len(ask_prompts) >= 4:
                    break

        user_msgs.reverse()
        ask_prompts.reverse()
        user_context = "\n\n".join(user_msgs) if user_msgs else "(no additional user messages)"
        asked_context = "\n\n".join(ask_prompts) if ask_prompts else "(no ask_user prompts recorded)"

        # The verifier should primarily judge based on tool outputs. Only include an answer
        # excerpt when we have no tool evidence (pure Q&A runs).
        answer_raw = str(run.vars.get("_temp", {}).get("final_answer") or "")
        answer_excerpt = ""
        if not tool_msgs and answer_raw.strip():
            answer_excerpt = _truncate_block(answer_raw.strip(), max_chars=answer_limit)

        prompt = (
            "You are a verifier. Review whether the user's request has been fully satisfied.\n"
            "Be strict: only count actions that are supported by the tool outputs.\n"
            "If anything is missing, propose the NEXT ACTIONS.\n"
            "Prefer returning `next_tool_calls` over `next_prompt`.\n"
            "Return JSON ONLY.\n\n"
            f"User request:\n{task}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Recent ask_user prompts:\n{asked_context}\n\n"
            f"Recent user messages:\n{user_context}\n\n"
            + (f"Current answer (excerpt):\n{answer_excerpt}\n\n" if answer_excerpt else "")
            + f"Tool outputs:\n{observations}\n\n"
            f"Allowed tools:\n{_format_allowed_tools()}\n\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "missing": {"type": "array", "items": {"type": "string"}},
                "next_prompt": {"type": "string"},
                "next_tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["complete", "missing", "next_prompt", "next_tool_calls"],
            "additionalProperties": False,
        }

        emit("review_request", {"tool_messages": len(tool_msgs)})

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_schema": schema,
            "response_schema_name": "ReActVerifier",
            "params": {"temperature": 0.2},
        }
        sys = _system_prompt(runtime_ns)
        if sys is not None:
            payload["system_prompt"] = sys
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        return StepPlan(
            node_id="review",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.review_llm_response",
            ),
            next_node="review_parse",
        )

    def review_parse_node(run: RunState, ctx) -> StepPlan:
        _, _, runtime_ns, temp, _ = ensure_react_vars(run)
        resp = temp.get("review_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}

        data = resp.get("data")
        if data is None and isinstance(resp.get("content"), str):
            try:
                data = json.loads(resp["content"])
            except Exception:
                data = None
        if not isinstance(data, dict):
            data = {}

        complete = bool(data.get("complete"))
        missing = data.get("missing") if isinstance(data.get("missing"), list) else []
        next_prompt = data.get("next_prompt")
        next_prompt_text = str(next_prompt or "").strip()
        next_tool_calls_raw = data.get("next_tool_calls")
        next_tool_calls: list[dict[str, Any]] = []
        if isinstance(next_tool_calls_raw, list):
            for item in next_tool_calls_raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                args = item.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name:
                    next_tool_calls.append({"name": name, "arguments": args})

        emit("review", {"complete": complete, "missing": missing})
        temp.pop("review_llm_response", None)

        if complete:
            return StepPlan(node_id="review_parse", next_node="done")

        if next_tool_calls:
            temp["pending_tool_calls"] = next_tool_calls
            emit("review_tool_calls", {"count": len(next_tool_calls)})
            return StepPlan(node_id="review_parse", next_node="act")

        # Behavioral validation: if incomplete but no tool calls, re-ask reviewer once with stricter rules.
        if not complete and not next_tool_calls:
            try:
                retry_count = int(runtime_ns.get("review_retry_count") or 0)
            except Exception:
                retry_count = 0
            if retry_count < 1:
                runtime_ns["review_retry_count"] = retry_count + 1
                inbox = runtime_ns.get("inbox")
                if not isinstance(inbox, list):
                    inbox = []
                    runtime_ns["inbox"] = inbox
                inbox.append(
                    {
                        "content": (
                            "[Review] Your last review output was not actionable. "
                            "If incomplete, you MUST return at least one `next_tool_call` "
                            "(use `ask_user` if you need clarification). Return JSON only."
                        )
                    }
                )
                emit("review_retry_unactionable", {"retry": retry_count + 1})
                return StepPlan(node_id="review_parse", next_node="review")

        runtime_ns["review_retry_count"] = 0
        if next_prompt_text:
            inbox = runtime_ns.get("inbox")
            if not isinstance(inbox, list):
                inbox = []
                runtime_ns["inbox"] = inbox
            inbox.append({"content": f"[Review] {next_prompt_text}"})
        return StepPlan(node_id="review_parse", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_react_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(
            _new_message(ctx, role="user", content=f"[User response]: {response_text}")
        )
        temp.pop("user_response", None)

        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_react_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer})

        # Prefer _limits.current_iteration, fall back to scratchpad
        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)

        # Persist the final user-facing answer into the conversation history so it shows up
        # in /history and becomes part of the next run's seed context.
        messages = context.get("messages")
        if isinstance(messages, list):
            last = messages[-1] if messages else None
            last_role = last.get("role") if isinstance(last, dict) else None
            last_content = last.get("content") if isinstance(last, dict) else None
            if last_role != "assistant" or str(last_content or "") != answer:
                messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer"}))

        return StepPlan(
            node_id="done",
            complete_output={
                "answer": answer,
                "iterations": iterations,
                "messages": list(context.get("messages") or []),
            },
        )

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, _, limits = ensure_react_vars(run)

        # Prefer _limits, fall back to scratchpad
        max_iterations = int(limits.get("max_iterations", 0) or scratchpad.get("max_iterations", 25) or 25)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations})

        messages = list(context.get("messages") or [])
        last_content = messages[-1]["content"] if messages else "Max iterations reached"
        return StepPlan(
            node_id="max_iterations",
            complete_output={
                "answer": last_content,
                "iterations": max_iterations,
                "messages": messages,
            },
        )

    return WorkflowSpec(
        workflow_id=str(workflow_id or "react_agent"),
        entry_node="init",
        nodes={
            "init": init_node,
            "plan": plan_node,
            "plan_parse": plan_parse_node,
            "reason": reason_node,
            "tool_retry_minimal": tool_retry_minimal_node,
            "empty_response_retry": empty_response_retry_node,
            "parse": parse_node,
            "act": act_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "maybe_review": maybe_review_node,
            "review": review_node,
            "review_parse": review_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )
