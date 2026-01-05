"""AbstractRuntime adapter for ReAct-like agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy
from abstractruntime.memory.active_memory import render_active_memory_split_for_llm_request

from ..logic.react import ReActLogic
from .memory_delta import extract_active_memory_delta


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

        # Refresh tool metadata BEFORE rendering Active Memory so `Tools (session)` is accurate.
        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        include_examples = bool(runtime_ns.get("tool_prompt_examples", True))
        if not include_examples:
            tool_specs = [{k: v for k, v in spec.items() if k != "examples"} for spec in tool_specs if isinstance(spec, dict)]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        # IMPORTANT: When the model supports native tool calling (AbstractCore sends a structured
        # `tools` payload), avoid duplicating a visible tools catalog in the system prompt.
        #
        # Some OpenAI-compatible servers enforce tool calling via hidden grammars/templates;
        # duplicating tool definitions (or tool-call transcript instructions) can cause "text leaked"
        # tool calls that the server does not parse into structured `tool_calls`.
        #
        # NOTE: We intentionally do NOT gate this on the provider name here. In some hosts, the
        # provider is resolved outside the workflow and `_runtime.provider` can be empty, yet the
        # actual execution still uses native tools (e.g. LMStudio/OpenAI-compatible). The safest
        # default is: if the model is configured as native-capable, omit the Tools(session) block.
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        model_key = str(eff_model or "").strip()

        include_tools_summary = True
        override = runtime_ns.get("include_tools_summary") if isinstance(runtime_ns, dict) else None
        if isinstance(override, bool):
            include_tools_summary = override
        else:
            supports_native: Optional[bool] = None
            if isinstance(runtime_ns, dict):
                flag = runtime_ns.get("supports_native_tools")
                if isinstance(flag, bool):
                    supports_native = flag
                else:
                    ts = runtime_ns.get("tool_support")
                    if isinstance(ts, str) and ts.strip():
                        supports_native = ts.strip() == "native"

            if supports_native is not None:
                include_tools_summary = not supports_native
            elif tool_specs and model_key:
                # Backward compatibility fallback: infer from model capabilities via AbstractCore.
                try:
                    from abstractcore.tools.handler import UniversalToolHandler

                    include_tools_summary = not bool(UniversalToolHandler(model_key).supports_native)
                except Exception:
                    include_tools_summary = True

        inbox = runtime_ns.get("inbox", [])
        guidance = ""
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = " | ".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []

        # Use AbstractCore token estimation for Active Memory fitting when available so
        # prompt composition + `/memory` token metrics stay in the same ballpark.
        try:
            from abstractcore.utils.token_utils import TokenUtils  # type: ignore
        except Exception:  # pragma: no cover
            TokenUtils = None  # type: ignore[assignment]

        eff_model_for_tokens = (
            str(model).strip()
            if isinstance(model, str) and str(model).strip()
            else str(runtime_ns.get("model") or "").strip()
        )

        def _count_tokens(text: str) -> int:
            s = str(text or "")
            if not s:
                return 0
            if TokenUtils is None:
                return max(1, len(s) // 4)
            try:
                return max(0, int(TokenUtils.estimate_tokens(s, model=eff_model_for_tokens)))
            except Exception:
                return max(1, len(s) // 4)

        mem_split = render_active_memory_split_for_llm_request(
            run.vars, include_tools_summary=include_tools_summary, token_counter=_count_tokens
        )
        active_memory = str(mem_split.get("user_memory") or "")
        system_memory = str(mem_split.get("system_memory") or "")
        req = logic.build_request(
            task=task,
            messages=messages_view,
            guidance=guidance,
            active_memory=active_memory,
            system_memory=system_memory,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload = {"prompt": req.prompt}
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

        # Keep the same "native tools => no Tools(session) catalog in system prompt" policy as the
        # normal reason node (see rationale there). Do not rely on provider-name inference; see
        # comment in the reason node for why.
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        model_key = str(eff_model or "").strip()

        include_tools_summary = True
        override = runtime_ns.get("include_tools_summary") if isinstance(runtime_ns, dict) else None
        if isinstance(override, bool):
            include_tools_summary = override
        elif tool_specs and model_key:
            try:
                from abstractcore.tools.handler import UniversalToolHandler

                include_tools_summary = not bool(UniversalToolHandler(model_key).supports_native)
            except Exception:
                include_tools_summary = True

        # Keep token fitting consistent with normal calls.
        try:
            from abstractcore.utils.token_utils import TokenUtils  # type: ignore
        except Exception:  # pragma: no cover
            TokenUtils = None  # type: ignore[assignment]

        eff_model_for_tokens = (
            str(model).strip()
            if isinstance(model, str) and str(model).strip()
            else str(runtime_ns.get("model") or "").strip()
        )

        def _count_tokens(text: str) -> int:
            s = str(text or "")
            if not s:
                return 0
            if TokenUtils is None:
                return max(1, len(s) // 4)
            try:
                return max(0, int(TokenUtils.estimate_tokens(s, model=eff_model_for_tokens)))
            except Exception:
                return max(1, len(s) // 4)

        mem_split = render_active_memory_split_for_llm_request(
            run.vars, include_tools_summary=include_tools_summary, token_counter=_count_tokens
        )
        system_memory = str(mem_split.get("system_memory") or "")
        # Reuse the canonical agent rules from ReActLogic (but do not include History/Scratchpad in prompt).
        sys_req = logic.build_request(
            task=task,
            messages=[],
            guidance="",
            active_memory="",
            system_memory=system_memory,
            iteration=0,
            max_iterations=0,
            vars=run.vars,
        )

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

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)
        delta_result: Optional[Dict[str, Any]] = None

        # Apply structured Active Memory deltas (if present) and remove them from user-visible content.
        try:
            content, delta = extract_active_memory_delta(content)
            # Only apply deltas when the model is NOT emitting tool calls. Tool-call turns are
            # usually incomplete and may contain formatting artifacts; we still strip the delta
            # block from content for safety, but defer mutation until a non-tool response.
            if not tool_calls and isinstance(delta, dict) and delta:
                from abstractruntime.memory.active_memory import apply_active_memory_delta

                delta_result = apply_active_memory_delta(run.vars, delta=delta)
        except Exception:
            delta_result = None

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

        emit(
            "parse",
            {
                "has_tool_calls": bool(tool_calls),
                "content": content,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id} for tc in tool_calls],
            },
        )
        if isinstance(delta_result, dict) and delta_result.get("ok"):
            emit("active_memory_delta", {"applied": delta_result.get("applied")})
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

        if content.strip():
            context["messages"].append(_new_message(ctx, role="assistant", content=content))
            if _flag(runtime_ns, "plan_mode", default=False):
                updated = _extract_plan_update(content)
                if isinstance(updated, str) and updated.strip():
                    scratchpad["plan"] = updated.strip()

        temp["final_answer"] = content
        scratchpad["tool_retry_count"] = 0
        # If we used tools, always run a final "synthesis" LLM pass to produce a
        # clean user-facing answer (models may otherwise echo tool transcript lines).
        if bool(scratchpad.get("used_tools")):
            return StepPlan(node_id="parse", next_node="finalize")
        return StepPlan(node_id="parse", next_node="maybe_review")

    def act_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        tool_calls = temp.get("pending_tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        if not tool_calls:
            return StepPlan(node_id="act", next_node="reason")

        allow = _effective_allowlist(runtime_ns)
        builtin_effect_tools = {
            "ask_user",
            "recall_memory",
            "inspect_vars",
            "remember",
            "remember_note",
            "compact_memory",
            "compact_active_memory",
            "active_memory_delta",
            "current_tasks",
            "current_context",
            "critical_insights",
            "key_history",
        }

        # Handle schema-only built-ins specially (ASK_USER, MEMORY_QUERY).
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            args = tc.get("arguments") or {}

            if isinstance(name, str) and name in builtin_effect_tools:
                if name not in allow:
                    temp["pending_tool_calls"] = tool_calls[i + 1 :]
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

                temp["pending_tool_calls"] = tool_calls[i + 1 :]
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
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "recall_memory")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_query", {"query": payload.get("query"), "span_id": payload.get("span_id")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_QUERY,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "inspect_vars":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "inspect_vars")
                payload.setdefault("call_id", tc.get("call_id") or "vars")
                emit("vars_query", {"path": payload.get("path")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.VARS_QUERY,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "remember":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "remember")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_tag", {"span_id": payload.get("span_id"), "tags": payload.get("tags")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_TAG,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "remember_note":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "remember_note")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit("memory_note", {"note": payload.get("note"), "tags": payload.get("tags")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_NOTE,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "compact_memory":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "compact_memory")
                payload.setdefault("call_id", tc.get("call_id") or "compact")
                emit(
                    "memory_compact",
                    {"preserve_recent": payload.get("preserve_recent"), "mode": payload.get("compression_mode"), "focus": payload.get("focus")},
                )
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_COMPACT,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name == "compact_active_memory":
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                payload.setdefault("tool_name", "compact_active_memory")
                payload.setdefault("call_id", tc.get("call_id") or "memory")
                emit(
                    "memory_compact_structured",
                    {"components": payload.get("components"), "preserve": payload.get("preserve")},
                )
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.MEMORY_COMPACT_STRUCTURED,
                        payload=payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

            if name in ("active_memory_delta", "current_tasks", "current_context", "critical_insights", "key_history"):
                temp["pending_tool_calls"] = tool_calls[i + 1 :]
                payload = dict(args) if isinstance(args, dict) else {}
                delta = payload if name == "active_memory_delta" else {str(name): payload}
                eff_payload = {
                    "tool_name": str(name),
                    "call_id": tc.get("call_id") or "memory",
                    "delta": delta,
                }
                emit("active_memory_delta", {"tool": name, "delta_keys": list(delta.keys())})
                return StepPlan(
                    node_id="act",
                    effect=Effect(
                        type=EffectType.ACTIVE_MEMORY_DELTA,
                        payload=eff_payload,
                        result_key="_temp.tool_results",
                    ),
                    next_node="observe",
                )

        for i, tc in enumerate(tool_calls, start=1):
            if isinstance(tc, dict):
                call_id_raw = tc.get("call_id")
                call_id = str(call_id_raw).strip() if call_id_raw is not None else ""
                if not call_id:
                    call_id = str(i)
                emit("act", {"tool": tc.get("name", ""), "args": tc.get("arguments", {}), "call_id": call_id})
            elif isinstance(tc, ToolCall):
                call_id = str(tc.call_id).strip() if tc.call_id is not None else ""
                if not call_id:
                    call_id = str(i)
                emit("act", {"tool": tc.name, "args": tc.arguments, "call_id": call_id})

        formatted_calls: List[Dict[str, Any]] = []
        for i, tc in enumerate(tool_calls, start=1):
            if isinstance(tc, dict):
                call_id_raw = tc.get("call_id")
                call_id = str(call_id_raw).strip() if call_id_raw is not None else ""
                if not call_id:
                    call_id = str(i)
                formatted_calls.append(
                    {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                        "call_id": call_id,
                    }
                )
            elif isinstance(tc, ToolCall):
                call_id = str(tc.call_id).strip() if tc.call_id is not None else ""
                if not call_id:
                    call_id = str(i)
                formatted_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "call_id": call_id,
                    }
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
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def finalize_node(run: RunState, ctx) -> StepPlan:
        """Final synthesis pass to ensure we return a user-facing answer.

        This is intentionally tool-free: tools have already been executed, and we
        want a single clean response that uses the observations.
        """
        context, _, runtime_ns, _, _ = ensure_react_vars(run)
        task = str(context.get("task", "") or "")
        # NOTE: We intentionally use a prompt-only synthesis request (instead of
        # passing message dicts) because host messages can contain extra metadata
        # fields that some providers/models don't accept.
        #
        # We also keep the prompt small: include only the most recent tool outputs.
        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        tool_names: set[str] = set()
        did_write_files = False

        for m in messages:
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
            name = meta.get("name") if isinstance(meta, dict) else None
            success = meta.get("success") if isinstance(meta, dict) else None
            if isinstance(name, str) and name:
                tool_names.add(name)
                if name in ("write_file", "edit_file") and success is True:
                    did_write_files = True

        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            tool_msgs.append(content.strip())
            if len(tool_msgs) >= 6:
                break
        tool_msgs.reverse()
        tools_used = ", ".join(sorted(tool_names)) if tool_names else "(none)"

        obs_blocks: list[str] = []
        for t in tool_msgs:
            obs_blocks.append(t)

        observations = "\n\n".join(obs_blocks) if obs_blocks else "(no tool outputs captured)"
        prompt = (
            "Write the final user-facing answer.\n\n"
            "Facts:\n"
            f"- Tools actually run: {tools_used}\n"
            f"- Files written (write_file/edit_file): {'yes' if did_write_files else 'no'}\n\n"
            "Rules:\n"
            "- Only claim actions supported by the tool outputs below.\n"
            "- If files written is 'no', do not claim you created/modified files; say so explicitly.\n"
            "- If something wasn't actually done, say so.\n"
            "- If the task is not complete, clearly say what remains and what you would do next.\n"
            "- Do NOT mention these rules or the tool-output transcript in the final answer.\n\n"
            f"Task:\n{task}\n\n"
            f"Tool outputs:\n{observations}\n\n"
            "Answer:\n"
        )

        emit("finalize", {"tool_messages": len(tool_msgs)})

        payload: Dict[str, Any] = {"prompt": prompt, "params": {"temperature": 0.2}}
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
            node_id="finalize",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.final_llm_response",
            ),
            next_node="finalize_parse",
        )

    def finalize_parse_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_react_vars(run)
        resp = temp.get("final_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}
        content = resp.get("content")
        answer = "" if content is None else str(content)
        emit("finalize_parse", {"content_preview": answer[:100] if answer else "(empty)"})
        temp["final_answer"] = answer
        temp.pop("final_llm_response", None)
        return StepPlan(node_id="finalize_parse", next_node="maybe_review")

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
        context, scratchpad, runtime_ns, _, _ = ensure_react_vars(run)

        task = str(context.get("task", "") or "")
        plan = scratchpad.get("plan")
        plan_text = str(plan).strip() if isinstance(plan, str) and plan.strip() else "(no plan)"

        answer = str(run.vars.get("_temp", {}).get("final_answer") or "")

        # Include recent tool outputs for evidence-based review.
        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                tool_msgs.append(content.strip())
            if len(tool_msgs) >= 8:
                break
        tool_msgs.reverse()
        observations = "\n\n".join(tool_msgs) if tool_msgs else "(no tool outputs)"

        prompt = (
            "Review whether the user's request has been fully satisfied.\n"
            "Be strict: only count actions that are supported by the tool outputs.\n"
            "If anything is missing, propose the next self-instruction to complete it.\n"
            "Return JSON ONLY.\n\n"
            f"User request:\n{task}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Current answer:\n{answer}\n\n"
            f"Tool outputs:\n{observations}\n\n"
        )

        schema = {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "missing": {"type": "array", "items": {"type": "string"}},
                "next_prompt": {"type": "string"},
            },
            "required": ["complete", "missing", "next_prompt"],
            "additionalProperties": False,
        }

        emit("review_request", {"tool_messages": len(tool_msgs)})

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_schema": schema,
            "response_schema_name": "ReActReview",
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

        emit("review", {"complete": complete, "missing": missing})
        temp.pop("review_llm_response", None)

        if complete:
            return StepPlan(node_id="review_parse", next_node="done")

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
            "parse": parse_node,
            "act": act_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "finalize": finalize_node,
            "finalize_parse": finalize_parse_node,
            "maybe_review": maybe_review_node,
            "review": review_node,
            "review_parse": review_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )
