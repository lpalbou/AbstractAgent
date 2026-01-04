"""AbstractRuntime adapter for CodeAct agents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall, ToolDefinition
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy
from abstractruntime.memory.active_memory import render_active_memory_split_for_llm_request

from ..logic.codeact import CodeActLogic
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


def ensure_codeact_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

    for key in ("llm_response", "tool_results", "pending_tool_calls", "user_response", "final_answer", "pending_code"):
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

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_codeact_workflow(
    *,
    logic: CodeActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> WorkflowSpec:
    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)

    def _current_tool_defs() -> list[ToolDefinition]:
        defs = getattr(logic, "tools", None)
        if not isinstance(defs, list):
            try:
                defs = list(defs)  # type: ignore[arg-type]
            except Exception:
                defs = []
        return [t for t in defs if getattr(t, "name", None)]

    def _tool_by_name() -> dict[str, ToolDefinition]:
        out: dict[str, ToolDefinition] = {}
        for t in _current_tool_defs():
            name = getattr(t, "name", None)
            if isinstance(name, str) and name.strip():
                out[name] = t
        return out

    def _default_allowlist() -> list[str]:
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
        if raw is None:
            return []
        if isinstance(raw, str):
            val = raw.strip()
            return [val] if val else []
        if isinstance(raw, list):
            out: list[str] = []
            seen: set[str] = set()
            for item in raw:
                if not isinstance(item, str):
                    continue
                name = item.strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append(name)
            return out
        return []

    def _effective_allowlist(runtime_ns: Dict[str, Any]) -> list[str]:
        if isinstance(runtime_ns, dict) and "allowed_tools" in runtime_ns:
            normalized = _normalize_allowlist(runtime_ns.get("allowed_tools"))
            # Filter to currently known tools (dynamic), preserving order.
            current = _tool_by_name()
            filtered = [name for name in normalized if name in current]
            runtime_ns["allowed_tools"] = filtered
            return filtered
        return list(_default_allowlist())

    def _allowed_tool_defs(allowlist: list[str]) -> list[ToolDefinition]:
        tool_by_name = _tool_by_name()
        out: list[ToolDefinition] = []
        for name in allowlist:
            tool = tool_by_name.get(name)
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
        if not re.search(r"(?m)^\s*(?:[-*]|\d+\.)\s+", plan_text):
            return None
        return plan_text

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)
        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]
        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        runtime_ns["tool_specs"] = [t.to_dict() for t in allowed_defs]
        runtime_ns["toolset_id"] = _compute_toolset_id(runtime_ns["tool_specs"])
        runtime_ns.setdefault("allowed_tools", allow)
        runtime_ns.setdefault("inbox", [])

        emit("init", {"task": task})
        if _flag(runtime_ns, "plan_mode", default=False) and not isinstance(scratchpad.get("plan"), str):
            return StepPlan(node_id="init", next_node="plan")
        return StepPlan(node_id="init", next_node="reason")

    def plan_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, _ = ensure_codeact_vars(run)
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
        context, scratchpad, _, temp, _ = ensure_codeact_vars(run)
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
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)

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

        inbox = runtime_ns.get("inbox", [])
        guidance = ""
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = " | ".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []

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

        # Same policy as ReAct: when the model supports native tools, avoid duplicating a visible
        # tools catalog in the system prompt (can conflict with provider tool grammars).
        #
        # Do not rely on provider-name inference here: some hosts resolve the provider outside the
        # workflow and `_runtime.provider` can be empty, yet native tools are still used.
        model_key = str(runtime_ns.get("model") or "").strip()

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

        # Use AbstractCore token estimation for Active Memory fitting when available so
        # prompt composition + `/memory` token metrics stay in the same ballpark.
        try:
            from abstractcore.utils.token_utils import TokenUtils  # type: ignore
        except Exception:  # pragma: no cover
            TokenUtils = None  # type: ignore[assignment]

        # CodeAct workflow does not capture provider/model as closure vars; rely on runtime vars.
        eff_model_for_tokens = str(runtime_ns.get("model") or "").strip()

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
            task=str(context.get("task", "") or ""),
            messages=messages_view,
            guidance=guidance,
            active_memory=active_memory,
            system_memory=system_memory,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload: Dict[str, Any] = {"prompt": req.prompt, "tools": list(tool_specs)}
        sys = _system_prompt(runtime_ns) or req.system_prompt
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys
        if req.max_tokens is not None:
            payload["params"] = {"max_tokens": req.max_tokens}

        return StepPlan(
            node_id="reason",
            effect=Effect(
                type=EffectType.LLM_CALL,
                payload=payload,
                result_key="_temp.llm_response",
            ),
            next_node="parse",
        )

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_codeact_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)
        delta_result: Optional[Dict[str, Any]] = None

        # Apply structured Active Memory deltas (if present) and remove them from user-visible content.
        try:
            content, delta = extract_active_memory_delta(content)
            # Only apply deltas when the model is NOT emitting tool calls. Tool-call turns are
            # usually incomplete; we still strip the delta block from content for safety.
            if not tool_calls and isinstance(delta, dict) and delta:
                from abstractruntime.memory.active_memory import apply_active_memory_delta

                delta_result = apply_active_memory_delta(run.vars, delta=delta)
        except Exception:
            delta_result = None

        if content:
            context["messages"].append(_new_message(ctx, role="assistant", content=content))
            if _flag(runtime_ns, "plan_mode", default=False):
                updated = _extract_plan_update(content)
                if isinstance(updated, str) and updated.strip():
                    scratchpad["plan"] = updated.strip()

        temp.pop("llm_response", None)
        emit("parse", {"has_tool_calls": bool(tool_calls), "content_preview": (content[:100] if content else "(no content)")})
        if isinstance(delta_result, dict) and delta_result.get("ok"):
            emit("active_memory_delta", {"applied": delta_result.get("applied")})

        if tool_calls:
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            return StepPlan(node_id="parse", next_node="act")

        code = logic.extract_code(content)
        if code:
            temp["pending_code"] = code
            return StepPlan(node_id="parse", next_node="execute_code")

        temp["final_answer"] = content
        return StepPlan(node_id="parse", next_node="maybe_review")

    def act_node(run: RunState, ctx) -> StepPlan:
        _, _, runtime_ns, temp, _ = ensure_codeact_vars(run)
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
                formatted_calls.append({"name": tc.get("name", ""), "arguments": tc.get("arguments", {}), "call_id": call_id})
            elif isinstance(tc, ToolCall):
                call_id = str(tc.call_id).strip() if tc.call_id is not None else ""
                if not call_id:
                    call_id = str(i)
                formatted_calls.append({"name": tc.name, "arguments": tc.arguments, "call_id": call_id})

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls, "allowed_tools": list(allow)},
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def execute_code_node(run: RunState, ctx) -> StepPlan:
        _, _, _, temp, _ = ensure_codeact_vars(run)
        code = temp.get("pending_code")
        if not isinstance(code, str) or not code.strip():
            return StepPlan(node_id="execute_code", next_node="reason")

        temp.pop("pending_code", None)
        emit("act", {"tool": "execute_python", "args": {"code": "(inline)", "timeout_s": 10.0}})

        return StepPlan(
            node_id="execute_code",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [
                        {
                            "name": "execute_python",
                            "arguments": {"code": code, "timeout_s": 10.0},
                            "call_id": "code",
                        }
                    ]
                },
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_codeact_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []

        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")
            # Prefer a tool-supplied human/LLM-friendly rendering when present.
            def _display(v: Any) -> str:
                if isinstance(v, dict):
                    rendered = v.get("rendered")
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered.strip()
                return "" if v is None else str(v)

            display = _display(output)
            if not success:
                # Preserve structured outputs for provenance, but show a clean string to the LLM/UI.
                display = _display(output) if isinstance(output, dict) else str(error or output)
            rendered = logic.format_observation(
                name=name,
                output=display,
                success=success,
            )
            # Observability: avoid truncating normal tool results in step events.
            # Keep a bounded preview for huge tool outputs to avoid bloating traces/ledgers.
            preview = rendered
            if len(preview) > 1000:
                preview = preview[:1000] + f"\n… (truncated, {len(rendered):,} chars total)"
            emit("observe", {"tool": name, "success": success, "result": preview})
            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={"name": name, "call_id": r.get("call_id"), "success": success},
                )
            )

        temp.pop("tool_results", None)
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_codeact_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)

        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="reason")

    def maybe_review_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, runtime_ns, _, _ = ensure_codeact_vars(run)

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
        context, scratchpad, runtime_ns, _, _ = ensure_codeact_vars(run)
        task = str(context.get("task", "") or "")
        plan = scratchpad.get("plan")
        plan_text = str(plan).strip() if isinstance(plan, str) and plan.strip() else "(no plan)"
        answer = str(run.vars.get("_temp", {}).get("final_answer") or "")

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
            "response_schema_name": "CodeActReview",
            "params": {"temperature": 0.2},
        }
        sys = _system_prompt(runtime_ns)
        if sys is not None:
            payload["system_prompt"] = sys

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
        _, _, runtime_ns, temp, _ = ensure_codeact_vars(run)
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

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_codeact_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer})

        # Prefer _limits.current_iteration, fall back to scratchpad
        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)

        # Persist the final answer into the conversation history so it becomes part of the
        # next run's seed context and shows up in /history.
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
        context, scratchpad, _, _, limits = ensure_codeact_vars(run)

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
        workflow_id="codeact_agent",
        entry_node="init",
        nodes={
            "init": init_node,
            "plan": plan_node,
            "plan_parse": plan_parse_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "execute_code": execute_code_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "maybe_review": maybe_review_node,
            "review": review_node,
            "review_parse": review_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )
