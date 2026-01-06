"""AbstractRuntime adapter for MemAct (memory-enhanced agent)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy

from ..logic.memact import MemActLogic


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


def ensure_memact_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

    if not isinstance(context.get("messages"), list):
        context["messages"] = []
    if not isinstance(runtime_ns.get("inbox"), list):
        runtime_ns["inbox"] = []

    iteration = scratchpad.get("iteration")
    if not isinstance(iteration, int):
        try:
            scratchpad["iteration"] = int(iteration or 0)
        except Exception:
            scratchpad["iteration"] = 0
    max_iterations = scratchpad.get("max_iterations")
    if not isinstance(max_iterations, int):
        try:
            scratchpad["max_iterations"] = int(max_iterations or 25)
        except Exception:
            scratchpad["max_iterations"] = 25
    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    used_tools = scratchpad.get("used_tools")
    if not isinstance(used_tools, bool):
        scratchpad["used_tools"] = bool(used_tools) if used_tools is not None else False

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def create_memact_workflow(
    *,
    logic: MemActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    workflow_id: str = "memact_agent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> WorkflowSpec:
    """Adapt MemActLogic to an AbstractRuntime workflow."""

    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)

    def _current_tool_defs() -> list[Any]:
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
            if not name or name in seen:
                continue
            if name not in current:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _effective_allowlist(runtime_ns: Dict[str, Any]) -> list[str]:
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

    def _system_prompt_override(runtime_ns: Dict[str, Any]) -> Optional[str]:
        raw = runtime_ns.get("system_prompt") if isinstance(runtime_ns, dict) else None
        if isinstance(raw, str) and raw.strip():
            return raw
        return None

    def _sanitize_llm_messages(messages: Any) -> List[Dict[str, str]]:
        if not isinstance(messages, list) or not messages:
            return []
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
            entry: Dict[str, str] = {"role": role, "content": content_str}
            if role == "tool":
                meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
                call_id = meta.get("call_id") if isinstance(meta, dict) else None
                if call_id is not None and str(call_id).strip():
                    entry["tool_call_id"] = str(call_id).strip()
            out.append(entry)
        return out

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_memact_vars(run)
        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        # Ensure MemAct Active Memory exists (seeded by agent.start when available).
        from abstractruntime.memory.active_memory import ensure_memact_memory

        ensure_memact_memory(run.vars)

        task = str(context.get("task", "") or "")
        context["task"] = task
        messages = context["messages"]
        if task and (not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != task):
            messages.append(_new_message(ctx, role="user", content=task))

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)
        runtime_ns.setdefault("inbox", [])

        emit("init", {"task": task})
        return StepPlan(node_id="init", next_node="reason")

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_memact_vars(run)

        iteration = int(limits.get("current_iteration", 0) or 0)
        max_iterations = int(limits.get("max_iterations", 25) or scratchpad.get("max_iterations", 25) or 25)
        if max_iterations < 1:
            max_iterations = 1

        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        scratchpad["iteration"] = iteration + 1
        limits["current_iteration"] = iteration + 1

        task = str(context.get("task", "") or "")
        messages_view = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        # Inbox is a small, host/agent-controlled injection channel.
        guidance = ""
        inbox = runtime_ns.get("inbox", [])
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
            vars=run.vars,
        )

        from abstractruntime.memory.active_memory import render_memact_system_prompt

        memory_prompt = render_memact_system_prompt(run.vars)
        base_sys = _system_prompt_override(runtime_ns) or req.system_prompt
        system_prompt = (memory_prompt + "\n\n" + str(base_sys or "")).strip()

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload: Dict[str, Any] = {"prompt": ""}
        payload["messages"] = _sanitize_llm_messages(messages_view)
        if tool_specs:
            payload["tools"] = list(tool_specs)
        if system_prompt:
            payload["system_prompt"] = system_prompt
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        params: Dict[str, Any] = {"temperature": 0.2 if tool_specs else 0.7}
        if req.max_tokens is not None:
            params["max_tokens"] = req.max_tokens
        payload["params"] = params

        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.llm_response"),
            next_node="parse",
        )

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_memact_vars(run)
        response = temp.get("llm_response", {})
        content, tool_calls = logic.parse_response(response)
        temp.pop("llm_response", None)

        emit(
            "parse",
            {
                "has_tool_calls": bool(tool_calls),
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id} for tc in tool_calls],
            },
        )

        if tool_calls:
            # Keep any user-facing prelude content (optional) in history.
            clean = str(content or "").strip()
            if clean:
                context["messages"].append(_new_message(ctx, role="assistant", content=clean))
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            return StepPlan(node_id="parse", next_node="act")

        # Tool-free: draft answer becomes input to the envelope finalization call.
        temp["draft_answer"] = str(content or "").strip()
        scratchpad["tool_retry_count"] = 0
        return StepPlan(node_id="parse", next_node="finalize")

    def act_node(run: RunState, ctx) -> StepPlan:
        # Queue semantics: preserve ordering and avoid dropping calls when schema-only tools
        # (ask_user/memory/etc.) are interleaved with normal tools.
        context, _, runtime_ns, temp, _ = ensure_memact_vars(run)
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

        tool_queue: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_queue, start=1):
            if not isinstance(item, dict):
                continue
            d = dict(item)
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

        if _is_builtin(tool_queue[0]):
            tc = tool_queue[0]
            name = str(tc.get("name") or "").strip()
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}

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

            if temp.get("pending_tool_calls"):
                return StepPlan(node_id="act", next_node="act")
            return StepPlan(node_id="act", next_node="reason")

        batch: List[Dict[str, Any]] = []
        for tc in tool_queue:
            if _is_builtin(tc):
                break
            batch.append(tc)

        remaining = tool_queue[len(batch) :]
        temp["pending_tool_calls"] = list(remaining)

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
        context, scratchpad, _, temp, _ = ensure_memact_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []
        if results:
            scratchpad["used_tools"] = True

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
                display = _display(output) if isinstance(output, dict) else str(error or output)
            rendered = logic.format_observation(name=name, output=display, success=success)
            emit("observe", {"tool": name, "success": success})

            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={"name": name, "call_id": r.get("call_id"), "success": success},
                )
            )

        temp.pop("tool_results", None)
        pending = temp.get("pending_tool_calls", [])
        if isinstance(pending, list) and pending:
            return StepPlan(node_id="observe", next_node="act")
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_memact_vars(run)
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

    def finalize_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_memact_vars(run)
        _ = scratchpad

        task = str(context.get("task", "") or "")
        messages_view = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)
        payload_messages = _sanitize_llm_messages(messages_view)

        draft = str(temp.get("draft_answer") or "").strip()
        if draft:
            payload_messages = list(payload_messages) + [{"role": "assistant", "content": draft}]

        from abstractruntime.memory.active_memory import MEMACT_ENVELOPE_SCHEMA_V1, render_memact_system_prompt

        memory_prompt = render_memact_system_prompt(run.vars)
        base_sys = _system_prompt_override(runtime_ns) or ""

        finalize_rules = (
            "Finalize by returning a single JSON object that matches the required schema.\n"
            "Rules:\n"
            "- Put your normal user-facing answer in `content`.\n"
            "- For each memory module, decide what unitary statements to add/remove.\n"
            "- Do NOT include timestamps in added statements; the runtime will add timestamps.\n"
            "- HISTORY is append-only and must be experiential; do NOT include raw commands or tool-call syntax.\n"
            "- If you have no changes for a module, use empty lists.\n"
        )

        system_prompt = (memory_prompt + "\n\n" + str(base_sys or "")).strip()
        prompt = (
            "Finalize now.\n\n"
            f"User request:\n{task}\n\n"
            f"{finalize_rules}\n"
            "Return ONLY the JSON object.\n"
        ).strip()
        payload_messages = list(payload_messages) + [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "prompt": "",
            "messages": payload_messages,
            "system_prompt": system_prompt,
            "response_schema": MEMACT_ENVELOPE_SCHEMA_V1,
            "response_schema_name": "MemActEnvelopeV1",
            "params": {"temperature": 0.2},
        }

        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        emit("finalize_request", {"has_draft": bool(draft)})

        return StepPlan(
            node_id="finalize",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.finalize_llm_response"),
            next_node="finalize_parse",
        )

    def finalize_parse_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, _, temp, _ = ensure_memact_vars(run)
        resp = temp.get("finalize_llm_response", {})
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

        content = data.get("content")
        final_answer = str(content or "").strip()

        from abstractruntime.memory.active_memory import apply_memact_envelope

        apply_memact_envelope(run.vars, envelope=data)

        temp.pop("finalize_llm_response", None)
        temp.pop("draft_answer", None)
        temp["final_answer"] = final_answer
        scratchpad["used_tools"] = bool(scratchpad.get("used_tools"))

        emit("finalize", {"has_answer": bool(final_answer)})
        return StepPlan(node_id="finalize_parse", next_node="done")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_memact_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer})

        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)

        messages = context.get("messages")
        if isinstance(messages, list):
            last = messages[-1] if messages else None
            last_role = last.get("role") if isinstance(last, dict) else None
            last_content = last.get("content") if isinstance(last, dict) else None
            if last_role != "assistant" or str(last_content or "") != answer:
                messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer"}))

        return StepPlan(
            node_id="done",
            complete_output={"answer": answer, "iterations": iterations, "messages": list(context.get("messages") or [])},
        )

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, _, limits = ensure_memact_vars(run)
        max_iterations = int(limits.get("max_iterations", 0) or scratchpad.get("max_iterations", 25) or 25)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations})

        messages = list(context.get("messages") or [])
        last_content = messages[-1]["content"] if messages else "Max iterations reached"
        return StepPlan(
            node_id="max_iterations",
            complete_output={"answer": last_content, "iterations": max_iterations, "messages": messages},
        )

    return WorkflowSpec(
        workflow_id=str(workflow_id or "memact_agent"),
        entry_node="init",
        nodes={
            "init": init_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "finalize": finalize_node,
            "finalize_parse": finalize_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
        },
    )
