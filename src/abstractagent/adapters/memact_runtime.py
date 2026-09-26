"""AbstractRuntime adapter for MemAct (memory-enhanced agent)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy
from abstractruntime.turn_grounding import stamp_user_turn_grounding

from .generation_params import (
    DELEGATE_SUBSTRATE_KEYS,
    coerce_iterations,
    compose_prompt_slots,
    context_usage_warning,
    guidance_wrapper,
    normalize_thinking,
    prompt_cache_capture,
    resolve_max_iterations,
    runtime_llm_params,
    suppress_loop_tail,
)
from .media import extract_media_from_context
from .announced_calls import (
    classify_no_call_reply,
    is_not_an_answer,
    no_call_error_text,
    reprompt_text,
    resolve_checks,
    tool_calls_from_reasoning,
    verbatim_reply,
)
from .transcripts import (
    assistant_tool_calls_payload,
    ensure_tool_call_ids,
    extract_reasoning_text,
    parse_content_preview,
    place_loop_tail,
    sanitize_transcript_messages,
    synthetic_call_id,
)
from .loop_hooks import LoopHooks, undelivered_inbox_stats
from .tool_allowlist import note_pruned_grants
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



def _new_assistant_message_with_tool_calls(
    ctx: Any,
    *,
    content: str,
    tool_calls: List[Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assistant message preserving tool_calls metadata (0011 shared shape)."""
    msg = _new_message(ctx, role="assistant", content=content, metadata=metadata)
    tc_payload = assistant_tool_calls_payload(tool_calls)
    if tc_payload:
        msg["tool_calls"] = tc_payload
    return msg

def ensure_memact_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Captured BEFORE ensure_limits materializes defaults (0029 #6).
    _raw_limits = run.vars.get("_limits")
    _caller_set_budget = isinstance(_raw_limits, dict) and _raw_limits.get("max_iterations") is not None
    _raw_scratchpad = run.vars.get("scratchpad")
    _legacy_budget = "max_iterations" in run.vars or (
        isinstance(_raw_scratchpad, dict) and _raw_scratchpad.get("max_iterations") is not None
    )

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
    # Legacy flat-key migration parity (0029 #6 sharpened here: MemAct never
    # migrated the flat budget at all — a raw vars={"max_iterations": N}
    # was ignored entirely, not just out-prioritized).
    if "iteration" in run.vars and "iteration" not in scratchpad:
        scratchpad["iteration"] = run.vars.pop("iteration")
    if "max_iterations" in run.vars and "max_iterations" not in scratchpad:
        scratchpad["max_iterations"] = run.vars.pop("max_iterations")

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
            scratchpad["max_iterations"] = int(max_iterations or 20)
        except Exception:
            scratchpad["max_iterations"] = 20
    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    # 0029 #6: explicit legacy budget seeds _limits unless the caller set it.
    if _legacy_budget and not _caller_set_budget:
        limits["max_iterations"] = scratchpad["max_iterations"]

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
    hooks: Optional[LoopHooks] = None,
    workflow_id: str = "memact_agent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> WorkflowSpec:
    """Adapt MemActLogic to an AbstractRuntime workflow."""

    if hooks is not None and not hooks.agent:
        hooks.agent = str(workflow_id or "memact_agent")

    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)
        if hooks is not None:
            # First-class hooks (shared contract, see adapters/loop_hooks.py):
            # dispatch never raises; follow-ups re-enter the flat stream.
            for fu_step, fu_data in hooks.dispatch(step, data):
                if on_step:
                    on_step(fu_step, fu_data)

    def _fold_hook_steering(runtime_ns: Dict[str, Any]) -> None:
        if hooks is None or not isinstance(runtime_ns, dict):
            return
        for text in hooks.drain_pending_injections(hooks.current_run_id()):
            inbox = runtime_ns.get("inbox")
            if not isinstance(inbox, list):
                inbox = []
                runtime_ns["inbox"] = inbox
            inbox.append({"role": "system", "content": str(text or "")})

    def _with_run_context(node_fn):
        """Bind the hooks' run context around node execution (per-run queues:
        one workflow product serves many runs; see adapters/loop_hooks.py)."""
        if hooks is None:
            return node_fn

        def wrapper(run, ctx):
            hooks.push_run(str(getattr(run, "run_id", "") or ""))
            try:
                return node_fn(run, ctx)
            finally:
                hooks.pop_run()

        wrapper.__name__ = getattr(node_fn, "__name__", "node")
        return wrapper

    def _discard_hook_steering_at_terminal() -> None:
        if hooks is None:
            return
        dropped = hooks.discard_run(hooks.current_run_id())
        if dropped:
            emit("hook_steer_discarded", {"count": dropped})

    def _note_undelivered_inbox_at_terminal(runtime_ns: Dict[str, Any]) -> None:
        """Conclude-phase drain honesty (0026): durable-inbox guidance that
        landed after the loop's last drain (e.g. during the finalize call)
        can no longer influence this run — emit loudly instead of completing
        over it silently. Entries stay in the durable vars."""
        stats = undelivered_inbox_stats(runtime_ns)
        if stats:
            emit("inbox_undelivered", stats)

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
            raw = runtime_ns.get("allowed_tools")
            normalized = _normalize_allowlist(raw)
            # Works-or-loud: names the grant lost are recorded durably, never
            # silently dropped (shared note — one source across adapters).
            payload = note_pruned_grants(runtime_ns, raw, normalized)
            if payload is not None:
                emit("allowlist_pruned", payload)
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

    def _compose_system_prompt_extra(runtime_ns: Dict[str, Any], *, base: str) -> str:
        """Append the SHARED named slots at fixed post-base positions (fable5
        A-F1 2026-07-13): this adapter wrote the sub-agent directive into
        delegated children's system_prompt_extra but never read the key — the
        directive was silently dropped. Slot order/headers live in ONE place
        (`generation_params.PROMPT_SLOTS`, all three adapters); values must be
        byte-stable for the run (cache contract; docs/skills-attachment.md)."""
        return compose_prompt_slots(str(base or ""), runtime_ns)

    def _system_prompt_override(runtime_ns: Dict[str, Any]) -> Optional[str]:
        raw = runtime_ns.get("system_prompt") if isinstance(runtime_ns, dict) else None
        if isinstance(raw, str) and raw.strip():
            return raw
        return None

    def _sanitize_llm_messages(messages: Any) -> List[Dict[str, Any]]:
        """Shared extraction (backlog 0011): assistant `tool_calls` metadata
        survives to the wire and orphaned tool messages are repaired — multi-
        iteration tool use no longer 400s on strict providers. MemAct bounds
        no message sizes at this boundary (no truncate hook)."""
        return sanitize_transcript_messages(messages)

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
        return StepPlan(node_id="init", next_node="compose")

    def compose_node(run: RunState, ctx) -> StepPlan:
        """Optional, runtime-owned memory composition step (v0).

        When enabled via `_runtime.memact_composer.enabled`, this node queries the
        temporal KG (`MEMORY_KG_QUERY`) using the latest user message as stimulus
        and maps the selected packets into MemAct CURRENT CONTEXT entries.

        This runs *before* `reason` so it does not consume agent iterations.
        """
        context, _, runtime_ns, temp, _ = ensure_memact_vars(run)

        cfg_raw = runtime_ns.get("memact_composer") if isinstance(runtime_ns, dict) else None
        cfg = cfg_raw if isinstance(cfg_raw, dict) else {}
        enabled = bool(cfg.get("enabled"))
        if not enabled:
            return StepPlan(node_id="compose", next_node="reason")

        # Derive stimulus from the latest user message, falling back to `context.task`.
        stimulus = ""
        stimulus_message_id: Optional[str] = None
        messages = context.get("messages")
        if isinstance(messages, list):
            for m in reversed(messages):
                if not isinstance(m, dict):
                    continue
                if str(m.get("role") or "") != "user":
                    continue
                raw = m.get("content")
                if raw is None:
                    continue
                text = str(raw).strip()
                if not text:
                    continue
                stimulus = text
                meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
                mid = meta.get("message_id") if isinstance(meta, dict) else None
                if isinstance(mid, str) and mid.strip():
                    stimulus_message_id = mid.strip()
                break
        if not stimulus:
            stimulus = str(context.get("task", "") or "").strip()

        if not stimulus:
            return StepPlan(node_id="compose", next_node="reason")

        recall_level = str(cfg.get("recall_level") or "urgent").strip().lower() or "urgent"
        scope = str(cfg.get("scope") or "session").strip().lower() or "session"
        marker = str(cfg.get("marker") or "KG:").strip() or "KG:"
        max_items = cfg.get("max_items")
        max_items_int: Optional[int] = None
        if max_items is not None and not isinstance(max_items, bool):
            try:
                mi = int(float(max_items))
            except Exception:
                mi = None
            if isinstance(mi, int) and mi > 0:
                max_items_int = mi

        # Build a stable "composition key" so we don't re-query on tool iterations.
        compose_key_parts = [
            stimulus_message_id or stimulus,
            recall_level,
            scope,
            str(cfg.get("limit") or ""),
            str(cfg.get("min_score") or ""),
            str(cfg.get("max_input_tokens") or cfg.get("max_in_tokens") or ""),
        ]
        compose_key = "|".join([p for p in compose_key_parts if p is not None])

        bucket_raw = temp.get("memact_composer")
        bucket: Dict[str, Any] = bucket_raw if isinstance(bucket_raw, dict) else {}
        temp["memact_composer"] = bucket

        # If we already applied the composer for this key and have no pending results, skip.
        if bucket.get("applied_key") == compose_key and "kg_result" not in bucket:
            return StepPlan(node_id="compose", next_node="reason")

        kg_result = bucket.get("kg_result")
        # Present-but-NON-DICT = a FAILED compose, never a missing one (0029
        # #12): a custom host handler returning a string/list used to fall
        # through to "schedule KG query", and idempotency replayed the
        # identical effect instantly — compose spun forever, consuming no
        # iteration budget. Fail the compose loudly and continue to reason.
        if "kg_result" in bucket and not isinstance(kg_result, dict):
            bucket.pop("kg_result", None)
            bucket["applied_key"] = compose_key
            emit(
                "compose",
                {
                    "ok": False,
                    "stimulus": stimulus,
                    "recall_level": recall_level,
                    "scope": scope,
                    "error": f"#FALLBACK KG handler returned {type(kg_result).__name__}, expected dict — composition skipped",
                },
            )
            return StepPlan(node_id="compose", next_node="reason")
        if isinstance(kg_result, dict):
            try:
                from abstractruntime.memory.memact_composer import compose_memact_current_context_from_kg_result

                out = compose_memact_current_context_from_kg_result(
                    run.vars,
                    kg_result=kg_result,
                    stimulus=stimulus,
                    marker=marker,
                    max_items=max_items_int,
                )
            except Exception as e:
                out = {"ok": False, "error": str(e), "delta": {}, "trace": {}}

            bucket.pop("kg_result", None)
            bucket["applied_key"] = compose_key
            bucket["last_stimulus"] = stimulus

            # Persist a small trace for UI/debuggers (bounded list).
            try:
                from abstractruntime.memory.active_memory import ensure_memact_memory

                mem = ensure_memact_memory(run.vars)
                traces = mem.get("composer_traces")
                if not isinstance(traces, list):
                    traces = []
                    mem["composer_traces"] = traces

                timestamp: Optional[str] = None
                now_iso = getattr(ctx, "now_iso", None)
                if callable(now_iso):
                    timestamp = str(now_iso())
                if not timestamp:
                    from datetime import datetime, timezone

                    timestamp = datetime.now(timezone.utc).isoformat()

                trace_entry = {
                    "at": timestamp,
                    "compose_key": compose_key,
                    "ok": bool(out.get("ok")),
                    "trace": out.get("trace"),
                }
                traces.insert(0, trace_entry)
                del traces[25:]
            except Exception:
                pass

            emit("compose", {"ok": bool(out.get("ok")), "stimulus": stimulus, "recall_level": recall_level, "scope": scope})
            return StepPlan(node_id="compose", next_node="reason")

        # No result yet: schedule KG query.
        payload: Dict[str, Any] = {
            "query_text": stimulus,
            "recall_level": recall_level,
            "scope": scope,
        }
        for src_key, dst_key in (
            ("limit", "limit"),
            ("min_score", "min_score"),
            ("max_input_tokens", "max_input_tokens"),
            ("max_in_tokens", "max_input_tokens"),
            ("model", "model"),
        ):
            if src_key in cfg:
                payload[dst_key] = cfg.get(src_key)

        # Default packing model: re-use the configured LLM model when available.
        if "model" not in payload:
            model_name = runtime_ns.get("model")
            if isinstance(model_name, str) and model_name.strip():
                payload["model"] = model_name.strip()

        # Store key so we can attribute the result even if the stimulus changes later.
        bucket["pending_key"] = compose_key

        emit("compose_query", {"stimulus": stimulus, "recall_level": recall_level, "scope": scope})
        return StepPlan(
            node_id="compose",
            effect=Effect(type=EffectType.MEMORY_KG_QUERY, payload=payload, result_key="_temp.memact_composer.kg_result"),
            next_node="compose",
        )

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_memact_vars(run)

        iteration = int(limits.get("current_iteration", 0) or 0)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1

        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        scratchpad["iteration"] = iteration + 1
        limits["current_iteration"] = iteration + 1

        task = str(context.get("task", "") or "")

        # Inbox is a small, host/agent-controlled injection channel. Drained guidance joins the
        # durable transcript as a user interjection (maintainer ruling 2026-07-09, same as the
        # ReAct adapter): the previous rendering folded it into the system prompt for ONE call,
        # which both mutated the cached prefix and was forgotten on the next cycle — a final
        # answer written later re-anchored on the original task and dropped the correction.
        _fold_hook_steering(runtime_ns)
        guidance = ""
        inbox = runtime_ns.get("inbox", [])
        if isinstance(inbox, list) and inbox:
            inbox_messages = [str(m.get("content", "") or "") for m in inbox if isinstance(m, dict)]
            guidance = "\n".join([m for m in inbox_messages if m])
            runtime_ns["inbox"] = []
        if guidance:
            context["messages"].append(
                _new_message(
                    ctx,
                    role="user",
                    # Lane-honest wrapper (c2792/c2796/c2798): see guidance_wrapper.
                    content=f"{guidance_wrapper(runtime_ns)}\n{guidance}",
                    metadata={"kind": "operator_guidance"},
                )
            )
            emit("inbox_drained", {"chars": len(guidance)})
        # STAMP THE TURN ONCE (mission A ported to MemAct by mission A3,
        # 2026-09-22). See the identical note in react_runtime/codeact_runtime:
        # the grounding envelope must be written into the DURABLE turn, once, so
        # the bytes a turn is SENT with are the bytes it is STORED with and turn
        # N's prompt stays an exact byte-prefix of turn N+1's. Idempotent, and it
        # refuses payload-synthesized carriers.
        stamp_user_turn_grounding(context.get("messages"))
        messages_view = ActiveContextPolicy.select_active_messages_for_llm_from_run(run)

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        tool_specs = [t.to_dict() for t in allowed_defs]
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        req = logic.build_request(
            task=task,
            messages=messages_view,
            # Guidance no longer rides the system prompt (cache stability + durability): it is
            # already in `messages_view` as a durable transcript message (see the drain above).
            guidance="",
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,
        )

        from abstractruntime.memory.active_memory import render_memact_system_prompt

        memory_prompt = render_memact_system_prompt(run.vars)
        base_sys = _system_prompt_override(runtime_ns) or req.system_prompt
        system_prompt = _compose_system_prompt_extra(
            runtime_ns, base=(memory_prompt + "\n\n" + str(base_sys or "")).strip()
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})
        ctx_warn = context_usage_warning(limits, scratchpad)
        if ctx_warn:
            emit("context_warning", ctx_warn)


        payload: Dict[str, Any] = {"prompt": ""}
        sanitized_messages = _sanitize_llm_messages(messages_view)
        if sanitized_messages:
            payload["messages"] = sanitized_messages
        else:
            # Ensure LLM_CALL contract is satisfied even when callers provide only `context.task`
            # and the active message view is empty.
            task_text = str(task or "").strip()
            if task_text:
                payload["prompt"] = task_text
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media
        if tool_specs:
            payload["tools"] = list(tool_specs)
        if system_prompt:
            payload["system_prompt"] = system_prompt

        # Volatile loop position rides a trailing ephemeral message, never the
        # system prompt (0212 propagated, fable5 2026-07-13 — the head counter
        # busted the prefix cache every cycle). Adjacency guard mirrors ReAct.
        # `_runtime.suppress_loop_tail` (c2447): loop tails are task-agent
        # chrome — entity-lane hosts suppress the tail (ReAct/CodeAct parity).
        # Mission A3: one placement rule for all three loops
        # (`transcripts.place_loop_tail`). The position line is chrome: dropped
        # in the chat shape (it used to merge INTO the user's durable message —
        # mission A's bug 1b), durable and marked in the tool-loop shape so
        # iteration N's prompt stays an exact prefix of N+1's.
        chrome_parts = [] if suppress_loop_tail(runtime_ns) else [f"[loop] iteration {int(iteration + 1)} of {int(max_iterations)}."]
        if isinstance(payload.get("messages"), list) and chrome_parts:
            payload["messages"] = place_loop_tail(
                durable_messages=context.get("messages"),
                payload_messages=payload["messages"],
                chrome_parts=chrome_parts,
                actionable_parts=[],
                new_message=lambda **kw: _new_message(ctx, **kw),
            )

        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        params: Dict[str, Any] = {"temperature": 0.2 if tool_specs else 0.7}
        if req.max_tokens is not None:
            params["max_tokens"] = req.max_tokens
        payload["params"] = runtime_llm_params(runtime_ns, extra=params)
        # Ledger record of a re-prompt (mission AGX; twin of react_runtime).
        _reprompt = run.vars.get("_temp", {}).get("reprompt") if isinstance(run.vars.get("_temp"), dict) else None
        if isinstance(_reprompt, dict) and _reprompt.get("reason"):
            payload["_runtime_observability"] = {
                "reprompted": str(_reprompt.get("reason")),
                "reprompt_detail": str(_reprompt.get("detail") or ""),
                "reprompted_cycle": _reprompt.get("cycle"),
            }

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
        # One re-prompt per step (mission AGX): popped once, here.
        prior_reprompt = temp.pop("reprompt", None)
        if not isinstance(prior_reprompt, dict):
            prior_reprompt = None

        # COMMON CORE parse payload (0028 contract wave, 2026-07-14): every
        # loop guarantees has_tool_calls + tool_calls + content_preview.
        parse_payload: Dict[str, Any] = {
            "has_tool_calls": bool(tool_calls),
            "tool_calls": [{"name": tc.name, "arguments": (dict(tc.arguments) if isinstance(tc.arguments, dict) else (list(tc.arguments) if isinstance(tc.arguments, list) else tc.arguments)), "call_id": tc.call_id} for tc in tool_calls],
            "content_preview": parse_content_preview(content),
            # Reasoning parity (reasoning-first-citizen plan, agent section):
            # the separated thinking channel was surfaced by ReAct only —
            # additive on the common core, same shared reader.
            "reasoning": extract_reasoning_text(response),
        }
        # Additive cache observability (0030 residue, 2026-07-15).
        cache_struct = prompt_cache_capture(response)
        if cache_struct is not None:
            parse_payload["prompt_cache"] = cache_struct
        emit("parse", parse_payload)

        if tool_calls:
            # Durable tool_calls preservation (0011): the assistant turn that
            # PROPOSED the batch must announce the ids its tool results answer
            # — content-only appends orphaned every tool message on strict
            # providers (native OpenAI 400s the request at iteration 2).
            # Mission A3: stamp the fallback ids HERE, once, so the announcement
            # and the batch `act_node` queues share one id namespace (they used to
            # mint `call_{i+1}` and `str(idx)` independently, which orphaned every
            # result into an `[unpaired tool result]` user carrier).
            ensure_tool_call_ids(tool_calls)
            context["messages"].append(
                _new_assistant_message_with_tool_calls(
                    ctx,
                    content=str(content or "").strip(),
                    tool_calls=tool_calls,
                    metadata={"kind": "tool_calls"},
                )
            )
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            # tool_proposed on all three loops (0026 follow-up, 2026-07-15):
            # the canonical commit signal, same raw step + payload as ReAct.
            _ptc: Dict[str, Any] = {"count": len(tool_calls)}
            _from_reasoning = tool_calls_from_reasoning(response)
            if _from_reasoning:
                _ptc["from_reasoning"] = _from_reasoning
            if prior_reprompt is not None:
                _ptc["after_reprompt"] = str(prior_reprompt.get("reason") or "")
            emit("parse_tool_calls", _ptc)
            return StepPlan(node_id="parse", next_node="act")

        # A reply with NO tool call that is not an answer (mission AGX; twin
        # of react_runtime.parse_node, see its comment and announced_calls):
        # an announcement ("Let me verify ...") or tool-call markup that could
        # not run. Re-prompt ONCE with the reply verbatim; a second failure
        # ends the step with a visible error instead of publishing either.
        _check_announced, _check_unrunnable = resolve_checks(runtime_ns, suppressed=suppress_loop_tail(runtime_ns))
        _specs_now = runtime_ns.get("tool_specs") if isinstance(runtime_ns, dict) else None
        _tools_offered = isinstance(_specs_now, list) and bool(_specs_now)
        no_call_reason, no_call_detail = classify_no_call_reply(
            response,
            content,
            tools_offered=_tools_offered,
            check_announcement=_check_announced,
            check_unrunnable=_check_unrunnable,
        )
        if no_call_reason:
            _cycle_i = int(scratchpad.get("iteration", 0) or 0)
            raw_reply = verbatim_reply(response, fallback=str(content or ""))
            temp["pending_tool_calls"] = []
            if prior_reprompt is not None:
                error_text = no_call_error_text(no_call_reason, no_call_detail)
                context["messages"].append(
                    _new_message(
                        ctx,
                        role="assistant",
                        content=raw_reply,
                        metadata={"kind": "reprompt_failed_reply", "cycle": _cycle_i, "reason": no_call_reason},
                    )
                )
                scratchpad["no_tool_call_stop"] = {
                    "reason": no_call_reason,
                    "detail": no_call_detail,
                    "cycle": _cycle_i,
                    "error": error_text,
                }
                emit(
                    "parse_reprompt_failed",
                    {"cycle": _cycle_i, "reason": no_call_reason, "detail": no_call_detail, "error": error_text},
                )
                context["messages"].append(
                    _new_message(ctx, role="assistant", content=error_text, metadata={"kind": "error"})
                )
                temp["final_answer"] = error_text
                return StepPlan(node_id="parse", next_node="done")
            context["messages"].append(
                _new_message(
                    ctx,
                    role="assistant",
                    content=raw_reply,
                    metadata={"kind": "reprompted_reply", "cycle": _cycle_i, "reason": no_call_reason},
                )
            )
            context["messages"].append(
                _new_message(
                    ctx,
                    role="user",
                    content=reprompt_text(
                        no_call_reason, tools_offered=_tools_offered, suppressed=suppress_loop_tail(runtime_ns)
                    ),
                    metadata={"kind": "reprompt", "cycle": _cycle_i, "reason": no_call_reason},
                )
            )
            temp["reprompt"] = {"reason": no_call_reason, "detail": no_call_detail, "cycle": _cycle_i}
            emit("parse_reprompt", {"cycle": _cycle_i, "reason": no_call_reason, "detail": no_call_detail})
            return StepPlan(node_id="parse", next_node="reason")

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
            return StepPlan(node_id="act", next_node="compose")

        allow = _effective_allowlist(runtime_ns)
        builtin_effect_tools = {
            "ask_user",
            "recall_memory",
            "inspect_vars",
            "remember",
            "remember_note",
            "compact_memory",
            "delegate_agent",
        }

        tool_queue: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_queue, start=1):
            if not isinstance(item, dict):
                continue
            d = dict(item)
            call_id = str(d.get("call_id") or "").strip()
            if not call_id:
                # Mission A3: one shared fallback formula (see
                # `transcripts.synthetic_call_id`). `idx` is 1-based here, the
                # formula is 0-based. `str(idx)` disagreed with the durable
                # announcement and orphaned every result into a user carrier.
                d["call_id"] = synthetic_call_id(idx - 1)
            tool_queue.append(d)

        if not tool_queue:
            temp["pending_tool_calls"] = []
            return StepPlan(node_id="act", next_node="compose")

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

            if name == "delegate_agent":
                delegated_task = str(args.get("task") or "").strip()
                delegated_context = str(args.get("context") or "").strip()

                tools_raw = args.get("tools")
                if tools_raw is None:
                    # Inherit the current allowlist, but avoid recursive delegation and avoid waiting on ask_user
                    # unless explicitly enabled.
                    child_allow = [t for t in allow if t not in {"delegate_agent", "ask_user"}]
                else:
                    # Grant containment (tool-tiers adversary P0, 2026-07-22; see
                    # react_runtime's delegate branch): the model-controlled
                    # `tools` arg normalized against the FULL registry — a child
                    # could hold tools the parent was never granted. Child
                    # exposure is a SUBSET of the parent's, always.
                    parent_allow = set(allow)
                    requested = _normalize_allowlist(tools_raw)
                    child_allow = [t for t in requested if t in parent_allow]
                    if requested and not child_allow:
                        temp["tool_results"] = {
                            "results": [
                                {
                                    "call_id": str(tc.get("call_id") or ""),
                                    "name": "delegate_agent",
                                    "success": False,
                                    "output": None,
                                    "error": (
                                        "delegate_agent tools must be a subset of the parent's allowed tools; "
                                        f"none of {sorted(set(requested))} are granted to this run"
                                    ),
                                }
                            ]
                        }
                        return StepPlan(node_id="act", next_node="observe")

                if not delegated_task:
                    temp["tool_results"] = {
                        "results": [
                            {
                                "call_id": str(tc.get("call_id") or ""),
                                "name": "delegate_agent",
                                "success": False,
                                "output": None,
                                "error": "delegate_agent requires a non-empty task",
                            }
                        ]
                    }
                    return StepPlan(node_id="act", next_node="observe")

                combined_task = delegated_task
                if delegated_context:
                    combined_task = f"{delegated_task}\n\nContext:\n{delegated_context}"

                # Child iteration budget (backlog 0012 / C3 — ReAct's ruled resolution
                # applied to this sibling; the shared DELEGATE_AGENT_TOOL schema already
                # documents it): an explicit `max_iterations` tool arg wins; otherwise
                # the child inherits the PARENT's budget with the ruled 20 as floor.
                # The old hardcoded 10 made the shared schema lie for this loop.
                parent_iterations = resolve_max_iterations(
                    run.vars.get("_limits") if isinstance(run.vars.get("_limits"), dict) else {},
                    run.vars.get("scratchpad") if isinstance(run.vars.get("scratchpad"), dict) else {},
                )
                # Arg-coercion tolerance (fable5 P1 2026-07-13): string-float
                # budgets ("8.5") raised in int() and silently widened.
                raw_child_iters = args.get("max_iterations")
                child_val = coerce_iterations(raw_child_iters) if raw_child_iters is not None else None
                if raw_child_iters is not None and child_val is None:
                    emit(
                        "delegate_agent_budget_fallback",
                        {"raw": str(raw_child_iters), "warning": "#FALLBACK unparseable max_iterations; inheriting parent"},
                    )
                if isinstance(child_val, int):
                    # Explicit values are honored even when falsy (0029 #15,
                    # resolver-contract parity): explicit <=0 clamps to the
                    # loop floor of 1 — it must never silently WIDEN to the
                    # inheritance default.
                    child_iterations = child_val if child_val >= 1 else 1
                else:
                    # Absent or unparseable: inherit the parent's budget with
                    # the ruled 20 as the floor guard.
                    child_iterations = max(int(parent_iterations or 0), 20)

                sub_vars: Dict[str, Any] = {
                    "context": {"task": combined_task, "messages": []},
                    "_runtime": {
                        "allowed_tools": list(child_allow),
                        "system_prompt_extra": (
                            "You are a delegated sub-agent.\n"
                            "- Focus ONLY on the delegated task.\n"
                            "- Use ONLY the allowed tools when needed.\n"
                            "- Do not ask the user questions; if blocked, state assumptions and proceed.\n"
                            "- Return a concise result suitable for the parent agent to act on.\n"
                        ),
                    },
                    "_limits": {"max_iterations": child_iterations},
                }
                # Substrate + sampling inheritance (fable5 A-F7/B-F6 2026-07-13;
                # see react_runtime's delegate branch): the child inherits the
                # parent's effective provider/model/temperature/seed explicitly.
                for _k in (
                    "provider",
                    "model",
                    "temperature",
                    "seed",
                    # Same per-run-behavior class (wave adversaries 2026-07-13):
                    # a parent with thinking off / a capped output budget /
                    # example-free tool prompts should not delegate a child
                    # that silently reverts to provider defaults.
                    "thinking",
                    "speculation",
                    # Live token streaming (2026-09-26): a delegated child
                    # streams its LLM calls to the same live view as the
                    # parent; False (explicit off) is inherited too.
                    "stream",
                    "max_output_tokens",
                    "tool_prompt_examples",
                    # Approval policy inherits monotonically (tool-tiers adversary
                    # P0, 2026-07-22; see react_runtime's delegate branch).
                    "tool_policy",
                ):
                    _v = runtime_ns.get(_k)
                    if _v is not None:
                        sub_vars["_runtime"][_k] = deepcopy(_v) if _k == "speculation" else _v

                # Host-gated substrate palette (0030 promoted 2026-07-13): the
                # `substrate` arg names a profile the HOST granted via
                # `_runtime.delegate_substrates` = {name: {provider, model}}.
                # Unknown/absent names fail as a TOOL error (the parent decides
                # what to do; never a run failure) and raw provider/model
                # strings are structurally impossible here — the model choosing
                # its own substrate would be self-escalation. The palette does
                # NOT propagate to grandchildren: each level needs its own grant.
                substrate_name = str(args.get("substrate") or "").strip()
                if substrate_name:
                    palette = runtime_ns.get("delegate_substrates")
                    profile = palette.get(substrate_name) if isinstance(palette, dict) else None
                    prof_provider = str(profile.get("provider") or "").strip() if isinstance(profile, dict) else ""
                    prof_model = str(profile.get("model") or "").strip() if isinstance(profile, dict) else ""
                    if not (prof_provider and prof_model):
                        available = sorted(str(k) for k in palette.keys()) if isinstance(palette, dict) else []
                        # Malformed ≠ unknown (wave adversary P2: "Unknown
                        # substrate 'x'. Available: x." declared a name unknown
                        # and available in one sentence).
                        if profile is not None:
                            err = (
                                f"Delegate substrate '{substrate_name}' is granted but malformed — "
                                "a profile needs non-empty 'provider' and 'model'. Ask the host to fix the grant."
                            )
                        elif available:
                            err = f"Unknown delegate substrate '{substrate_name}'. Available: " + ", ".join(available) + "."
                        else:
                            err = f"Unknown delegate substrate '{substrate_name}'. No substrate palette was granted for this run."
                        temp["tool_results"] = {
                            "results": [
                                {
                                    "call_id": str(tc.get("call_id") or ""),
                                    "name": "delegate_agent",
                                    "success": False,
                                    "output": None,
                                    "error": err,
                                }
                            ]
                        }
                        return StepPlan(node_id="act", next_node="observe")
                    # Version-skew loudness (wave adversary P1): a grant carrying
                    # keys this build does not understand is silently half-applied
                    # otherwise — warn, never drop silently.
                    unknown_keys = sorted(set(profile.keys()) - DELEGATE_SUBSTRATE_KEYS) if isinstance(profile, dict) else []
                    if unknown_keys:
                        emit(
                            "delegate_agent_substrate_skew",
                            {
                                "substrate": substrate_name,
                                "ignored_keys": unknown_keys,
                                "warning": "#FALLBACK unrecognized substrate profile keys ignored (version skew?)",
                            },
                        )
                    sub_vars["_runtime"]["provider"] = prof_provider
                    sub_vars["_runtime"]["model"] = prof_model
                    # Reasoning member of the selection triple (reasoning-
                    # first-citizen plan; see react_runtime's delegate branch):
                    # declared-and-valid wins; present-but-invalid warns and
                    # keeps the inherited value (deny-safe).
                    substrate_emit = {"substrate": substrate_name, "provider": prof_provider, "model": prof_model}
                    if isinstance(profile, dict) and profile.get("thinking") is not None:
                        prof_thinking = normalize_thinking(profile.get("thinking"))
                        if prof_thinking is not None:
                            sub_vars["_runtime"]["thinking"] = prof_thinking
                            substrate_emit["thinking"] = prof_thinking
                        else:
                            emit(
                                "delegate_agent_substrate_skew",
                                {
                                    "substrate": substrate_name,
                                    "ignored_keys": ["thinking"],
                                    "warning": "#FALLBACK substrate thinking value not recognized; child keeps the inherited value",
                                },
                            )
                    if isinstance(profile, dict) and profile.get("speculation") is not None:
                        sub_vars["_runtime"]["speculation"] = deepcopy(profile["speculation"])
                        substrate_emit["speculation"] = deepcopy(profile["speculation"])
                    emit("delegate_agent_substrate", substrate_emit)

                payload = {
                    "workflow_id": str(getattr(run, "workflow_id", "") or "memact_agent"),
                    "vars": sub_vars,
                    "async": False,
                    "include_traces": False,
                    # Tool-mode wrapper so the parent receives a normal tool observation (no run failure on child failure).
                    "wrap_as_tool_result": True,
                    "tool_name": "delegate_agent",
                    "call_id": str(tc.get("call_id") or ""),
                }
                emit("delegate_agent", {"tools": list(child_allow), "call_id": payload.get("call_id")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.START_SUBWORKFLOW, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
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
            return StepPlan(node_id="act", next_node="compose")

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

        # Idempotency discriminator (fable5 P0 2026-07-13; see react_runtime's
        # act_node): a later byte-identical batch replayed the stale ledger
        # result. Persisted counter = crash-safe dedup + fresh key per issuance.
        scratchpad_ns = run.vars.get("scratchpad") if isinstance(run.vars.get("scratchpad"), dict) else {}
        act_seq = int(scratchpad_ns.get("act_seq") or 0) + 1
        scratchpad_ns["act_seq"] = act_seq

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls, "allowed_tools": list(allow), "act_seq": act_seq},
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
            emit("observe", {"tool": name, "success": success, "result": rendered, "call_id": str(r.get("call_id") or "")})

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
        return StepPlan(node_id="observe", next_node="compose")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_memact_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)

        # Per-turn report state resets at the turn boundary (wave-E P2-4
        # symmetry, 2026-07-14): finalize_skipped is per-turn report state.
        scratchpad.pop("finalize_skipped", None)

        if temp.get("pending_tool_calls"):
            return StepPlan(node_id="handle_user_response", next_node="act")
        return StepPlan(node_id="handle_user_response", next_node="compose")

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

        system_prompt = _compose_system_prompt_extra(
            runtime_ns, base=(memory_prompt + "\n\n" + str(base_sys or "")).strip()
        )
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
            "params": runtime_llm_params(runtime_ns, extra={"temperature": 0.2}),
            # Finalize-failure containment (fable5 P1 2026-07-13 — the c1128
            # incident class, unpatched in MemAct's MANDATORY finalize path): a
            # terminally failed structured finalize call must never kill a run
            # whose _temp.draft_answer already holds the model's real answer.
            # finalize_parse falls back to the draft on the absorbed record.
            "_absorb_failure": True,
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

        draft = str(temp.get("draft_answer") or "").strip()

        absorbed = resp.get("absorbed_failure")
        if absorbed is not None:
            # The finalize call failed terminally (runtime `_absorb_failure`
            # shape). A finalize aid must never destroy a held answer: the
            # draft the model already produced IS the answer; the envelope
            # application is skipped (memory updates lost for this turn — the
            # loud marker says so).
            reason = str(absorbed or "").strip() or "unknown error"
            skipped = scratchpad.get("finalize_skipped")
            if not isinstance(skipped, list):
                skipped = []
                scratchpad["finalize_skipped"] = skipped
            skipped.append({"reason": reason, "warning": "#FALLBACK"})
            emit("finalize_skipped", {"reason": reason, "warning": "#FALLBACK", "used_draft_answer": bool(draft)})
            temp.pop("finalize_llm_response", None)
            temp.pop("draft_answer", None)
            temp["final_answer"] = draft or "No answer provided"
            return StepPlan(node_id="finalize_parse", next_node="done")

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
        if not final_answer and draft:
            # Non-JSON / empty-content finalize output: the draft is the real
            # answer — never discard it for "No answer provided" (fable5 P1).
            emit("finalize_used_draft", {"warning": "#FALLBACK finalize envelope unparseable; draft answer kept"})
            final_answer = draft

        from abstractruntime.memory.active_memory import apply_memact_envelope

        apply_memact_envelope(run.vars, envelope=data)

        temp.pop("finalize_llm_response", None)
        temp.pop("draft_answer", None)
        temp["final_answer"] = final_answer
        scratchpad["used_tools"] = bool(scratchpad.get("used_tools"))

        emit("finalize", {"has_answer": bool(final_answer)})
        return StepPlan(node_id="finalize_parse", next_node="done")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_memact_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer, "outcome": "final_answer"})

        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)

        messages = context.get("messages")
        if isinstance(messages, list):
            last = messages[-1] if messages else None
            last_role = last.get("role") if isinstance(last, dict) else None
            last_content = last.get("content") if isinstance(last, dict) else None
            if last_role != "assistant" or str(last_content or "") != answer:
                messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer"}))

        _discard_hook_steering_at_terminal()
        _note_undelivered_inbox_at_terminal(runtime_ns)
        return StepPlan(
            node_id="done",
            complete_output={
                "answer": answer,
                "iterations": iterations,
                "messages": list(context.get("messages") or []),
                # Canonical turn_end vocabulary (final_answer | iteration_budget).
                "outcome": "final_answer",
                # Loudness parity (fable5 2026-07-13): a skipped finalize must
                # be visible in the run OUTPUT, not only the emit lane.
                "finalize_skipped": bool(scratchpad.get("finalize_skipped")),
                **(
                    {"no_tool_call_stop": dict(scratchpad["no_tool_call_stop"])}
                    if isinstance(scratchpad.get("no_tool_call_stop"), dict)
                    else {}
                ),
            },
        )

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_memact_vars(run)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations, "outcome": "iteration_budget"})

        messages = list(context.get("messages") or [])
        # The answer is the agent's LAST WORDS, not the raw last message
        # (0029 #10; MemAct also prefers the held draft — the answer the
        # finalize path would have polished). `.get`, not [] (fable5
        # 2026-07-13): content-less messages raised here.
        temp_ns = run.vars.get("_temp") if isinstance(run.vars.get("_temp"), dict) else {}
        answer = str(temp_ns.get("draft_answer") or "").strip()
        if not answer:
            for msg in reversed(messages):
                # A re-prompted (failed) reply is never "last words" (mission AGX).
                if isinstance(msg, dict) and msg.get("role") == "assistant" and not is_not_an_answer(msg):
                    text = str(msg.get("content") or "")
                    if text.strip():
                        answer = text
                        break
        if not answer:
            answer = f"Reached the iteration budget ({max_iterations}) without producing an answer."
        last = messages[-1] if messages else None
        if not (isinstance(last, dict) and last.get("role") == "assistant" and str(last.get("content") or "") == answer):
            messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer", "budget_exhausted": True}))
            context["messages"] = messages
        _discard_hook_steering_at_terminal()
        _note_undelivered_inbox_at_terminal(runtime_ns)
        return StepPlan(
            node_id="max_iterations",
            complete_output={
                "answer": answer,
                "iterations": max_iterations,
                "messages": messages,
                "outcome": "iteration_budget",
                # Always-present at BOTH terminals (wave-F P3 shape parity).
                "finalize_skipped": bool(scratchpad.get("finalize_skipped")),
            },
        )

    return WorkflowSpec(
        workflow_id=str(workflow_id or "memact_agent"),
        entry_node="init",
        nodes={
            node_id: _with_run_context(node_fn)
            for node_id, node_fn in {
            "init": init_node,
            "compose": compose_node,
            "reason": reason_node,
            "parse": parse_node,
            "act": act_node,
            "observe": observe_node,
            "handle_user_response": handle_user_response_node,
            "finalize": finalize_node,
            "finalize_parse": finalize_parse_node,
            "done": done_node,
            "max_iterations": max_iterations_node,
            }.items()
        },
    )
