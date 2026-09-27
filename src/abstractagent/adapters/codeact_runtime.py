"""AbstractRuntime adapter for CodeAct agents."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall, ToolDefinition
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.memory.active_context import ActiveContextPolicy
from abstractruntime.turn_grounding import stamp_user_turn_grounding

from .generation_params import (
    DELEGATE_SUBSTRATE_KEYS,
    coerce_iterations,
    coerce_verifier_tool_arguments,
    compose_prompt_slots,
    context_usage_warning,
    executor_tool_names,
    guidance_wrapper,
    normalize_thinking,
    prompt_cache_capture,
    resolve_max_iterations,
    runtime_llm_params,
    suppress_loop_tail,
    tool_tags_map,
    verifier_execution_preference,
    verifier_response_schema,
)
from .media import extract_media_from_context
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
from .read_orchestration import (
    advance_last_successful_read_batch,
    detect_nearby_same_file_staircase,
    detect_redundant_same_file_full_reread,
)
from .tool_allowlist import note_pruned_grants
from ..logic.codeact import CodeActLogic


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


def _push_inbox(runtime_ns: Dict[str, Any], content: str) -> None:
    if not isinstance(runtime_ns, dict):
        return
    inbox = runtime_ns.get("inbox")
    if not isinstance(inbox, list):
        inbox = []
        runtime_ns["inbox"] = inbox
    inbox.append({"role": "system", "content": str(content or "")})


def ensure_codeact_vars(run: RunState) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place.

    Returns:
        Tuple of (context, scratchpad, runtime_ns, temp, limits) dicts.
    """
    # Captured BEFORE ensure_limits materializes defaults (0029 #6): an
    # explicit legacy/flat budget used to lose to the materialized
    # `_limits.max_iterations=20` at the resolver (limits wins).
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
        scratchpad["max_iterations"] = 20
    elif not isinstance(max_iterations, int):
        try:
            scratchpad["max_iterations"] = int(max_iterations)
        except (TypeError, ValueError):
            scratchpad["max_iterations"] = 20

    if scratchpad["max_iterations"] < 1:
        scratchpad["max_iterations"] = 1

    # 0029 #6: explicit legacy budget seeds _limits unless the caller set it.
    if _legacy_budget and not _caller_set_budget:
        limits["max_iterations"] = scratchpad["max_iterations"]

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
    hooks: Optional[LoopHooks] = None,
    workflow_id: str = "codeact_agent",
) -> WorkflowSpec:
    if hooks is not None and not hooks.agent:
        hooks.agent = "codeact_agent"

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
        landed after the loop's last drain can no longer influence this run —
        emit loudly instead of completing over it silently. Entries stay in
        the durable vars (the record shows what missed)."""
        stats = undelivered_inbox_stats(runtime_ns)
        if stats:
            emit("inbox_undelivered", stats)


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
            raw = runtime_ns.get("allowed_tools")
            normalized = _normalize_allowlist(raw)
            # Filter to currently known tools (dynamic), preserving order.
            current = _tool_by_name()
            filtered = [name for name in normalized if name in current]
            # Works-or-loud: names the grant lost are recorded durably, never
            # silently dropped (shared note — one source across adapters).
            payload = note_pruned_grants(runtime_ns, raw, filtered)
            if payload is not None:
                emit("allowlist_pruned", payload)
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

    def _compose_system_prompt(runtime_ns: Dict[str, Any], *, base: str) -> str:
        """Override-or-base + the SHARED named slots (fable5 A-F1 2026-07-13):
        this adapter WROTE system_prompt_extra into its delegated children but
        never READ it — the sub-agent directive was silently dropped on every
        CodeAct delegation. Slot order/headers live in ONE place
        (`generation_params.PROMPT_SLOTS`, all three adapters); values must be
        byte-stable for the run (cache contract; docs/skills-attachment.md)."""
        override = _system_prompt(runtime_ns)
        sys = override if override is not None else base
        return compose_prompt_slots(str(sys or ""), runtime_ns)

    def _sanitize_llm_messages(messages: Any, *, limits: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Convert runtime-owned message dicts into OpenAI-style {role, content, ...}.

        Shared extraction (backlog 0011): delegates to adapters/transcripts.py —
        assistant `tool_calls` metadata now survives to the wire and orphaned
        tool messages are repaired, so multi-iteration tool use no longer 400s
        on strict providers (native OpenAI's assistant-tool_calls-then-tool-
        messages contract). CodeAct's one local concern stays local: the
        optional marked-truncation bounds ride in as the truncate hook.
        """
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
                # Even at a tiny bound, never emit an UNMARKED slice
                # (ADR-0026; 0029 #15 — this branch used to drop the suffix).
                keep = max(0, max_chars - 1)
                suffix = "…"
            #[WARNING:TRUNCATION] bounded message content for LLM payload
            return text[:keep].rstrip() + suffix

        def _bound(text: str, role: str) -> str:
            limit = max_tool_message_chars if role == "tool" else max_message_chars
            return _truncate(text, max_chars=limit)

        return sanitize_transcript_messages(messages, truncate=_bound)

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

        payload: Dict[str, Any] = {"prompt": prompt, "params": runtime_llm_params(runtime_ns, extra={"temperature": 0.2})}
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media
        sys = _compose_system_prompt(runtime_ns, base="")
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys

        # Same per-run routing as reason (split-brain fix, 2026-07-13): a run
        # routed to a specific model must plan on that model too.
        eff_provider = runtime_ns.get("provider")
        eff_model = runtime_ns.get("model")
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
        else:
            # Backward compatibility: use scratchpad
            iteration = int(scratchpad.get("iteration", 0) or 0)
        # Presence-based budget resolution (explicit 0 clamps to 1, never falls
        # open to the default — agency-caps adversary P2-2).
        max_iterations = resolve_max_iterations(limits, scratchpad)

        if iteration >= max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        # Update both for transition period
        scratchpad["iteration"] = iteration + 1
        limits["current_iteration"] = iteration + 1

        # Inbox is a small, host/agent-controlled injection channel. Drained guidance joins the
        # durable transcript as a user interjection (maintainer ruling 2026-07-09, same as the
        # ReAct adapter): the previous rendering folded it into the system prompt for ONE call,
        # which both mutated the cached prefix and was forgotten on the next cycle — a final
        # answer written later re-anchored on the original task and dropped the correction.
        _fold_hook_steering(runtime_ns)
        inbox = runtime_ns.get("inbox", [])
        guidance = ""
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

        # STAMP THE TURN ONCE (mission A ported to CodeAct by mission A3,
        # 2026-09-22). The runtime grounds every LLM turn with a
        # `<runtime_metadata>` envelope carrying the local time. Injected at the
        # PAYLOAD boundary and never stored, it made the durable transcript hold
        # `do X` while the model was sent `<runtime_metadata>{…}</runtime_metadata>\ndo X`,
        # so the same turn re-rendered differently one call later and no prefix
        # cache could restore past it. Stamping the DURABLE message here makes the
        # bytes we send the bytes we store; `_normalize_turn_grounding` keeps what
        # it finds. Idempotent, so a tool loop stamps its turn exactly once, and it
        # refuses payload-synthesized carriers.
        stamp_user_turn_grounding(context.get("messages"))

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

        req = logic.build_request(
            task=str(context.get("task", "") or ""),
            messages=messages_view,
            # Guidance no longer rides the system prompt (cache stability + durability): it is
            # already in `messages_view` as a durable transcript message (see the drain above).
            guidance="",
            iteration=iteration + 1,
            max_iterations=max_iterations,
            vars=run.vars,  # Pass vars for _limits access
        )

        emit("reason", {"iteration": iteration + 1, "max_iterations": max_iterations, "has_guidance": bool(guidance)})
        ctx_warn = context_usage_warning(limits, scratchpad)
        if ctx_warn:
            emit("context_warning", ctx_warn)


        # IMPORTANT: When we send `messages`, do not also send a non-empty `prompt`.
        # Some providers/servers will append `prompt` as an extra user message even when the
        # current request is already present in `messages`, which duplicates user turns and
        # wastes context budget.
        payload: Dict[str, Any] = {
            "prompt": "",
            "messages": _sanitize_llm_messages(messages_view, limits=limits),
            "tools": list(tool_specs),
        }
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media
        sys = _compose_system_prompt(runtime_ns, base=str(req.system_prompt or ""))
        if isinstance(sys, str) and sys.strip():
            payload["system_prompt"] = sys

        # Volatile per-call state (loop position + live plan) rides a TRAILING
        # ephemeral message, never the system prompt (0212 propagated from
        # ReAct, fable5 2026-07-13: the `Iteration: N/M` head line busted the
        # provider prefix cache on EVERY cycle — full re-prefill per call on
        # local servers). Adjacency guard mirrors ReAct: merge into a trailing
        # user message; else append flagged `volatile: True` (runtime excludes
        # flagged messages from the cache fingerprint and strips the key).
        # `_runtime.suppress_loop_tail` (c2447): loop tails are task-agent
        # chrome — entity-lane hosts suppress the whole block (ReAct parity).
        chrome_parts: list[str] = []
        tail_parts: list[str] = []
        if not suppress_loop_tail(runtime_ns):
            chrome_parts.append(f"[loop] iteration {int(iteration + 1)} of {int(max_iterations)}.")
            if _flag(runtime_ns, "plan_mode", default=False):
                plan_text = scratchpad.get("plan")
                if isinstance(plan_text, str) and plan_text.strip():
                    # ADR-0026 (2026-08-02 purge): same as ReAct — the plan flows
                    # whole unless the caller sets _limits.plan_render_max_chars.
                    plan_render = plan_text.strip()
                    try:
                        _plan_cap = int(limits.get("plan_render_max_chars", -1))
                    except (TypeError, ValueError):
                        _plan_cap = -1
                    if _plan_cap > 0 and len(plan_render) > _plan_cap:
                        #[WARNING:TRUNCATION] caller-set _limits.plan_render_max_chars bound
                        plan_render = plan_render[:_plan_cap].rstrip() + (
                            f"\n… #[WARNING:TRUNCATION] plan clipped to the caller-set "
                            f"_limits.plan_render_max_chars={_plan_cap} ({len(plan_text.strip()):,} chars total)"
                        )
                    tail_parts.append(f"[plan]\n{plan_render}")
        # Mission A3: one placement rule for all three loops
        # (`transcripts.place_loop_tail`). Chat shape: chrome is dropped and
        # only the actionable plan merges (this loop used to merge the position
        # line INTO the user's durable message — mission A's bug 1b, never ported
        # here). Tool-loop shape: the tail becomes a durable, marked message, so
        # iteration N's prompt stays an exact prefix of N+1's.
        if isinstance(payload.get("messages"), list) and (chrome_parts or tail_parts):
            payload["messages"] = place_loop_tail(
                durable_messages=context.get("messages"),
                payload_messages=payload["messages"],
                chrome_parts=chrome_parts,
                actionable_parts=tail_parts,
                new_message=lambda **kw: _new_message(ctx, **kw),
            )

        # Per-run substrate override honesty (fable5 B-F10 2026-07-13): ReAct
        # and MemAct honor `_runtime.provider/model` (the gateway's per-run
        # routing channel); CodeAct silently ignored it — a host swapping
        # models per run got the runtime default with no warning.
        eff_provider = runtime_ns.get("provider")
        eff_model = runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        params: Dict[str, Any] = {}
        if req.max_tokens is not None:
            params["max_tokens"] = req.max_tokens
        payload["params"] = runtime_llm_params(runtime_ns, extra=params)

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

        temp.pop("llm_response", None)
        # Fence decision computed BEFORE the emit (wave-F P3: on the fenced
        # path the parse payload said has_tool_calls=False with no code
        # signal, then the loop executed code — an action a common-core
        # consumer never saw coming). Same conditions as the extraction
        # branch below, which reuses this result (one decision, two readers).
        fenced_code: Optional[str] = None
        if not tool_calls and isinstance(content, str) and content.strip():
            if not content.lstrip().upper().startswith("FINAL:") and logic.fenced_fallback_enabled(runtime_ns):
                fenced_code = logic.extract_code(content)
        # COMMON CORE parse payload (0028 contract wave, 2026-07-14): every
        # loop guarantees has_tool_calls + tool_calls + content_preview (200
        # chars); extras are loop-specific additions (has_code is CodeAct's).
        # Preview bound unified 100 -> 200 (declared in hooks.md).
        parse_payload: Dict[str, Any] = {
            "has_tool_calls": bool(tool_calls),
            "tool_calls": [
                {"name": tc.name, "arguments": (dict(tc.arguments) if isinstance(tc.arguments, dict) else (list(tc.arguments) if isinstance(tc.arguments, list) else tc.arguments)), "call_id": tc.call_id} for tc in tool_calls
            ],
            "content_preview": parse_content_preview(content),
            "has_code": bool(fenced_code),
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
            try:
                read_hint = detect_nearby_same_file_staircase(
                    scratchpad.get("last_successful_read_batch"),
                    tool_calls,
                    previously_warned_signature=str(scratchpad.get("read_orchestration_last_hint") or ""),
                )
                if read_hint is None:
                    read_hint = detect_redundant_same_file_full_reread(
                        scratchpad.get("last_successful_read_batch"),
                        tool_calls,
                        previously_warned_signature=str(scratchpad.get("read_orchestration_last_hint") or ""),
                    )
                if read_hint is not None:
                    _push_inbox(runtime_ns, str(read_hint.get("message") or ""))
                    scratchpad["read_orchestration_last_hint"] = str(read_hint.get("signature") or "")
                    emit(
                        "parse_read_orchestration_hint",
                        {
                            "path": read_hint.get("path"),
                            "mode": str(read_hint.get("mode") or "read_orchestration"),
                            "enforcement": str(read_hint.get("enforcement") or "advise"),
                        },
                    )
                    # ADVICE, NOT REFUSAL — see the twin comment in
                    # react_runtime.parse_node and the read_orchestration
                    # module docstring. Dropping a side-effect-free read cost
                    # a task iteration and returned the model no data.
                    if str(read_hint.get("enforcement") or "advise") != "advise":
                        temp["pending_tool_calls"] = []
                        return StepPlan(node_id="parse", next_node="reason")
            except Exception:
                pass
            # A non-empty reply ends the CONSECUTIVE-empty streak (fable5 P1
            # 2026-07-13: without the reset, two recovered empties early in a
            # run made every LATER single empty reply skip its retries and end
            # the run with the "can't proceed" error despite remaining budget).
            scratchpad["empty_response_retry_count"] = 0
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
                    content=str(content or ""),
                    tool_calls=tool_calls,
                    metadata={"kind": "tool_calls"},
                )
            )
            if content and _flag(runtime_ns, "plan_mode", default=False):
                updated = _extract_plan_update(content)
                if isinstance(updated, str) and updated.strip():
                    scratchpad["plan"] = updated.strip()
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            # tool_proposed on all three loops (0026 follow-up, 2026-07-15):
            # the canonical commit signal, same raw step + payload as ReAct.
            # The fenced-code path deliberately stays out — no tool batch is
            # proposed there; `parse`'s additive `has_code` carries it.
            emit("parse_tool_calls", {"count": len(tool_calls)})
            return StepPlan(node_id="parse", next_node="act")

        # Empty response is an invalid step: recover with a bounded retry that carries evidence.
        if not isinstance(content, str) or not content.strip():
            try:
                empty_retries = int(scratchpad.get("empty_response_retry_count") or 0)
            except Exception:
                empty_retries = 0

            if empty_retries < 2:
                scratchpad["empty_response_retry_count"] = empty_retries + 1
                emit("parse_retry_empty", {"retries": empty_retries + 1})
                inbox = runtime_ns.get("inbox")
                if not isinstance(inbox, list):
                    inbox = []
                    runtime_ns["inbox"] = inbox
                inbox.append(
                    {
                        "content": (
                            "[Recover] Your last message was empty. Continue the task now. "
                            "If you need info, CALL tools (preferred). Do not output an empty message."
                        )
                    }
                )
                return StepPlan(node_id="parse", next_node="reason")

            safe = (
                "I can't proceed: the model repeatedly returned empty outputs (no content, no tool calls).\n"
                "Please retry, reduce context, or switch models."
            )
            context["messages"].append(_new_message(ctx, role="assistant", content=safe, metadata={"kind": "error"}))
            temp["final_answer"] = safe
            temp["pending_tool_calls"] = []
            scratchpad["empty_response_retry_count"] = 0
            return StepPlan(node_id="parse", next_node="maybe_review")

        def _extract_final_answer(text: str) -> tuple[bool, str]:
            if not isinstance(text, str) or not text.strip():
                return False, ""
            s = text.lstrip()
            if s.upper().startswith("FINAL:"):
                return True, s[len("FINAL:") :].lstrip()
            return False, text

        # Reaching here = a NON-EMPTY reply: the consecutive-empty streak ends
        # regardless of which branch (FINAL / fenced / default-final) exits.
        scratchpad["empty_response_retry_count"] = 0

        # Intent before extraction (backlog 0010 / A4): the FINAL check MUST
        # precede fenced-code extraction — a reply "FINAL: here's an example:
        # ```python ..." is a final ANSWER whose illustrative block used to
        # execute (unintended execution on a turn that carried no action
        # intent). A final answer never executes anything.
        raw = str(content or "").strip()
        is_final, final = _extract_final_answer(raw)
        if is_final:
            if final:
                context["messages"].append(_new_message(ctx, role="assistant", content=final))
                if _flag(runtime_ns, "plan_mode", default=False):
                    updated = _extract_plan_update(final)
                    if isinstance(updated, str) and updated.strip():
                        scratchpad["plan"] = updated.strip()
            temp["final_answer"] = final or "No answer provided"
            temp["pending_tool_calls"] = []
            return StepPlan(node_id="parse", next_node="maybe_review")

        # Fenced-code fallback for prompted models (the CodeAct convention the
        # system prompt teaches when enabled). A1 landed 2026-07-14 (operator
        # green light): the DEFAULT is capability-conditional — explicit
        # `_runtime.codeact_fenced_fallback` wins both ways, otherwise
        # NOT supports_native_tools (runtime seeds the bit per run). ONE
        # resolver shared with the prompt line (CodeActLogic.
        # fenced_fallback_enabled) so parser and teaching can never disagree:
        # teaching a disabled channel manufactures dead replies; extracting an
        # untaught one executes illustrations.
        if fenced_code:
            if content:
                context["messages"].append(_new_message(ctx, role="assistant", content=content))
                if _flag(runtime_ns, "plan_mode", default=False):
                    updated = _extract_plan_update(content)
                    if isinstance(updated, str) and updated.strip():
                        scratchpad["plan"] = updated.strip()
            temp["pending_code"] = fenced_code
            return StepPlan(node_id="parse", next_node="execute_code")

        # Default: treat as a final answer even without an explicit FINAL marker.
        if raw:
            context["messages"].append(_new_message(ctx, role="assistant", content=raw))
            if _flag(runtime_ns, "plan_mode", default=False):
                updated = _extract_plan_update(raw)
                if isinstance(updated, str) and updated.strip():
                    scratchpad["plan"] = updated.strip()
        temp["final_answer"] = raw or "No answer provided"
        temp["pending_tool_calls"] = []
        scratchpad["empty_response_retry_count"] = 0
        return StepPlan(node_id="parse", next_node="maybe_review")

    def act_node(run: RunState, ctx) -> StepPlan:
        # Treat `_temp.pending_tool_calls` as a durable queue to avoid dropping tool calls when
        # schema-only tools (ask_user/memory/etc.) are interleaved with normal tools.
        context, _, runtime_ns, temp, _ = ensure_codeact_vars(run)
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
            "delegate_agent",
        }

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
                # Mission A3: one shared fallback formula (see
                # `transcripts.synthetic_call_id`); `str(idx)` disagreed with the
                # durable announcement and orphaned every result.
                d["call_id"] = synthetic_call_id(idx)
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

            # Pop builtin.
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
                    "workflow_id": str(getattr(run, "workflow_id", "") or "codeact_agent"),
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

        # Idempotency discriminator (fable5 P0 2026-07-13; see react_runtime's
        # act_node): the runtime strips call_ids from the effect hash and scans
        # the whole ledger, so a later byte-identical batch REPLAYED the stale
        # prior result. The persisted monotonic counter keeps crash-replay
        # dedup while giving each genuine issuance a fresh key.
        scratchpad_ns = run.vars.get("scratchpad") if isinstance(run.vars.get("scratchpad"), dict) else {}
        act_seq = int(scratchpad_ns.get("act_seq") or 0) + 1
        scratchpad_ns["act_seq"] = act_seq
        temp["current_tool_batch"] = list(formatted_calls)

        return StepPlan(
            node_id="act",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={"tool_calls": formatted_calls, "allowed_tools": list(allow), "act_seq": act_seq},
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def execute_code_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, runtime_ns, temp, _ = ensure_codeact_vars(run)
        code = temp.get("pending_code")
        if not isinstance(code, str) or not code.strip():
            return StepPlan(node_id="execute_code", next_node="reason")

        temp.pop("pending_code", None)
        # ADR-0027: fenced-code execution is a model-authored compute path, so
        # it must not smuggle in a hidden 10s kill switch. Timeouts belong to
        # explicit callers; the auto-issued fallback tool call carries only the
        # code and inherits execute_python's current default (no timeout).
        emit("act", {"tool": "execute_python", "args": {"code": "(inline)"}})
        allow = _effective_allowlist(runtime_ns)

        # Same P0 class, sharper here: re-running the SAME fenced code block is
        # the point of CodeAct (re-check mutable state) — without the seq, the
        # second run replayed the first run's ledger result.
        act_seq = int(scratchpad.get("act_seq") or 0) + 1
        scratchpad["act_seq"] = act_seq
        temp["current_tool_batch"] = [
            {
                "name": "execute_python",
                "arguments": {"code": code},
                "call_id": "code",
            }
        ]

        return StepPlan(
            node_id="execute_code",
            effect=Effect(
                type=EffectType.TOOL_CALLS,
                payload={
                    "tool_calls": [
                        {
                            "name": "execute_python",
                            "arguments": {"code": code},
                            "call_id": "code",
                        }
                    ],
                    "allowed_tools": list(allow),
                    "act_seq": act_seq,
                },
                result_key="_temp.tool_results",
            ),
            next_node="observe",
        )

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_codeact_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []

        read_batch = advance_last_successful_read_batch(
            scratchpad.get("last_successful_read_batch"),
            temp.get("current_tool_batch"),
            results,
        )
        if read_batch is not None:
            scratchpad["last_successful_read_batch"] = read_batch
        else:
            scratchpad.pop("last_successful_read_batch", None)
        scratchpad.pop("read_orchestration_last_hint", None)
        temp.pop("current_tool_batch", None)

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
                #[WARNING:TRUNCATION] bounded preview for observability payloads
                preview = preview[:1000] + f"\n… (truncated, {len(rendered):,} chars total)"
            emit("observe", {"tool": name, "success": success, "result": preview, "call_id": str(r.get("call_id") or "")})
            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={"name": name, "call_id": r.get("call_id"), "success": success},
                )
            )

        temp.pop("tool_results", None)
        # Reset verifier/review rounds after MODEL-issued tool activity so the
        # verifier can run again on the next candidate answer. VERIFIER-FORCED
        # batches do NOT reset (c2856 re-review blowup, same fix as ReAct): the
        # reset re-armed the verifier's own budget through calls it forced
        # itself, unbounding consecutive review rounds on one answer.
        if not temp.get("review_forced_batch"):
            scratchpad["review_count"] = 0
        pending = temp.get("pending_tool_calls", [])
        if isinstance(pending, list) and pending:
            return StepPlan(node_id="observe", next_node="act")
        temp["pending_tool_calls"] = []
        # Forced-batch marker ends with its batch — later activity is the model's.
        temp.pop("review_forced_batch", None)
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_codeact_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)

        # Per-turn report state resets at the turn boundary (wave-E P2-4
        # symmetry with ReAct, 2026-07-14): stale prior-turn review skips must
        # not pollute the next turn's output; ledger/emits keep the history.
        scratchpad.pop("review_skipped", None)
        scratchpad.pop("last_successful_read_batch", None)
        scratchpad.pop("read_orchestration_last_hint", None)
        # Forced-batch marker must not survive a user interaction (ReAct symmetry).
        temp.pop("review_forced_batch", None)

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
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)
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
                # Never an unmarked slice (ADR-0026; 0029 #15).
                keep = max(0, max_chars - 1)
                suffix = "…"
            #[WARNING:TRUNCATION] bounded transcript blocks for prompt reconstruction
            return s[:keep].rstrip() + suffix

        def _format_allowed_tools() -> str:
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
        try:
            max_tool_msgs = int(limits.get("review_max_tool_messages", -1))
        except Exception:
            max_tool_msgs = -1
        try:
            max_user_msgs = int(limits.get("review_max_user_messages", -1))
        except Exception:
            max_user_msgs = -1

        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                tool_msgs.append(_truncate_block(content.strip(), max_chars=tool_limit))
            # ADR-0026 (2026-08-02 purge): the char clips here were already
            # caller-set (-1 default), but the MESSAGE-COUNT windows were not —
            # a verifier told to judge only from tool outputs was silently shown
            # the last 8. Caller-set now; unset (<=0) = every tool message.
            if max_tool_msgs > 0 and len(tool_msgs) >= max_tool_msgs:
                #[WARNING:TRUNCATION] caller-set _limits.review_max_tool_messages window
                tool_msgs.append(
                    f"#[WARNING:TRUNCATION] older tool outputs omitted by the caller-set "
                    f"_limits.review_max_tool_messages={max_tool_msgs} window"
                )
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
                    if max_user_msgs > 0 and len(user_msgs) >= max_user_msgs:
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
                if max_user_msgs > 0 and len(ask_prompts) >= max_user_msgs:
                    break

        user_msgs.reverse()
        ask_prompts.reverse()
        user_context = "\n\n".join(user_msgs) if user_msgs else "(no additional user messages)"
        asked_context = "\n\n".join(ask_prompts) if ask_prompts else "(no ask_user prompts recorded)"

        answer_raw = str(run.vars.get("_temp", {}).get("final_answer") or "")
        answer_excerpt = ""
        if not tool_msgs and answer_raw.strip():
            answer_excerpt = _truncate_block(answer_raw.strip(), max_chars=answer_limit)

        # Execution preference (c2725/c2735 R-Type evidence; same seam as the
        # ReAct verifier): executor-tagged tools in the allowlist teach the
        # verifier that an unexecuted artifact is unverified. Empty block =
        # byte-identical prompt when no executor tool is granted.
        executors = executor_tool_names(allow, tool_tags=tool_tags_map(getattr(logic, "tools", None)))
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
            + verifier_execution_preference(executors)
        )

        # Strict-expressible shared schema (arguments ride as a JSON string —
        # a free-form {"type":"object"} dict is refused by OpenAI-strict
        # validators; see verifier_response_schema for the full rationale).
        schema = verifier_response_schema()

        emit("review_request", {"tool_messages": len(tool_msgs)})

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_schema": schema,
            "response_schema_name": "CodeActVerifier",
            "params": runtime_llm_params(runtime_ns, extra={"temperature": 0.2}),
            # Review-failure containment (backlog 0027; same contract as the
            # ReAct verifier): a terminally failed verifier call lands as
            # {"ok": False, "absorbed_failure": <error>} at result_key instead
            # of failing a run that already holds a valid answer; review_parse
            # degrades to accept-with-#FALLBACK. Runtimes without the
            # absorption mechanism ignore this key (behavior unchanged there).
            "_absorb_failure": True,
        }
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media
        sys = _compose_system_prompt(runtime_ns, base="")
        if sys:
            payload["system_prompt"] = sys

        # Same per-run routing as reason (split-brain fix, 2026-07-13): with
        # review defaulting ON, a review on a DIFFERENT (possibly unloaded)
        # model than the override made every verifier round fail-and-absorb.
        eff_provider = runtime_ns.get("provider")
        eff_model = runtime_ns.get("model")
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
        _, scratchpad, runtime_ns, temp, _ = ensure_codeact_vars(run)
        resp = temp.get("review_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}

        absorbed = resp.get("absorbed_failure")
        if absorbed is not None:
            # Review-failure containment (backlog 0027): the verifier call
            # failed terminally (runtime `_absorb_failure` shape). Verifier
            # failure must never be worse than no verifier — accept the held
            # final answer and complete, loudly. Without this, the absorbed
            # record would fall through the tolerant parse into the
            # "unactionable" retry (re-issuing a failing call) and then
            # re-enter `reason` — worse than no verifier on both counts.
            reason = str(absorbed or "").strip() or "unknown error"
            skipped = scratchpad.get("review_skipped")
            if not isinstance(skipped, list):
                skipped = []
                scratchpad["review_skipped"] = skipped
            skipped.append({"reason": reason, "warning": "#FALLBACK"})
            emit(
                "review_skipped",
                {"reason": reason, "warning": "#FALLBACK", "accepted_held_answer": True},
            )
            temp.pop("review_llm_response", None)
            return StepPlan(node_id="review_parse", next_node="done")

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
                # `arguments` arrives as a JSON-encoded string per the strict
                # schema; dicts (legacy/lenient shape) pass through unchanged.
                args = coerce_verifier_tool_arguments(item.get("arguments"))
                if name:
                    next_tool_calls.append({"name": name, "arguments": args})

        emit("review", {"complete": complete, "missing": missing})
        temp.pop("review_llm_response", None)

        if complete:
            return StepPlan(node_id="review_parse", next_node="done")

        if next_tool_calls:
            temp["pending_tool_calls"] = next_tool_calls
            # Mark the batch verifier-forced: observe must not reset the review
            # budget for it (c2856 re-review blowup class).
            temp["review_forced_batch"] = True
            emit("review_tool_calls", {"count": len(next_tool_calls)})
            return StepPlan(node_id="review_parse", next_node="act")

        # Incomplete but no actionable tool calls. The old "nudge then
        # re-review" was DELETED (fable5 P1 2026-07-13, mirroring ReAct's
        # earlier resolution): the re-review payload was byte-identical, so
        # the runtime idempotency layer REPLAYED the first verdict — a
        # guaranteed no-op — and the reviewer-directed nudge ("Return JSON
        # only") then leaked into the MAIN model's durable guidance tail at
        # the next reason drain, steering the agent toward emitting raw JSON.
        runtime_ns.pop("review_retry_count", None)
        if next_prompt_text:
            inbox = runtime_ns.get("inbox")
            if not isinstance(inbox, list):
                inbox = []
                runtime_ns["inbox"] = inbox
            inbox.append({"content": f"[Review] {next_prompt_text}"})
        return StepPlan(node_id="review_parse", next_node="reason")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_codeact_vars(run)
        answer = str(temp.get("final_answer") or "No answer provided")
        emit("done", {"answer": answer, "outcome": "final_answer"})

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

        _discard_hook_steering_at_terminal()
        _note_undelivered_inbox_at_terminal(runtime_ns)
        # Loudness parity with ReAct's report line (fable5 P2 2026-07-13): a
        # skipped verifier round must be visible in the RUN OUTPUT, not only in
        # the emit lane — CodeAct's output carried no report, so the #FALLBACK
        # marker was invisible to output-only consumers.
        skipped_reviews = scratchpad.get("review_skipped")
        has_skips = isinstance(skipped_reviews, list) and bool(skipped_reviews)
        complete_output: Dict[str, Any] = {
            "answer": answer,
            "iterations": iterations,
            "messages": list(context.get("messages") or []),
            # Machine-readable terminal outcome (canonical turn_end vocabulary,
            # final_answer | iteration_budget — see react_runtime's done node).
            "outcome": "final_answer",
            # One shape across loops (wave adversary P1): review_skipped is an
            # ALWAYS-PRESENT bool everywhere; the detail list rides a separate
            # key (it was a present-only-when-truthy list here — a consumer
            # coded to the documented bool got type instability).
            "review_skipped": has_skips,
        }
        if has_skips:
            complete_output["review_skipped_details"] = list(skipped_reviews)
        return StepPlan(node_id="done", complete_output=complete_output)

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_codeact_vars(run)

        # Prefer _limits, fall back to scratchpad
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations, "outcome": "iteration_budget"})

        messages = list(context.get("messages") or [])
        # The answer is the agent's LAST WORDS, not the raw last message
        # (0029 #10: the tail is often a tool observation or a recovery user
        # note — reporting it as "answer" lied about what the agent said).
        # Full conclusion-call parity with ReAct stays gated on 0021; this is
        # the honesty half. `.get`, not [] (fable5 2026-07-13): content-less
        # host-seeded messages raised KeyError at the terminal.
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                text = str(msg.get("content") or "")
                if text.strip():
                    answer = text
                    break
        if not answer:
            answer = f"Reached the iteration budget ({max_iterations}) without producing an answer."
        # Transcript ends with the agent's answer (final_answer marker parity
        # with the done terminal) unless it already does.
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
                # Always-present at BOTH terminals (wave-F P3: the batch-3
                # "always-present bool" claim was false at this one).
                "review_skipped": bool(scratchpad.get("review_skipped")),
            },
        )

    return WorkflowSpec(
        workflow_id=workflow_id,
        entry_node="init",
        nodes={
            node_id: _with_run_context(node_fn)
            for node_id, node_fn in {
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
            }.items()
        },
    )
