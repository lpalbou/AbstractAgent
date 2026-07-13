"""AbstractRuntime adapter for canonical ReAct agents.

This adapter implements a deterministic ReAct loop:

  init → reason → parse → (act → observe → reason)* → done

Policy (for now):
- Do NOT truncate ReAct loop context (history/scratchpad).
- Do NOT cap tool-steps to tiny token budgets.
- Do NOT require "FINAL:" markers or other termination hacks.

The loop continues whenever the model emits tool calls.
It ends only when the model emits **no tool calls** and provides an answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces

from .generation_params import resolve_max_iterations, runtime_llm_params
from .loop_hooks import LoopHooks
from .media import extract_media_from_context
from .tool_allowlist import note_pruned_grants
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


def _new_assistant_message_with_tool_calls(
    ctx: Any,
    *,
    content: str,
    tool_calls: List[ToolCall],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an assistant message that preserves tool call metadata for OpenAI transcripts."""

    msg = _new_message(ctx, role="assistant", content=content, metadata=metadata)

    tc_payload: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, ToolCall):
            continue
        name = str(tc.name or "").strip()
        if not name:
            continue
        call_id = tc.call_id
        call_id_str = str(call_id).strip() if call_id is not None else ""
        if not call_id_str:
            call_id_str = f"call_{i+1}"
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        tc_payload.append(
            {
                "type": "function",
                "id": call_id_str,
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )

    if tc_payload:
        msg["tool_calls"] = tc_payload
    return msg


def ensure_react_vars(
    run: RunState,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place."""

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

    if not isinstance(scratchpad.get("cycles"), list):
        scratchpad["cycles"] = []

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

    used_tools = scratchpad.get("used_tools")
    if not isinstance(used_tools, bool):
        scratchpad["used_tools"] = bool(used_tools) if used_tools is not None else False

    return context, scratchpad, runtime_ns, temp, limits


def _compute_toolset_id(tool_specs: List[Dict[str, Any]]) -> str:
    normalized = sorted((dict(s) for s in tool_specs), key=lambda s: str(s.get("name", "")))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"ts_{digest}"


def _tool_call_signature(name: str, args: Any) -> str:
    def _abbrev(v: Any, *, max_chars: int = 140) -> str:
        if v is None:
            return ""
        s = str(v)
        if len(s) <= max_chars:
            return s
        #[WARNING:TRUNCATION] bounded argument preview in tool-call signatures (ADR-0026: marked, full args stay in scratchpad/ledger)
        return f"{s[: max(0, max_chars - 1)]}… (truncated, {len(s):,} chars total)"

    def _hash_str(s: str) -> str:
        try:
            return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        except Exception:
            return "sha256_err"

    n = str(name or "").strip() or "tool"
    if not isinstance(args, dict) or not args:
        return f"{n}()"

    # Special-case common large-argument tools so rendered signatures (conclusion prompt,
    # final report) don't explode. Content is replaced by len+sha256 provenance (ADR-0026
    # compression-with-provenance, not silent truncation).
    if n == "write_file":
        fp = args.get("file_path") if isinstance(args.get("file_path"), str) else args.get("path")
        mode = args.get("mode") if isinstance(args.get("mode"), str) else "w"
        content = args.get("content")
        if isinstance(content, str):
            tag = f"<str len={len(content)} sha256={_hash_str(content)}>"
        else:
            tag = "<str len=0>"
        return f"write_file(file_path={_abbrev(fp)!r}, mode={_abbrev(mode)!r}, content={tag})"

    if n == "edit_file":
        fp = args.get("file_path") if isinstance(args.get("file_path"), str) else args.get("path")
        edits = args.get("edits")
        n_edits = len(edits) if isinstance(edits, list) else 0
        return f"edit_file(file_path={_abbrev(fp)!r}, edits={n_edits})"

    if n == "fetch_url":
        url = args.get("url")
        include_full = args.get("include_full_content")
        return f"fetch_url(url={_abbrev(url)!r}, include_full_content={include_full})"

    if n == "web_search":
        q = args.get("query")
        num = args.get("num_results")
        return f"web_search(query={_abbrev(q)!r}, num_results={num})"

    if n == "execute_command":
        cmd = args.get("command")
        return f"execute_command(command={_abbrev(cmd, max_chars=220)!r})"

    # Generic, but bounded: hash long strings to avoid leaking large blobs into the prompt.
    summarized: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 160:
            summarized[str(k)] = f"<str len={len(v)} sha256={_hash_str(v)}>"
        else:
            summarized[str(k)] = v
    try:
        arg_str = json.dumps(summarized, ensure_ascii=False, sort_keys=True)
    except Exception:
        arg_str = str(summarized)
    arg_str = _abbrev(arg_str, max_chars=260)
    return f"{n}({arg_str})"


def _tool_call_fingerprint(name: str, args: Any) -> str:
    """Return a stable, bounded fingerprint for tool-call repeat detection.

    Important: do not embed large string blobs (file contents / web pages) in the fingerprint.
    """

    def _hash_str(s: str) -> str:
        try:
            return hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            return "sha256_err"

    def _canon(v: Any) -> Any:
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            if len(v) <= 200:
                return v
            return {"_type": "str", "len": len(v), "sha256": _hash_str(v)[:16]}
        if isinstance(v, list):
            return [_canon(x) for x in v[:25]]
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for k in sorted(v.keys(), key=lambda x: str(x)):
                out[str(k)] = _canon(v.get(k))
            return out
        return {"_type": type(v).__name__}

    payload = {"name": str(name or "").strip(), "args": _canon(args if isinstance(args, dict) else {})}
    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        raw = str(payload)
    try:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "fingerprint_err"


_FINALISH_RE = re.compile(
    r"(?i)\b(final answer|here is|here['’]s|here are|below is|below are|done|completed|in summary|summary|result)\b"
)

_WAITING_RE = re.compile(
    r"(?i)\b("
    r"let me know|your next step|what would you like|tell me|"
    r"i can help|i'm ready|i am ready|"
    r"i'll wait|i will wait|waiting for|"
    r"no tool calls?"
    r")\b"
)

_DEFERRED_ACTION_INTENT_RE = re.compile(
    # Only treat as "missing tool calls" when the model *commits to acting*
    # (first-person intent) rather than providing a final answer.
    r"(?i)\b(i will|i['’]?ll|let me|i am going to|i['’]?m going to|i need to)\b"
)

_DEFERRED_ACTION_VERB_RE = re.compile(
    # Verbs that typically imply external actions (tools/files/web/edits).
    r"(?i)\b(read|open|search|list|skim|inspect|explore|scan|run|execute|edit|fetch|download|creat(?:e|ing))\b"
)

_TOOL_CALL_MARKERS = ("<function_call>", "<tool_call>", "<|tool_call|>", "```tool_code")

# Paired fenced blocks (```lang ... ```). Used to make prose heuristics fence-blind:
# fenced content is quoted material (code samples, entity election fences like
# ```diary/```feel — frozen visit seam spec a2a 0013 §4 line 6), never the assistant's
# own action commitment. An unterminated trailing fence stays in the prose view
# (conservative: no pair, no exclusion).
_FENCED_BLOCK_RE = re.compile(r"(?s)```[^\n`]*\n.*?```")


def _contains_tool_call_markup(text: str) -> bool:
    s = str(text or "")
    if not s.strip():
        return False
    low = s.lower()
    return any(m in low for m in _TOOL_CALL_MARKERS)


_TOOL_CALL_STRIP_RE = re.compile(
    r"(?is)"
    r"<function_call>\s*.*?\s*</function_call>|"
    r"<tool_call>\s*.*?\s*</tool_call>|"
    r"<\|tool_call\|>.*?<\|/tool_call\|>|"
    r"```tool_code\s*.*?```"
)


def _strip_tool_call_markup(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    try:
        return _TOOL_CALL_STRIP_RE.sub("", raw)
    except Exception:
        return raw


# --- Act-only tool results (frozen visit seam spec, a2a thread 0013 v2 §2) ---
#
# Act-only tools (e.g. entity diary reads) must never land their CONTENT in any
# at-rest surface outside its one home (G1: "the book's words never rest outside
# the book"). The durable transcript carries an ACT-FRAME REFERENCE instead; the
# runtime LLM_CALL handler dereferences it at the provider boundary (send time)
# into a wire COPY. The adapter is NOT the privacy mechanism (the effect handler
# is — words must never reach the tool-result channel); this rendering is the
# canonical-shape half plus fail-safe defense in depth.

_ACT_ONLY_KEY = "$act_only"


def _act_only_ref_content(frame: Dict[str, Any]) -> str:
    """Serialize an act-frame as canonical `$act_only` reference message content.

    Contract (spec v2 §2 + agent pins): the reference rides the tool message's
    CONTENT as one exact JSON object with a lone `$act_only` top-level key, so the
    runtime handler detects it by parse (json.loads), never by regex. Deterministic
    serialization (sorted keys) keeps the durable bytes stable for prefix caching.
    """
    return json.dumps({_ACT_ONLY_KEY: frame}, ensure_ascii=False, sort_keys=True)


def _act_only_frame_from_output(output: Any) -> Optional[Dict[str, Any]]:
    """Return the act-frame dict when a tool result output is `$act_only`-shaped.

    A handler-authored reference (`{"$act_only": {...}}`, lone top-level key) is
    honored regardless of local tool declarations — the effect handler is the
    enforcement authority and may mark results act-only on its own.
    """
    if (
        isinstance(output, dict)
        and set(output.keys()) == {_ACT_ONLY_KEY}
        and isinstance(output[_ACT_ONLY_KEY], dict)
    ):
        return dict(output[_ACT_ONLY_KEY])
    return None


def _act_only_frame_is_dereferenceable(frame: Dict[str, Any]) -> bool:
    """True when a frame may take the lone-key REF shape in the durable transcript.

    Cross-package wedge guard (found reading runtime's shipped dereference,
    identity/act_only.py): the LLM_CALL wrapper resolves every `$act_only` ref in
    the transcript at SEND time. Refs are durable, so a shape that references
    NOTHING (e.g. a failure/suppression record) must never take the ref form —
    it renders as labeled non-ref text instead (inert to the dereference pass).

    TWO addressable shapes (runtime's tool+args generalization, e-s 233 R3 /
    64398ff): entry-addressed refs carry a non-empty `entry_id` (diary_read);
    re-run refs carry an `args` dict (diary_list — the listing is re-executed
    fresh at send time; empty dict = list-everything, still a valid address).
    Deliberately SHAPE-based, no tool-name list copied here: which tools resolve
    is runtime's dispatch (ACT_ONLY_TOOLS), and since the wedge amendment an
    unknown ref tombstones loudly instead of failing the call — the shape rule
    only keeps reference-free records out of the resolver's path.
    """
    if str(frame.get("entry_id") or "").strip():
        return True
    return isinstance(frame.get("args"), dict)


def _act_only_record_content(frame: Dict[str, Any]) -> str:
    """Non-ref rendering for act-only frames that reference nothing.

    Deliberately NOT the lone-key `$act_only` shape (would be parsed as a ref and
    wedge the run — see `_act_only_frame_is_dereferenceable`). Carries act-frame
    fields only; never tool-surfaced words.
    """
    tool = str(frame.get("tool") or "tool")
    detail = json.dumps({k: v for k, v in frame.items() if k != "tool"}, ensure_ascii=False, sort_keys=True)
    return f"[{tool}]: act-only record (no content at rest): {detail}"


def _looks_like_deferred_action(text: str) -> bool:
    """Return True when the model claims it will take actions but emits no tool calls.

    This is intentionally conservative: false positives waste iterations and can "force"
    unnecessary tool calls. It should only trigger when the assistant message strongly
    suggests it is about to act (not answer).

    Fence-blind (frozen visit seam spec, a2a 0013 §4 line 6): all heuristics run on the
    PROSE view with paired ```fenced blocks removed — fenced content is quoted material
    (code samples, entity election fences like ```diary whose first-person text, e.g.
    "I will keep reading…", is diary content, not an action commitment). A retry
    triggered by fence content would discard a reply carrying elections — the exact
    consume-the-fences failure the spec's never-strip obligation forbids. A reply that
    is ONLY fences is a valid final answer (elections are reply content).
    """
    s = str(text or "").strip()
    if not s:
        return False
    prose = _FENCED_BLOCK_RE.sub("", s).strip()
    if not prose:
        return False
    # If the model is explicitly waiting for user direction, that's a valid final response.
    if _WAITING_RE.search(prose):
        return False
    # Common “final answer” framing (incl. typographic apostrophes).
    if _FINALISH_RE.search(prose):
        return False
    # If the model already produced a structured answer (headings/sections), don't retry.
    if re.search(r"(?m)^(#{1,6}\s+\\S|\\*\\*\\S)", prose):
        return False
    # Must contain first-person intent *and* an action-ish verb.
    if not _DEFERRED_ACTION_INTENT_RE.search(prose):
        return False
    if not _DEFERRED_ACTION_VERB_RE.search(prose):
        return False
    return True


def _push_inbox(runtime_ns: Dict[str, Any], content: str) -> None:
    if not isinstance(runtime_ns, dict):
        return
    inbox = runtime_ns.get("inbox")
    if not isinstance(inbox, list):
        inbox = []
        runtime_ns["inbox"] = inbox
    inbox.append({"role": "system", "content": str(content or "")})


def _drain_inbox(runtime_ns: Dict[str, Any]) -> str:
    inbox = runtime_ns.get("inbox")
    if not isinstance(inbox, list) or not inbox:
        return ""
    parts: list[str] = []
    for m in inbox:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
    runtime_ns["inbox"] = []
    return "\n".join(parts).strip()


def _boolish(value: Any) -> bool:
    """Best-effort coercion for runtime flags (bool/int/str)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    return False

def _apply_llm_payload_extras(runtime_ns: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Copy embedding-workflow payload keys from `_runtime` onto an LLM_CALL payload.

    G1 write direction (runtime b8b8c78/56e55c3): the entity runtime's result-boundary
    election capture keys book writes on the payload's `turn_id` (elections with no
    turn_id fail loud) and threads word-free `anchor_record_ids`/`anchor_graph_ids`
    into the DIARY_WRITE it performs. The embedding visit workflow sets
    `_runtime.turn_id` / `_runtime.llm_payload_extras` per turn (RENDER/ROUTE seam);
    outside visits the keys are absent and payloads are byte-unchanged.
    """
    if not isinstance(runtime_ns, dict):
        return
    turn_id_val = runtime_ns.get("turn_id")
    if isinstance(turn_id_val, str) and turn_id_val.strip():
        payload["turn_id"] = turn_id_val.strip()
    extras = runtime_ns.get("llm_payload_extras")
    if isinstance(extras, dict):
        for key, value in extras.items():
            if isinstance(key, str) and key.strip() and key not in payload:
                payload[key] = value


def _system_prompt_override(runtime_ns: Dict[str, Any]) -> Optional[str]:
    raw = runtime_ns.get("system_prompt") if isinstance(runtime_ns, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _system_prompt_extra(runtime_ns: Dict[str, Any]) -> Optional[str]:
    raw = runtime_ns.get("system_prompt_extra") if isinstance(runtime_ns, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _compose_system_prompt(runtime_ns: Dict[str, Any], *, base: str) -> str:
    override = _system_prompt_override(runtime_ns)
    extra = _system_prompt_extra(runtime_ns)
    sys = override if override is not None else base
    if extra:
        sys = f"{sys.rstrip()}\n\nAdditional system instructions:\n{extra}"
    return sys.strip()


def _max_output_tokens(runtime_ns: Dict[str, Any], limits: Dict[str, Any]) -> Optional[int]:
    # Canonical limit: _limits.max_output_tokens (None = unset).
    raw = None
    if isinstance(limits, dict) and "max_output_tokens" in limits:
        raw = limits.get("max_output_tokens")
    if raw is None and isinstance(runtime_ns, dict):
        raw = runtime_ns.get("max_output_tokens")
    if raw is None:
        return None
    try:
        val = int(raw)
    except Exception:
        return None
    return val if val > 0 else None


# NOTE (0212/0213): `_render_cycles_for_system_prompt` was removed. The per-cycle scratchpad is no
# longer rendered into the system prompt: it mutated the cached prefix every iteration (defeating
# provider prompt caching) and duplicated observations that already live in `context.messages`.
# The model's reasoning now stays in the transcript (assistant tool-call messages keep `content`),
# and `scratchpad.cycles` remains a durable host-side record (final report / conclusion prompt).


def _truncate_preview(text: str, *, max_chars: int) -> str:
    """Bounded preview for prompt/report rendering — never for durable state.

    ADR-0026: lossy truncation must never be silent. The returned text always carries an
    explicit `… (truncated, N chars total)` marker, and the full content remains durably
    available (scratchpad cycles in run vars + runtime ledger records).
    """
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    suffix = f"… (truncated, {len(s):,} chars total)"
    keep = max(0, max_chars - len(suffix))
    #[WARNING:TRUNCATION] bounded scratchpad/conclusion preview; full text stays in scratchpad.cycles + ledger
    return s[:keep].rstrip() + suffix


def _render_cycles_for_conclusion_prompt(scratchpad: Dict[str, Any]) -> str:
    cycles = scratchpad.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return ""

    # The conclusion prompt should have access to the full loop trace, but still needs
    # to be bounded (tool outputs may be huge).
    max_cycles = 25
    max_thought_chars = 900
    max_obs_chars = 360

    view = [c for c in cycles if isinstance(c, dict)]
    total = len(view)
    if total > max_cycles:
        view = view[-max_cycles:]

    lines: list[str] = []
    if total > len(view):
        lines.append(f"(showing last {len(view)} of {total} cycles)")
        lines.append("")

    for c in view:
        i = c.get("i")
        if i is None:
            continue
        lines.append(f"[cycle {i}]")

        thought = _truncate_preview(str(c.get("thought") or "").strip(), max_chars=max_thought_chars)
        if thought:
            lines.append(f"thought: {thought}")

        tcs = c.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            sigs: list[str] = []
            for tc in tcs:
                if isinstance(tc, dict):
                    sigs.append(_tool_call_signature(tc.get("name", ""), tc.get("arguments")))
            if sigs:
                lines.append("actions:")
                for s in sigs:
                    lines.append(f"- {s}")

        obs = c.get("observations")
        if isinstance(obs, list) and obs:
            lines.append("observations:")
            for o in obs:
                if not isinstance(o, dict):
                    continue
                name = str(o.get("name") or "tool")
                ok = bool(o.get("success"))
                out = o.get("output")
                err = o.get("error")
                if not ok:
                    text = str(err or out or "").strip()
                else:
                    if isinstance(out, dict):
                        url = out.get("url") if isinstance(out.get("url"), str) else None
                        status = out.get("status_code") if out.get("status_code") is not None else None
                        content_type = out.get("content_type") if isinstance(out.get("content_type"), str) else None
                        rendered = out.get("rendered") if isinstance(out.get("rendered"), str) else None
                        rendered_len = len(rendered) if isinstance(rendered, str) else None
                        parts: list[str] = []
                        if url:
                            parts.append(f"url={url}")
                        if status is not None:
                            parts.append(f"status={status}")
                        if content_type:
                            parts.append(f"type={content_type}")
                        if rendered_len is not None:
                            parts.append(f"rendered_len={rendered_len}")
                        if parts:
                            text = ", ".join(parts)
                        else:
                            # Structural key listing (not content); disclose when clipped.
                            keys_view = [str(k) for k in out.keys()]
                            text = f"keys={keys_view[:8]}" + (
                                f" (+{len(keys_view) - 8} more keys)" if len(keys_view) > 8 else ""
                            )
                    else:
                        text = str(out or "").strip()
                text = _truncate_preview(text, max_chars=max_obs_chars)
                lines.append(f"- [{name}] {'OK' if ok else 'ERR'}: {text}")

        lines.append("")

    return "\n".join(lines).strip()


def _render_final_report(task: str, scratchpad: Dict[str, Any]) -> str:
    cycles = scratchpad.get("cycles")
    if not isinstance(cycles, list):
        cycles = []
    lines: list[str] = []
    lines.append(f"task: {task}")
    lines.append(f"cycles: {len([c for c in cycles if isinstance(c, dict)])}")
    # Review-failure containment marker (backlog 0027): a skipped verifier round
    # must be visible in the report, not just in the emit lane.
    skipped_reviews = scratchpad.get("review_skipped")
    if isinstance(skipped_reviews, list):
        for s in skipped_reviews:
            if isinstance(s, dict) and s.get("reason"):
                lines.append(f"review: #FALLBACK skipped ({str(s.get('reason'))})")
    lines.append("")
    for c in cycles:
        if not isinstance(c, dict):
            continue
        i = c.get("i")
        lines.append(f"cycle {i}")
        thought = str(c.get("thought") or "").strip()
        if thought:
            lines.append(f"- thought: {thought}")
        tcs = c.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            lines.append("- actions:")
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                lines.append(f"  - {_tool_call_signature(tc.get('name',''), tc.get('arguments'))}")
        obs = c.get("observations")
        if isinstance(obs, list) and obs:
            lines.append("- observations:")
            for o in obs:
                if not isinstance(o, dict):
                    continue
                name = str(o.get("name") or "tool")
                ok = bool(o.get("success"))
                out = o.get("output")
                err = o.get("error")
                text = str(out if ok else (err or out) or "").strip()
                lines.append(f"  - [{name}] {'OK' if ok else 'ERR'}: {text}")
        lines.append("")
    return "\n".join(lines).strip()


def reset_react_turn(run_vars: Dict[str, Any]) -> None:
    """Reset per-TURN loop state so an embedding workflow can re-enter `reason`.

    Visit-workflow composition (frozen seam spec a2a 0013 §A/§4: "per-turn iteration
    budget rides `_limits`"): a resident/visit workflow cycles PARK → seam nodes →
    reason → … → final_next_node → PARK. Each turn gets a fresh iteration budget and
    clean per-turn scratch; the DURABLE LIFE — transcript (`context.messages`),
    `scratchpad.cycles`, `scratchpad.plan` — is deliberately untouched (append-only
    history, prefix-cache stable). The caller owns `_limits.max_iterations` (set once
    at OPEN); this resets only the counters/carriers a finished turn leaves behind.
    """
    scratchpad = run_vars.get("scratchpad")
    if isinstance(scratchpad, dict):
        scratchpad["iteration"] = 0
        scratchpad["review_count"] = 0
    limits = run_vars.get("_limits")
    if isinstance(limits, dict):
        limits["current_iteration"] = 0
    temp = run_vars.get("_temp")
    if isinstance(temp, dict):
        for key in (
            "final_answer",
            "react_output",
            "llm_response",
            "pending_tool_calls",
            "tool_results",
            "user_response",
            "review_llm_response",
            "max_iterations_llm_response",
            "max_iterations_conclude_retries",
            "turn_captures",
        ):
            temp.pop(key, None)


def create_react_workflow(
    *,
    logic: ReActLogic,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    hooks: Optional[LoopHooks] = None,
    workflow_id: str = "react_agent",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    final_next_node: Optional[str] = None,
) -> WorkflowSpec:
    """Adapt ReActLogic to an AbstractRuntime workflow.

    `hooks` (first-class loop hooks, 2026-07-12 directive): a `LoopHooks`
    registry whose handlers observe every emit point as structured
    `HookEvent`s (listen/capture) and may return guidance injections that
    fold into the DURABLE `_runtime.inbox` at the next reason boundary
    (steer — resume-safe, same channel as inject_guidance). `on_step` stays
    the flat compat surface; both may be wired at once.

    `final_next_node` (composition knob, frozen seam spec a2a 0013 §A): when set, the
    terminal nodes (`done`, `max_iterations`) do everything they do today — persist the
    final answer to the durable transcript, honor the never-strip election obligation,
    render the report — but HAND OFF to the named node instead of completing the run,
    stashing the would-be output dict at `_temp.react_output` (answer/report/iterations/
    messages/scratchpad). An embedding workflow (the entity visit TURN chain) merges
    these nodes with its own seam nodes (RECALL before `reason`, ELECTIONS after) and
    reads `_temp.final_answer` / `_temp.react_output` at the handoff. Default None:
    behavior unchanged (the run completes). Re-entry for the next turn goes through
    `reset_react_turn(run.vars)` then `reason` — never `init` (init seeds the task
    message; a visit's messages arrive from the PARK resume).
    """

    if hooks is not None and not hooks.agent:
        hooks.agent = str(workflow_id or "react_agent")

    def emit(step: str, data: Dict[str, Any]) -> None:
        if on_step:
            on_step(step, data)
        if hooks is not None:
            # Hook dispatch NEVER raises. Handler failures/steers come back
            # as follow-up (step, data) pairs for the flat on_step stream;
            # dispatch also notifies the handlers themselves (pure listen,
            # returns ignored — loop-bounded), so a hooks-only host still
            # observes hook_error / hook_steer.
            for fu_step, fu_data in hooks.dispatch(step, data):
                if on_step:
                    on_step(fu_step, fu_data)

    def _with_run_context(node_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Bind the hooks' run context around node execution so every
        dispatch and drain is keyed to the EXECUTING run — one workflow
        product serves many runs (delegate children, re-registration), and
        a queue keyed to nothing would leak steering across them."""
        if hooks is None:
            return node_fn

        def wrapper(run: RunState, ctx: Any) -> Any:
            hooks.push_run(str(getattr(run, "run_id", "") or ""))
            try:
                return node_fn(run, ctx)
            finally:
                hooks.pop_run()

        wrapper.__name__ = getattr(node_fn, "__name__", "node")
        return wrapper

    def _fold_hook_steering(runtime_ns: Dict[str, Any]) -> None:
        """Fold the CURRENT run's hook-queued injections into its durable
        inbox (reason boundary only — a hook never mutates state mid-node)."""
        if hooks is None or not isinstance(runtime_ns, dict):
            return
        for text in hooks.drain_pending_injections(hooks.current_run_id()):
            _push_inbox(runtime_ns, text)

    def _discard_hook_steering_at_terminal() -> None:
        """A completing run's undelivered steering must neither leak into a
        later run nor rot in host memory; the discard is LOUD when non-empty."""
        if hooks is None:
            return
        dropped = hooks.discard_run(hooks.current_run_id())
        if dropped:
            emit("hook_steer_discarded", {"count": dropped})

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

    def _act_only_tool_names() -> set[str]:
        # Keyed on core's first-class `act_only` field (ToolDefinition); fail-closed by
        # absence (getattr default False) so the check works before core ships the field
        # and becomes constructor-native after — one spelling, no tags fallback.
        out: set[str] = set()
        for name, t in _tool_by_name().items():
            if bool(getattr(t, "act_only", False)):
                out.add(name)
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
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, tuple):
            items = list(raw)
        elif isinstance(raw, str):
            items = [raw]
        else:
            items = []

        current = _tool_by_name()
        out: list[str] = []
        seen: set[str] = set()
        for t in items:
            if not isinstance(t, str):
                continue
            name = t.strip()
            if not name or name in seen or name not in current:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _note_pruned(runtime_ns: Dict[str, Any], raw: Any, normalized: list[str]) -> None:
        payload = note_pruned_grants(runtime_ns, raw, normalized)
        if payload is not None:
            emit("allowlist_pruned", payload)

    def _effective_allowlist(runtime_ns: Dict[str, Any]) -> list[str]:
        if isinstance(runtime_ns, dict) and "allowed_tools" in runtime_ns:
            raw = runtime_ns.get("allowed_tools")
            normalized = _normalize_allowlist(raw)
            _note_pruned(runtime_ns, raw, normalized)
            runtime_ns["allowed_tools"] = normalized
            return normalized
        # Factory channel (`create_react_workflow(allowed_tools=...)`): the
        # entity door passes its grant here, not via run vars — same
        # works-or-loud obligation, same note (when run vars can carry it).
        default = _default_allowlist()
        normalized = _normalize_allowlist(list(default))
        if isinstance(runtime_ns, dict):
            _note_pruned(runtime_ns, default, normalized)
        return normalized

    def _allowed_tool_defs(allow: list[str]) -> list[Any]:
        out: list[Any] = []
        current = _tool_by_name()
        for name in allow:
            tool = current.get(name)
            if tool is not None:
                out.append(tool)
        return out

    def _tool_prompt_examples_enabled(runtime_ns: Dict[str, Any]) -> bool:
        raw = runtime_ns.get("tool_prompt_examples") if isinstance(runtime_ns, dict) else None
        if raw is None:
            return True
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"0", "false", "no", "off", "disabled"}:
                return False
            if lowered in {"1", "true", "yes", "on", "enabled"}:
                return True
        return True

    def _materialize_tool_specs(defs: list[Any], *, include_examples: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in defs:
            try:
                d = t.to_dict()
            except Exception:
                continue
            if isinstance(d, dict):
                if not include_examples:
                    d = dict(d)
                    d.pop("examples", None)
                out.append(d)
        return out

    def _sanitize_llm_messages(messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            return []
        out: List[Dict[str, Any]] = []

        def _sanitize_tool_calls(raw: Any) -> Optional[list[dict[str, Any]]]:
            if not isinstance(raw, list) or not raw:
                return None
            cleaned: list[dict[str, Any]] = []
            for i, tc in enumerate(raw):
                if not isinstance(tc, dict):
                    continue
                tc_type = str(tc.get("type") or "function")
                if tc_type != "function":
                    continue
                call_id = tc.get("id")
                call_id_str = str(call_id).strip() if call_id is not None else ""
                if not call_id_str:
                    call_id_str = f"call_{i+1}"
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    args_str = "" if args is None else str(args)
                cleaned.append({"type": "function", "id": call_id_str, "function": {"name": name, "arguments": args_str}})
            return cleaned or None

        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip()
            if not role:
                continue
            content = m.get("content")
            content_str = "" if content is None else str(content)
            tool_calls_raw = m.get("tool_calls")
            tool_calls = _sanitize_tool_calls(tool_calls_raw)

            # Assistant tool-calls messages may legitimately have empty content, but must still be included.
            if not content_str.strip() and not (role == "assistant" and tool_calls):
                continue

            entry: Dict[str, Any] = {"role": role, "content": content_str}
            if role == "tool":
                meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
                call_id = meta.get("call_id") if isinstance(meta, dict) else None
                if call_id is not None and str(call_id).strip():
                    entry["tool_call_id"] = str(call_id).strip()
            elif role == "assistant" and tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)

        # Orphan repair (multi-turn correctness on strict providers): an assistant `tool_calls`
        # message whose ids are not answered by immediately-following tool messages makes native
        # OpenAI reject the WHOLE request ("must be followed by tool messages responding to each
        # tool_call_id" -> 400, run fails). The durable history legitimately produces this shape
        # for interactive builtins (ask_user resolves via an ASK_USER wait + a user message, never
        # a tool message). Pre-existing at git HEAD; repaired here at the payload boundary:
        # synthesize a deterministic tool result right after the assistant turn (adjacency is part
        # of the provider contract). Durable history is untouched; the synthetic text is stable so
        # the cached prefix stays byte-identical across iterations (0212).
        # Interactive builtins resolve through waits + user messages BY DESIGN; only their
        # unanswered ids may honestly be labeled "handled interactively". Anything else
        # unanswered is a genuinely lost result and must SAY so (Critic-3: a repair that
        # papers over real loss with a false claim actively misleads the model).
        _interactive_builtin_names = {"ask_user"}

        repaired: List[Dict[str, Any]] = []
        i = 0
        while i < len(out):
            entry = out[i]
            if entry.get("role") == "tool":
                # Orphan TOOL message (no immediately-preceding assistant tool_calls run —
                # e.g. a compaction/trim cut): strict providers 400 on it. Fold it into an
                # inert user-visible note instead (mirrors native-OpenAI's missing-id fold).
                repaired.append({
                    "role": "user",
                    "content": f"[unpaired tool result]: {str(entry.get('content') or '')}",
                })
                i += 1
                continue
            repaired.append(entry)
            i += 1
            if entry.get("role") != "assistant" or not entry.get("tool_calls"):
                continue
            want: Dict[str, str] = {}
            for tc in entry.get("tool_calls") or []:
                tid = str(tc.get("id") or "")
                if tid:
                    want[tid] = str(((tc.get("function") or {}).get("name")) or "")
            answered: set[str] = set()
            while i < len(out) and out[i].get("role") == "tool":
                tid = str(out[i].get("tool_call_id") or "")
                if tid:
                    answered.add(tid)
                repaired.append(out[i])
                i += 1
            for tid, name in want.items():
                if tid in answered:
                    continue
                if name in _interactive_builtin_names:
                    content = "[handled interactively; see the following conversation messages]"
                else:
                    content = f"[tool result missing (host error): {name or 'unknown tool'}]"
                repaired.append({"role": "tool", "tool_call_id": tid, "content": content})

        # Adjacent USER turns (operator guidance drained before the first assistant reply, or
        # across parse-retry cycles) are legal in durable history but 400 on alternation-strict
        # chat templates (Mistral/Gemma-class). Payload-boundary repair only: join them with a
        # blank line; the durable records stay distinct. User entries never carry tool_calls,
        # so a plain content merge is lossless.
        merged: List[Dict[str, Any]] = []
        for entry in repaired:
            if merged and entry.get("role") == "user" and merged[-1].get("role") == "user":
                prev = dict(merged[-1])
                prev["content"] = f"{str(prev.get('content') or '').rstrip()}\n\n{str(entry.get('content') or '')}"
                merged[-1] = prev
                continue
            merged.append(entry)
        return merged

    builtin_effect_tools = {
        "ask_user",
        "recall_memory",
        "inspect_vars",
        "remember",
        "remember_note",
        "compact_memory",
        "delegate_agent",
        "update_plan",
    }

    def init_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, _, limits = ensure_react_vars(run)

        scratchpad["iteration"] = 0
        limits["current_iteration"] = 0

        # Disable runtime-level input trimming for ReAct loops.
        if isinstance(runtime_ns, dict):
            runtime_ns.setdefault("disable_input_trimming", True)
        # Disable all truncation/capping knobs for ReAct runs (policy: full context for now).
        # These can be re-enabled later once correctness is proven.
        if isinstance(limits, dict):
            limits["max_output_tokens"] = None
            limits["max_input_tokens"] = None
            limits["max_history_messages"] = -1
            limits["max_message_chars"] = -1
            limits["max_tool_message_chars"] = -1

        task = str(context.get("task", "") or "")
        context["task"] = task
        msgs = context.get("messages")
        if not isinstance(msgs, list):
            msgs = []
            context["messages"] = msgs

        if task and (not msgs or msgs[-1].get("role") != "user" or msgs[-1].get("content") != task):
            msgs.append(_new_message(ctx, role="user", content=task))

        allow = _effective_allowlist(runtime_ns)
        allowed_defs = _allowed_tool_defs(allow)
        include_examples = _tool_prompt_examples_enabled(runtime_ns)
        tool_specs = _materialize_tool_specs(allowed_defs, include_examples=include_examples)
        runtime_ns["tool_specs"] = tool_specs
        runtime_ns["toolset_id"] = _compute_toolset_id(tool_specs)
        runtime_ns.setdefault("allowed_tools", allow)

        scratchpad.setdefault("cycles", [])
        return StepPlan(node_id="init", next_node="reason")

    def reason_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)

        # Durable resume safety:
        # - tool definitions can change across restarts (env/toolset swaps, staged deploy swaps)
        # - allowlists can be edited at runtime by hosts
        # `tool_specs` must match the effective allowlist + current tool defs, otherwise the LLM may
        # see tools it cannot execute ("tool not allowed") or see stale schemas (signature mismatch).
        try:
            if isinstance(runtime_ns, dict):
                allow = _effective_allowlist(runtime_ns)
                allowed_defs = _allowed_tool_defs(allow)
                include_examples = _tool_prompt_examples_enabled(runtime_ns)
                refreshed_specs = _materialize_tool_specs(allowed_defs, include_examples=include_examples)
                refreshed_id = _compute_toolset_id(refreshed_specs)
                prev_id = str(runtime_ns.get("toolset_id") or "")
                prev_specs = runtime_ns.get("tool_specs")
                if refreshed_id != prev_id or not isinstance(prev_specs, list):
                    runtime_ns["tool_specs"] = refreshed_specs
                    runtime_ns["toolset_id"] = refreshed_id
                    runtime_ns.setdefault("allowed_tools", allow)
        except Exception:
            pass

        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1

        iteration = int(scratchpad.get("iteration", 0) or 0) + 1
        if iteration > max_iterations:
            return StepPlan(node_id="reason", next_node="max_iterations")

        scratchpad["iteration"] = iteration
        limits["current_iteration"] = iteration

        task = str(context.get("task", "") or "")

        # Hook-queued steering folds into the durable inbox HERE (the one
        # drain point), so hook steer and gateway inject_guidance are one
        # mechanism observed by the same events.
        _fold_hook_steering(runtime_ns)
        guidance = _drain_inbox(runtime_ns)
        if guidance:
            # Drained guidance joins the durable transcript as a user interjection (maintainer
            # ruling 2026-07-09): the previous one-shot ephemeral tail was visible to exactly one
            # LLM call, so a final answer written any cycle later re-anchored on the original task
            # and dropped the correction. Append-only, so the cached prefix stays intact (0212);
            # user,user adjacency (guidance before the first assistant turn, or across
            # parse-retry cycles) is repaired at the payload boundary by _sanitize_llm_messages.
            context["messages"].append(
                _new_message(
                    ctx,
                    role="user",
                    content=f"[Operator guidance — this amends the task; the final answer must satisfy it]\n{guidance}",
                    metadata={"kind": "operator_guidance"},
                )
            )
            # The in-loop "message received" listen point (fleet seam): fires
            # whenever the reason boundary consumes delivered guidance.
            emit("inbox_drained", {"chars": len(guidance), "iteration": iteration})
        messages_view = list(context.get("messages") or [])
        req = logic.build_request(
            task=task,
            messages=messages_view,
            guidance="",
            iteration=iteration,
            max_iterations=max_iterations,
            vars=run.vars,
        )

        emit("reason", {"iteration": iteration, "max_iterations": max_iterations, "has_guidance": bool(guidance)})

        payload: Dict[str, Any] = {"prompt": ""}
        sanitized_messages = _sanitize_llm_messages(messages_view)
        if sanitized_messages:
            payload["messages"] = sanitized_messages
        else:
            # Ensure LLM_CALL contract is satisfied even for one-shot runs where callers
            # provide only `context.task` and no `context.messages`.
            task_text = str(task or "").strip()
            if task_text:
                payload["prompt"] = task_text
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media

        tool_specs = runtime_ns.get("tool_specs") if isinstance(runtime_ns, dict) else None
        if isinstance(tool_specs, list) and tool_specs:
            payload["tools"] = list(tool_specs)

        sys_base = str(req.system_prompt or "").strip()
        sys = _compose_system_prompt(runtime_ns, base=sys_base)
        # Prompt-prefix cache stability (0212) + context fidelity (0213):
        # Do NOT append the per-cycle scratchpad to the system prompt. It mutated the cached prefix
        # every iteration (defeating prompt caching) and double-carried observations already present
        # in `context.messages`. The model's reasoning now lives in the transcript (assistant
        # tool-call messages carry their `content`), so the system prompt stays byte-stable across
        # iterations. The durable `scratchpad.cycles` record remains for host-side observability.
        if sys:
            payload["system_prompt"] = sys

        # Volatile per-call state (loop position) rides a TRAILING ephemeral message so it never
        # enters the cached prefix (0212). This is analogous to how the runtime grounding envelope
        # is kept out of the stable prefix. Guidance no longer rides here: it is a durable
        # transcript message (see the drain above).
        tail_parts: list[str] = []
        tail_parts.append(f"[loop] iteration {int(iteration)} of {int(max_iterations)}.")
        plan_text = scratchpad.get("plan") if isinstance(scratchpad, dict) else None
        if isinstance(plan_text, str) and plan_text.strip():
            # Bound the rendered plan so a pathological (or runaway) plan cannot balloon every
            # subsequent request. The full plan stays durable in scratchpad; this is a display cap.
            plan_render = plan_text.strip()
            _plan_cap = 4000
            if len(plan_render) > _plan_cap:
                #[WARNING:TRUNCATION] bounded plan render in the trailing loop message
                plan_render = plan_render[:_plan_cap].rstrip() + f"\n… (plan truncated, {len(plan_text.strip()):,} chars total)"
            tail_parts.append(f"[plan]\n{plan_render}")
        tail_text = "\n\n".join(tail_parts).strip()
        if tail_text and isinstance(payload.get("messages"), list):
            msgs_out = list(payload["messages"])
            # Adjacency guard (Critic-3, 2026-07-09): when the payload already ends with a USER
            # message (first turn: the task; post-ask_user turns: the user's reply), MERGE the
            # volatile tail into it instead of appending a second consecutive user message —
            # alternation-strict chat templates (Mistral/Gemma-class) reject user,user with a
            # 400, and a separate trailing banner also steals the grounding envelope's
            # "last user message" slot from the real task. When the payload ends with
            # assistant/tool turns (the common tool-loop shape), the tail stays a separate
            # trailing message exactly as before (no adjacency, cache prefix untouched).
            if msgs_out and isinstance(msgs_out[-1], dict) and msgs_out[-1].get("role") == "user":
                last = dict(msgs_out[-1])
                last["content"] = f"{str(last.get('content') or '').rstrip()}\n\n{tail_text}"
                msgs_out[-1] = last
            else:
                # Structural volatile marker (B1 pair, code c971 → agent c978 →
                # runtime c986): runtime's llm_client EXCLUDES flagged messages
                # from the prompt-cache fingerprint sequence and STRIPS the key
                # before any provider/SDK sees it — so the per-cycle tail no
                # longer forces a full local-cache re-prefill. The merged branch
                # above cannot carry the flag (it holds the real task); that leg
                # dies with runtime's B3 boundary-merge.
                msgs_out.append({"role": "user", "content": tail_text, "volatile": True})
            payload["messages"] = msgs_out

        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        _apply_llm_payload_extras(runtime_ns, payload)

        params: Dict[str, Any] = {}
        max_out = _max_output_tokens(runtime_ns, limits)
        if isinstance(max_out, int) and max_out > 0:
            params["max_tokens"] = max_out
        # Tool calling is formatting-sensitive; bias toward a lower temperature when tools are present,
        # unless the caller explicitly sets `_runtime.temperature`.
        default_temp = 0.2 if isinstance(tool_specs, list) and tool_specs else 0.7
        payload["params"] = runtime_llm_params(runtime_ns, extra=params, default_temperature=default_temp)

        return StepPlan(
            node_id="reason",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.llm_response"),
            next_node="parse",
        )

    def parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)
        response = temp.get("llm_response", {})

        # Accumulate entity-runtime result-boundary captures across the turn's
        # iterations (G1 write direction, runtime b8b8c78): `diary_entries` (word-free
        # metadata) and `act_only_warnings` ride EACH LLM result and would be lost when
        # `_temp.llm_response` is overwritten next iteration — but the embedding visit
        # workflow's ELECT/FORM nodes need ALL of them (mid-loop elections included:
        # the model may elect, then call a tool, then answer). Keys are the entity
        # runtime's declared result contract, never guessed. Absent = untouched.
        if isinstance(response, dict) and (response.get("diary_entries") or response.get("act_only_warnings")):
            captures = temp.get("turn_captures")
            if not isinstance(captures, dict):
                captures = {"diary_entries": [], "act_only_warnings": []}
                temp["turn_captures"] = captures
            for capture_key in ("diary_entries", "act_only_warnings"):
                vals = response.get(capture_key)
                if isinstance(vals, list) and vals:
                    captures[capture_key] = list(captures.get(capture_key) or []) + list(vals)

        content, tool_calls = logic.parse_response(response)
        finish_reason = ""
        if isinstance(response, dict):
            fr = response.get("finish_reason")
            finish_reason = str(fr or "").strip().lower() if fr is not None else ""

        cycle_i = int(scratchpad.get("iteration", 0) or 0)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        reasoning_text = ""
        try:
            if isinstance(response, dict):
                rc = response.get("reasoning")
                if rc is None:
                    rc = response.get("reasoning_content")
                reasoning_text = str(rc or "")
        except Exception:
            reasoning_text = ""
        emit(
            "parse",
            {
                "iteration": cycle_i,
                "max_iterations": max_iterations,
                "has_tool_calls": bool(tool_calls),
                "content": str(content or ""),
                "reasoning": reasoning_text,
            },
        )
        cycle: Dict[str, Any] = {"i": cycle_i, "thought": content, "tool_calls": [], "observations": []}
        cycles = scratchpad.get("cycles")
        if isinstance(cycles, list):
            cycles.append(cycle)
        else:
            scratchpad["cycles"] = [cycle]

        if tool_calls:
            cycle["tool_calls"] = [tc.__dict__ for tc in tool_calls]

            # Loop guard: some models may repeat the exact same tool calls (including side effects)
            # even after receiving successful observations. Skip executing duplicates to avoid
            # repeatedly overwriting files or re-running commands.
            try:
                side_effect_tools = {
                    "write_file",
                    "edit_file",
                    "execute_command",
                    # Comms tools (side-effectful; avoid duplicate sends).
                    "send_email",
                    "send_whatsapp_message",
                    "send_telegram_message",
                    "send_telegram_artifact",
                }
                has_side_effect = any(
                    isinstance(getattr(tc, "name", None), str) and str(getattr(tc, "name") or "").strip() in side_effect_tools
                    for tc in tool_calls
                )

                if has_side_effect:
                    cycles_list = scratchpad.get("cycles")
                    prev_cycle: Optional[Dict[str, Any]] = None
                    if isinstance(cycles_list, list) and len(cycles_list) >= 2:
                        for c in reversed(cycles_list[:-1]):
                            if not isinstance(c, dict):
                                continue
                            prev_tcs = c.get("tool_calls")
                            if isinstance(prev_tcs, list) and prev_tcs:
                                prev_cycle = c
                                break

                    def _cycle_fps(c: Dict[str, Any]) -> list[str]:
                        tcs2 = c.get("tool_calls")
                        if not isinstance(tcs2, list) or not tcs2:
                            return []
                        fps: list[str] = []
                        for tc in tcs2:
                            if not isinstance(tc, dict):
                                continue
                            fps.append(_tool_call_fingerprint(tc.get("name", ""), tc.get("arguments")))
                        return fps

                    def _cycle_obs_all_ok(c: Dict[str, Any]) -> bool:
                        obs2 = c.get("observations")
                        if not isinstance(obs2, list) or not obs2:
                            return False
                        for o in obs2:
                            if not isinstance(o, dict):
                                return False
                            if o.get("success") is not True:
                                return False
                        return True

                    if prev_cycle is not None and _cycle_obs_all_ok(prev_cycle):
                        prev_fps = _cycle_fps(prev_cycle)
                        cur_fps = [_tool_call_fingerprint(tc.name, tc.arguments) for tc in tool_calls]
                        if prev_fps and prev_fps == cur_fps:
                            _push_inbox(
                                runtime_ns,
                                "You are repeating the exact same tool calls as the previous cycle, and they already succeeded.\n"
                                "Do NOT execute them again (to avoid duplicate side effects).\n"
                                "Instead, use the existing tool outputs and provide the final answer with NO tool calls.",
                            )
                            emit("parse_repeat_tool_calls", {"cycle": cycle_i, "count": len(tool_calls)})
                            temp["pending_tool_calls"] = []
                            return StepPlan(node_id="parse", next_node="reason")
            except Exception:
                pass

            # Keep tool transcript in context for OpenAI-compatible tool calling.
            # Context fidelity (0213): keep the model's OWN reasoning in the durable transcript
            # instead of dropping it (content="") and re-surfacing only a truncated, last-6-cycle
            # copy via the system prompt. Providers accept assistant messages carrying BOTH content
            # and tool_calls; this preserves multi-step coherence across long runs. The durable
            # scratchpad.cycles record is kept for host-side observability only.
            context["messages"].append(
                _new_assistant_message_with_tool_calls(
                    ctx,
                    content=str(content or ""),
                    tool_calls=tool_calls,
                    metadata={"kind": "tool_calls", "cycle": cycle_i},
                )
            )
            temp["pending_tool_calls"] = [tc.__dict__ for tc in tool_calls]
            emit("parse_tool_calls", {"count": len(tool_calls)})
            return StepPlan(node_id="parse", next_node="act")

        # If the model hit an output limit, treat the step as incomplete and continue.
        if finish_reason in {"length", "max_tokens"}:
            _push_inbox(
                runtime_ns,
                "Your previous response hit an output token limit before producing a complete tool call.\n"
                "Retry now: emit ONLY the next tool call(s) needed to make progress.\n"
                "Keep tool call arguments small (avoid large file contents / giant JSON blobs) to prevent tool-call truncation.\n"
                "For large files, create a small skeleton first, then refine via multiple smaller edits/tool calls.\n"
                "Do not write a long plan before tool calls.",
            )
            emit("parse_retry_truncated", {"cycle": cycle_i})
            return StepPlan(node_id="parse", next_node="reason")

        if not isinstance(content, str) or not content.strip():
            _push_inbox(runtime_ns, "Your previous response was empty. Continue the task.")
            emit("parse_retry_empty", {"cycle": cycle_i})
            return StepPlan(node_id="parse", next_node="reason")

        # Followthrough heuristic: retry when the model claims it will take actions but emits no tool calls.
        # Default ON (disable with `_runtime.check_plan=false`).
        raw_check_plan = runtime_ns.get("check_plan") if isinstance(runtime_ns, dict) else None
        check_plan = True if raw_check_plan is None else _boolish(raw_check_plan)
        if check_plan and cycle_i < max_iterations and _looks_like_deferred_action(content):
            _push_inbox(
                runtime_ns,
                "You said you would take an action, but you did not call any tools.\n"
                "If you need to act, call the next tool now (emit ONLY the next tool call(s)).\n"
                "If you are already done, provide the final answer with NO tool calls.",
            )
            emit("parse_retry_plan_only", {"cycle": cycle_i})
            return StepPlan(node_id="parse", next_node="reason")

        # Final answer candidate. Before stopping, optionally run a verification pass (0217):
        # a strict self-critique that can send the loop back to `act` with concrete next steps if
        # the task is not actually complete. Gated on `_runtime.review_mode` (default off at the
        # workflow level; ReactAgent enables it by default via _runtime).
        answer = str(content).strip()
        temp["final_answer"] = answer
        emit("parse_final", {"cycle": cycle_i})
        return StepPlan(node_id="parse", next_node="maybe_review")

    def act_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)

        pending = temp.get("pending_tool_calls", [])
        if not isinstance(pending, list):
            pending = []

        cycle_i = int(scratchpad.get("iteration", 0) or 0)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1

        tool_queue: list[Dict[str, Any]] = []
        for idx, tc in enumerate(pending):
            if isinstance(tc, ToolCall):
                d = tc.__dict__
            elif isinstance(tc, dict):
                d = dict(tc)
            else:
                continue
            if "call_id" not in d or not d.get("call_id"):
                d["call_id"] = str(idx)
            tool_queue.append(d)

        if not tool_queue:
            temp["pending_tool_calls"] = []
            return StepPlan(node_id="act", next_node="reason")

        allow = _effective_allowlist(runtime_ns)

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

            # `update_plan` is an always-available control-flow primitive (no external effect); it is
            # not an allowlisted external tool, so it bypasses the allowlist gate.
            if name and name not in allow and name != "update_plan":
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
                    msgs.append(
                        _new_message(ctx, role="assistant", content=f"[Agent question]: {question}", metadata={"kind": "ask_user_prompt"})
                    )

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

            if name == "update_plan":
                # Schema-only planning tool (0217): persist a structured checklist/plan to the
                # scratchpad. No effect is issued; the plan is rendered at the message TAIL on the
                # next reason step (cache-safe per 0212). This gives long tasks a stable anchor.
                plan_value = args.get("plan")
                if isinstance(plan_value, list):
                    plan_text = "\n".join(f"- {str(item)}" for item in plan_value if str(item).strip())
                else:
                    plan_text = str(plan_value or "").strip()
                scratchpad["plan"] = plan_text
                explanation = str(args.get("explanation") or "").strip()
                if explanation:
                    scratchpad["plan_explanation"] = explanation
                emit("update_plan", {"has_plan": bool(plan_text)})
                temp["tool_results"] = {
                    "results": [
                        {
                            "call_id": str(tc.get("call_id") or ""),
                            "name": "update_plan",
                            "success": True,
                            "output": "Plan updated." if plan_text else "Plan cleared.",
                            "error": None,
                        }
                    ]
                }
                return StepPlan(node_id="act", next_node="observe")

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
                emit("memory_compact", {"preserve_recent": payload.get("preserve_recent"), "mode": payload.get("compression_mode")})
                return StepPlan(
                    node_id="act",
                    effect=Effect(type=EffectType.MEMORY_COMPACT, payload=payload, result_key="_temp.tool_results"),
                    next_node="observe",
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
                    child_allow = _normalize_allowlist(tools_raw)

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

                # Child iteration budget (agency-caps ruling 2026-07-11: default caps are
                # 20; below-20 is an operator's explicit choice, never a code default).
                # The old hardcoded 10 was a fear-shaped number — a delegated research
                # subtask is exactly where iterate-until-satisfied matters. The child
                # inherits the PARENT's budget (a sub-agent is not a lesser agent), with
                # the ruled 20 as the floor guard against a parent that was itself
                # narrowed below the ruling. An explicit `max_iterations` tool arg wins.
                child_iterations: int
                raw_child_iters = args.get("max_iterations")
                try:
                    child_iterations = int(raw_child_iters) if raw_child_iters is not None else 0
                except Exception:
                    child_iterations = 0
                if child_iterations < 1:
                    child_iterations = max(int(max_iterations or 0), 20)

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

                payload = {
                    "workflow_id": str(getattr(run, "workflow_id", "") or "react_agent"),
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

            # Unknown builtin: continue.
            return StepPlan(node_id="act", next_node="act" if temp.get("pending_tool_calls") else "reason")

        batch: List[Dict[str, Any]] = []
        for tc in tool_queue:
            if _is_builtin(tc):
                break
            batch.append(tc)

        remaining = tool_queue[len(batch) :]
        temp["pending_tool_calls"] = list(remaining)

        formatted_calls: List[Dict[str, Any]] = []
        for tc in batch:
            emit(
                "act",
                {
                    "iteration": cycle_i,
                    "max_iterations": max_iterations,
                    "tool": tc.get("name", ""),
                    "args": tc.get("arguments", {}),
                    "call_id": str(tc.get("call_id") or ""),
                },
            )
            formatted_calls.append(
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", {}), "call_id": str(tc.get("call_id") or "")}
            )

        return StepPlan(
            node_id="act",
            effect=Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": formatted_calls, "allowed_tools": list(allow)}, result_key="_temp.tool_results"),
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
            # Verification budget is PER-ANSWER, not run-lifetime (mirror CodeAct, which resets
            # after tools execute). Any tool activity — model-issued or verifier-forced — means the
            # next final answer is a new claim that deserves its own review rounds. Without this,
            # a long run stops being verified after the first `review_max_rounds` checks.
            scratchpad["review_count"] = 0

        # Attach observations to the most recent cycle.
        cycles = scratchpad.get("cycles")
        last_cycle: Optional[Dict[str, Any]] = None
        if isinstance(cycles, list):
            for c in reversed(cycles):
                if isinstance(c, dict) and int(c.get("i") or -1) == int(scratchpad.get("iteration") or -1):
                    last_cycle = c
                    break

        def _display(v: Any) -> str:
            if isinstance(v, dict):
                rendered = v.get("rendered")
                if isinstance(rendered, str) and rendered.strip():
                    return rendered.strip()
            return "" if v is None else str(v)

        act_only_names = _act_only_tool_names()

        obs_list: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")

            # Act-only results (frozen seam spec, a2a 0013 v2 §2): the durable transcript,
            # scratchpad cycles, and emit lane carry the ACT-FRAME REFERENCE only — never
            # tool-surfaced content. Handler-authored refs are honored unconditionally;
            # declared act-only tools additionally get fail-safe rendering when a handler
            # misbehaves (raw output is suppressed, loudly, before it becomes permanent).
            frame = _act_only_frame_from_output(output)
            if frame is None and name in act_only_names:
                if success and isinstance(output, dict):
                    frame = dict(output)
                elif not success:
                    # Failure diagnostics ride the handler's ERROR channel by contract
                    # (a refused/failed act-only read returns no content); raw output is
                    # never rendered for a declared act-only tool.
                    frame = {"tool": name, "error": str(error or "").strip() or "act-only tool call failed"}
                else:
                    frame = {
                        "tool": name,
                        "error": "non-reference output from an act-only tool suppressed at render",
                        "warning": "#FALLBACK",
                    }
            if frame is not None:
                frame.setdefault("tool", name)
                # Ref shape only for frames that reference book content; records-of-acts
                # without an entry_id render as labeled non-ref text — inert to runtime's
                # send-time dereference pass, which loudly fails the LLM call on any
                # unresolvable ref (durable message -> a wedged run otherwise).
                if _act_only_frame_is_dereferenceable(frame):
                    rendered = _act_only_ref_content(frame)
                else:
                    rendered = _act_only_record_content(frame)
                emit("observe", {"tool": name, "success": success, "result": rendered})
                context["messages"].append(
                    _new_message(
                        ctx,
                        role="tool",
                        content=rendered,
                        metadata={"name": name, "call_id": r.get("call_id"), "success": success, "act_only": True},
                    )
                )
                obs_list.append(
                    {
                        "call_id": r.get("call_id"),
                        "name": name,
                        "success": success,
                        "output": {_ACT_ONLY_KEY: dict(frame)},
                        "error": error,
                        "rendered": rendered,
                    }
                )
                continue

            display = _display(output)
            if not success:
                display = _display(output) if isinstance(output, dict) else str(error or output)
            rendered = logic.format_observation(name=name, output=display, success=success)
            emit("observe", {"tool": name, "success": success, "result": rendered})

            context["messages"].append(
                _new_message(
                    ctx,
                    role="tool",
                    content=rendered,
                    metadata={"name": name, "call_id": r.get("call_id"), "success": success},
                )
            )

            obs_list.append(
                {
                    "call_id": r.get("call_id"),
                    "name": name,
                    "success": success,
                    "output": output,
                    "error": error,
                    "rendered": rendered,
                }
            )

        if last_cycle is not None:
            last_cycle["observations"] = obs_list

        temp.pop("tool_results", None)
        pending = temp.get("pending_tool_calls", [])
        if isinstance(pending, list) and pending:
            return StepPlan(node_id="observe", next_node="act")
        temp["pending_tool_calls"] = []
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, _, _, temp, _ = ensure_react_vars(run)
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

    def _review_truncate(text: str, *, max_chars: int) -> str:
        s = str(text or "")
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        suffix = f"\n… (truncated, {len(s):,} chars total)"
        keep = max_chars - len(suffix)
        if keep < 200:
            # Even at a tiny bound, never emit an UNMARKED slice (ADR-0026). Keep a short marker.
            keep = max(0, max_chars - 1)
            suffix = "…"
        #[WARNING:TRUNCATION] bounded verifier transcript blocks for prompt reconstruction
        return s[:keep].rstrip() + suffix

    def maybe_review_node(run: RunState, ctx) -> StepPlan:
        _, scratchpad, runtime_ns, _, _ = ensure_react_vars(run)

        raw_review = runtime_ns.get("review_mode") if isinstance(runtime_ns, dict) else None
        review_mode = _boolish(raw_review) if raw_review is not None else False
        if not review_mode:
            return StepPlan(node_id="maybe_review", next_node="done")

        try:
            max_rounds = int(runtime_ns.get("review_max_rounds", 1) or 0)
        except (TypeError, ValueError):
            max_rounds = 1
        if max_rounds < 0:
            max_rounds = 0

        try:
            count = int(scratchpad.get("review_count") or 0)
        except (TypeError, ValueError):
            count = 0
        if count >= max_rounds:
            return StepPlan(node_id="maybe_review", next_node="done")

        scratchpad["review_count"] = count + 1
        return StepPlan(node_id="maybe_review", next_node="review")

    def review_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)
        task = str(context.get("task", "") or "")
        plan = scratchpad.get("plan")
        plan_text = str(plan).strip() if isinstance(plan, str) and plan.strip() else "(no plan)"
        answer = str(temp.get("final_answer") or "").strip()

        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                tool_msgs.append(_review_truncate(content.strip(), max_chars=2000))
            if len(tool_msgs) >= 8:
                break
        tool_msgs.reverse()
        observations = "\n\n".join(tool_msgs) if tool_msgs else "(no tool outputs)"

        allow = _effective_allowlist(runtime_ns)
        prompt = (
            "You are a verifier. Review whether the user's request has been fully satisfied.\n"
            "Be strict: only count actions that are supported by the tool outputs.\n"
            "If anything is missing, propose the NEXT ACTIONS.\n"
            "Prefer returning `next_tool_calls` over `next_prompt`.\n"
            "Return JSON ONLY.\n\n"
            f"User request:\n{task}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Proposed final answer:\n{_review_truncate(answer, max_chars=4000)}\n\n"
            f"Tool outputs:\n{observations}\n\n"
            f"Allowed tools:\n{', '.join(allow) if allow else '(none)'}\n\n"
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
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
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
            "params": runtime_llm_params(runtime_ns, extra={}, default_temperature=0.2),
            # Review-failure containment (backlog 0027, c1128 incident class): the
            # verifier is a verification AID — a failed verifier call (structured
            # validation, provider error, anything) must never kill a run whose
            # `_temp.final_answer` already holds a valid answer. The runtime's
            # opt-in absorption converts the FINAL failed outcome (after its own
            # retries) into {"ok": False, "absorbed_failure": <error>} at
            # result_key and continues to review_parse, which degrades to
            # accept-with-#FALLBACK. Runtimes without the absorption mechanism
            # ignore this key — behavior there stays exactly as before.
            "_absorb_failure": True,
        }
        media = extract_media_from_context(context)
        if media:
            payload["media"] = media
        sys = _compose_system_prompt(runtime_ns, base="")
        if sys:
            payload["system_prompt"] = sys
        eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
        eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
        if isinstance(eff_provider, str) and eff_provider.strip():
            payload["provider"] = eff_provider.strip()
        if isinstance(eff_model, str) and eff_model.strip():
            payload["model"] = eff_model.strip()

        return StepPlan(
            node_id="review",
            effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.review_llm_response"),
            next_node="review_parse",
        )

    def review_parse_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        resp = temp.get("review_llm_response", {})
        if not isinstance(resp, dict):
            resp = {}

        absorbed = resp.get("absorbed_failure")
        if absorbed is not None:
            # The absorbed-failure record is the runtime's `_absorb_failure`
            # shape ({"ok": False, "absorbed_failure": <error>}) — the verifier
            # call failed terminally. Verifier failure must never be worse than
            # no verifier: accept the held final answer and complete, loudly
            # (#FALLBACK marker in scratchpad/report + a dedicated emit). The
            # ledger already recorded the effect failure honestly.
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
        next_prompt_text = str(data.get("next_prompt") or "").strip()
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
            # BUG FIX (adversarial review): forced tool calls MUST be preceded by an assistant
            # tool-calls message in the transcript, or the tool observations become orphans and
            # strict OpenAI-compatible providers 400 the next call ("tool message must be a response
            # to a preceding tool_calls"). Synthesize that assistant message here, mirroring
            # parse_node, with explicit call_ids shared by the assistant message AND the pending
            # calls so the ids line up when observe_node writes the tool results.
            synthesized: list[ToolCall] = []
            base = f"review_{int(scratchpad.get('iteration', 0) or 0)}_{hashlib.sha256(json.dumps(next_tool_calls, sort_keys=True, default=str).encode()).hexdigest()[:8]}"
            for i, item in enumerate(next_tool_calls):
                synthesized.append(
                    ToolCall(name=str(item.get("name") or ""), arguments=dict(item.get("arguments") or {}), call_id=f"{base}_{i}")
                )
            context["messages"].append(
                _new_assistant_message_with_tool_calls(
                    ctx,
                    content="",
                    tool_calls=synthesized,
                    metadata={"kind": "tool_calls", "source": "review"},
                )
            )
            temp["pending_tool_calls"] = [tc.__dict__ for tc in synthesized]
            emit("review_tool_calls", {"count": len(synthesized)})
            return StepPlan(node_id="review_parse", next_node="act")

        # Incomplete but no actionable tool calls. The old "nudge then re-review" was a no-op (the
        # re-review payload was byte-identical, so the runtime idempotency layer replayed the first
        # verdict) AND leaked the nudge into the MAIN model's guidance tail. Instead: if the verifier
        # gave a next_prompt, steer the main agent with it (bounded by the run-lifetime review
        # budget, which does not reset on this path); otherwise accept the answer.
        if next_prompt_text:
            _push_inbox(runtime_ns, f"[Review] {next_prompt_text}")
            return StepPlan(node_id="review_parse", next_node="reason")
        return StepPlan(node_id="review_parse", next_node="done")

    def done_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, limits = ensure_react_vars(run)
        task = str(context.get("task", "") or "")
        answer = str(temp.get("final_answer") or "No answer provided")

        # `handed_off` disambiguates composition (visit turns: the RUN continues
        # past this turn's final answer) from true run completion for hook
        # consumers reading turn_end/final_answer as terminal.
        emit("done", {"answer": answer, "handed_off": bool(final_next_node)})

        messages = context.get("messages")
        if isinstance(messages, list):
            last = messages[-1] if messages else None
            last_role = last.get("role") if isinstance(last, dict) else None
            last_content = last.get("content") if isinstance(last, dict) else None
            if last_role != "assistant" or str(last_content or "") != answer:
                messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer"}))

        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or 0)
        report = _render_final_report(task, scratchpad)

        output = {
            "answer": answer,
            "report": report,
            "iterations": iterations,
            "messages": list(context.get("messages") or []),
            "scratchpad": dict(scratchpad),
        }
        if final_next_node:
            # Composition handoff (visit TURN chain): the turn is finished but the RUN
            # continues — the seam node reads _temp.react_output / _temp.final_answer.
            temp["react_output"] = output
            return StepPlan(node_id="done", next_node=final_next_node)
        _discard_hook_steering_at_terminal()
        return StepPlan(node_id="done", complete_output=output)

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        emit("max_iterations", {"iterations": max_iterations})

        # Deterministic conclusion: when we hit the iteration cap, run one tool-free LLM call
        # to synthesize a final report + next steps while the scratchpad is still in context.
        resp = temp.get("max_iterations_llm_response")
        if not isinstance(resp, dict):
            _fold_hook_steering(runtime_ns)
            drained_guidance = _drain_inbox(runtime_ns)
            conclude_directive = (
                "You have reached the maximum allowed ReAct iterations.\n"
                "You MUST stop using tools now and provide a best-effort conclusion.\n\n"
                "In your response, include:\n"
                "1) A concise progress report (what you did + key observations).\n"
                "2) The best current answer you can give based on evidence.\n"
                "3) Remaining uncertainties / missing info.\n"
                "4) Next steps: exact actions to finish (files to inspect/edit, commands/tools to run, what to look for).\n\n"
                "Rules:\n"
                "- Do NOT call tools.\n"
                "- Do NOT output tool-call markup (e.g. <tool_call>...</tool_call>).\n"
                "- Do NOT mention internal scratchpads; just present the report.\n"
                "- Prefer bullet points and concrete next steps."
            )

            task = str(context.get("task", "") or "")
            messages_view = list(context.get("messages") or [])

            req = logic.build_request(
                task=task,
                messages=messages_view,
                guidance="",
                iteration=max_iterations,
                max_iterations=max_iterations,
                vars=run.vars,
            )

            payload: Dict[str, Any] = {"prompt": ""}
            sanitized_messages = _sanitize_llm_messages(messages_view)
            if sanitized_messages:
                payload["messages"] = sanitized_messages
            else:
                task_text = str(task or "").strip()
                if task_text:
                    payload["prompt"] = task_text

            media = extract_media_from_context(context)
            if media:
                payload["media"] = media

            sys_base = str(req.system_prompt or "").strip()
            sys = _compose_system_prompt(runtime_ns, base=sys_base)
            block_parts: list[str] = []
            if drained_guidance:
                block_parts.append(f"Host guidance:\n{drained_guidance}")
            block_parts.append(conclude_directive)
            sys = (f"{sys.rstrip()}\n\n## Max iterations reached\n" + "\n\n".join(block_parts)).strip()
            scratch_txt = _render_cycles_for_conclusion_prompt(scratchpad)
            if scratch_txt:
                sys = f"{sys.rstrip()}\n\n## Scratchpad (ReAct cycles so far)\n{scratch_txt}".strip()
            if sys:
                payload["system_prompt"] = sys

            eff_provider = provider if isinstance(provider, str) and provider.strip() else runtime_ns.get("provider")
            eff_model = model if isinstance(model, str) and model.strip() else runtime_ns.get("model")
            if isinstance(eff_provider, str) and eff_provider.strip():
                payload["provider"] = eff_provider.strip()
            if isinstance(eff_model, str) and eff_model.strip():
                payload["model"] = eff_model.strip()

            params: Dict[str, Any] = {}
            max_out = _max_output_tokens(runtime_ns, limits)
            if isinstance(max_out, int) and max_out > 0:
                params["max_tokens"] = max_out
            payload["params"] = runtime_llm_params(runtime_ns, extra=params, default_temperature=0.2)

            return StepPlan(
                node_id="max_iterations",
                effect=Effect(type=EffectType.LLM_CALL, payload=payload, result_key="_temp.max_iterations_llm_response"),
                next_node="max_iterations",
            )

        # We have a conclusion LLM response. Parse it and complete the run.
        content, tool_calls = logic.parse_response(resp)
        answer = str(content or "").strip()
        temp.pop("max_iterations_llm_response", None)

        # If the model still emitted tool calls, or if it leaked tool-call markup as plain text,
        # retry once with a stricter instruction.
        tool_tags = _contains_tool_call_markup(answer)
        if tool_calls or tool_tags:
            retries = int(temp.get("max_iterations_conclude_retries", 0) or 0)
            if retries < 1:
                temp["max_iterations_conclude_retries"] = retries + 1
                _push_inbox(
                    runtime_ns,
                    "You are out of iterations and tool use is disabled.\n"
                    "Return ONLY the final report and next steps as plain text.\n"
                    "Do NOT include any tool calls or tool-call markup (e.g. <tool_call>...</tool_call>).",
                )
                return StepPlan(node_id="max_iterations", next_node="max_iterations")
            # Last resort: strip any leaked tool markup so we don't persist it as the final answer.
            answer = _strip_tool_call_markup(answer).strip()

        if not answer:
            # Fallback: avoid returning the last tool observation as the "answer".
            # Provide a deterministic report so users don't lose scratchpad context.
            scratch_view = _render_cycles_for_conclusion_prompt(scratchpad)
            parts = [
                "Max iterations reached.",
                "I could not produce a final assistant response in time.",
            ]
            if scratch_view:
                parts.append("## Progress (from scratchpad)\n" + scratch_view)
            parts.append(
                "## Next steps\n"
                "- Increase `max_iterations` and rerun, or use `/conclude` earlier to force a wrap-up.\n"
                "- If you need me to continue, re-run with a higher iteration budget and I will pick up from the report above."
            )
            answer = "\n\n".join(parts).strip()

        # Persist final answer into the conversation history (so it shows up in /history and seeds next runs).
        messages = context.get("messages")
        if isinstance(messages, list):
            last = messages[-1] if messages else None
            last_role = last.get("role") if isinstance(last, dict) else None
            last_content = last.get("content") if isinstance(last, dict) else None
            if last_role != "assistant" or str(last_content or "") != answer:
                messages.append(_new_message(ctx, role="assistant", content=answer, metadata={"kind": "final_answer"}))

        temp["final_answer"] = answer
        report = _render_final_report(str(context.get("task") or ""), scratchpad)

        iterations = int(limits.get("current_iteration", 0) or scratchpad.get("iteration", 0) or max_iterations)
        output = {
            "answer": answer,
            "report": report,
            "iterations": iterations,
            "messages": list(context.get("messages") or []),
            "scratchpad": dict(scratchpad),
        }
        if final_next_node:
            # Composition handoff: budget exhaustion also ends the TURN, not the run —
            # the seam node decides what an out-of-budget visit turn does next.
            temp["react_output"] = output
            return StepPlan(node_id="max_iterations", next_node=final_next_node)
        _discard_hook_steering_at_terminal()
        return StepPlan(node_id="max_iterations", complete_output=output)

    return WorkflowSpec(
        workflow_id=str(workflow_id or "react_agent"),
        entry_node="init",
        nodes={
            node_id: _with_run_context(node_fn)
            for node_id, node_fn in {
                "init": init_node,
                "reason": reason_node,
                "parse": parse_node,
                "act": act_node,
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
