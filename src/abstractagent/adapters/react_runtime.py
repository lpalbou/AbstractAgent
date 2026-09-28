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
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall
from abstractruntime import Effect, EffectType, RunState, StepPlan, WorkflowSpec
from abstractruntime.core.vars import ensure_limits, ensure_namespaces
from abstractruntime.session_history import window_transcript
from abstractruntime.turn_grounding import stamp_user_turn_grounding, strip_turn_grounding

from .generation_params import (
    DELEGATE_SUBSTRATE_KEYS,
    coerce_iterations,
    coerce_verifier_tool_arguments,
    compose_prompt_slots,
    context_usage_warning,
    executor_tool_names,
    guidance_wrapper,
    is_side_effect_tool,
    normalize_thinking,
    prompt_cache_capture,
    resolve_max_iterations,
    runtime_llm_params,
    suppress_loop_tail,
    tool_tags_map,
    verifier_execution_preference,
    verifier_response_schema,
)
from .loop_hooks import LoopHooks, undelivered_inbox_stats
from .media import (
    accumulate_media,
    extract_media_from_context,
    extract_media_from_tool_result,
    media_item_key,
    merge_media_lists,
)
from .tool_failure_hints import diagnose_tool_failure, output_reads_as_error, render_hint_block
from .read_orchestration import (
    advance_last_successful_read_batch,
    detect_nearby_same_file_staircase,
    detect_redundant_same_file_full_reread,
)
from .tool_allowlist import note_pruned_grants
from .transcripts import (
    assistant_tool_calls_payload,
    elide_oversized_content,
    ensure_tool_call_ids,
    extract_reasoning_text,
    parse_content_preview,
    place_loop_tail,
    sanitize_transcript_messages,
    synthetic_call_id,
)
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
    # Shared shape (0011 extraction): one payload builder across the loops.
    tc_payload = assistant_tool_calls_payload(tool_calls)
    if tc_payload:
        msg["tool_calls"] = tc_payload
    return msg


def ensure_react_vars(
    run: RunState,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Ensure namespaced vars exist and migrate legacy flat keys in-place."""

    # Captured BEFORE ensure_limits materializes defaults (0029 #6): a raw
    # `runtime.start(vars={"max_iterations": 100})` used to run capped at 20 —
    # the materialized `_limits.max_iterations=20` beat the caller's explicit
    # legacy value at the resolver (limits wins over scratchpad). Explicit
    # values must win over defaults; the ruled 20 stays the no-value default.
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

    # 0029 #6: when the caller supplied an explicit legacy/flat budget and did
    # NOT set `_limits.max_iterations` itself, seed limits from it — otherwise
    # the materialized default (20) silently wins at the resolver.
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


def _evidence_text(context: Any, *, max_chars: int = 200_000) -> str:
    """What the ENVIRONMENT has told this run so far — user messages, tool
    outputs, system messages. Assistant turns are excluded on purpose: a value
    the model minted and then echoed in its own text is not evidence that the
    value exists, and treating it as such is exactly how `artifact_id="a1"`
    survived twelve calls without anyone noticing it came from nowhere.
    """
    msgs = context.get("messages") if isinstance(context, dict) else None
    if not isinstance(msgs, list):
        return ""
    parts: list[str] = []
    for m in msgs[-80:]:
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "").strip().lower() == "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            parts.append(c)
    joined = "\n".join(parts)
    return joined[-max_chars:] if len(joined) > max_chars else joined


def _tool_names_of(logic_obj: Any) -> list:
    names: list = []
    for t in getattr(logic_obj, "tools", None) or []:
        n = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if n:
            names.append(str(n))
    return names


def _observation_fingerprint(cycle: Any) -> Optional[str]:
    """Identity of what a cycle's tool batch RETURNED, or None when it
    returned nothing (a cycle the guard refused to execute).

    The 0017 detector fingerprints only the CALL, which cannot tell a stuck
    loop from a poll that is making progress: `check_status()` answered
    "10%", then "40%", then "90%" is the same call three times and is not
    stuck. (No ellipsis in this docstring on purpose: the ADR-0026 lossy-site
    guard scans for one next to a slice, and the slice here is a hash prefix.)
    Forensics on the nudge (2026-08-21) put a real polling loop in front of
    it and it fired. Identical call AND identical answer is the honest
    trigger.
    """
    obs = cycle.get("observations") if isinstance(cycle, dict) else None
    if not isinstance(obs, list) or not obs:
        return None
    parts: list[str] = []
    for o in obs:
        if not isinstance(o, dict):
            parts.append("?")
            continue
        body = o.get("error") if o.get("success") is not True else o.get("output")
        if body is None:
            body = o.get("output") if o.get("success") is not True else o.get("error")
        # Hash the body TEXT directly. `_tool_call_fingerprint` takes an
        # ARGUMENTS value and ignores a bare string, so routing observations
        # through it made every answer hash the same — the check silently
        # passed nothing (caught live: a 10%/45%/80% poll still read as
        # "identical answers").
        try:
            body_txt = body if isinstance(body, str) else json.dumps(body, sort_keys=True, default=str)
        except Exception:
            body_txt = str(body)
        parts.append(f"{o.get('name') or 'tool'}|{bool(o.get('success'))}|{body_txt}")
    try:
        return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "obs_fp_err"


def _stop_reason(
    *,
    code: str,
    finished: bool,
    budget_exhausted: bool,
    iterations: int,
    forced: Optional[Dict[str, Any]] = None,
    max_iterations: Optional[int] = None,
    by_operator: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The turn's verdict IN WORDS, authored where the facts are.

    Architecture (operator, 2026-08-21): hosts are thin. AbstractCode's TUI,
    AbstractObserver, the web client, a WhatsApp or Telegram bridge — all of
    them reach the same gateway and must render the same answer to "did it
    finish, and what do I do about it?" without each re-deriving it. The
    client that DID derive it locally got it wrong for two days: it read
    `outcome: "iteration_budget"` alone, missed the additive
    `conclusion_forced` beside it, and told operators to raise a budget that
    still had 38 of 50 iterations unspent.

    Only this node knows the whole truth — the ceiling, the iterations
    actually spent, whether a stuck pattern forced the stop, and whether the
    model was warned first — so this node writes the sentence. `label` is for
    a status line, `headline` and `remedy` for a card. Additive: `outcome`,
    `iterations` and `conclusion_forced` keep their existing shapes and
    meanings for anything already reading them.
    """
    kind = str((forced or {}).get("kind") or "")
    span = int((forced or {}).get("span") or 0)
    nudged = bool((forced or {}).get("nudged"))
    iters_txt = f" after {iterations} iterations" if iterations > 0 else ""

    if code == "final_answer":
        return {
            "code": code,
            "finished": True,
            "budget_exhausted": False,
            "iterations": iterations,
            "label": "done",
            "headline": "",
            "remedy": "",
        }

    if by_operator is not None:
        # Neither a failure nor a budget: the operator asked for the best
        # answer available and got it. Hosts must not dress this as a
        # truncation the agent caused.
        return {
            "code": "operator_conclude",
            "finished": False,
            "budget_exhausted": False,
            "iterations": iterations,
            "label": f"concluded on request{iters_txt}",
            "headline": (
                f"You asked the agent to conclude{iters_txt}, so it stopped work and answered "
                "with what it had."
            ),
            "remedy": "Anything left unfinished is listed in the answer — send it as a follow-up turn.",
        }

    if kind:
        shape = (
            "alternated between the same two tool batches"
            if kind == "oscillation"
            else "repeated the same tool batch"
        )
        times = f" {span} times" if span > 0 else ""
        warned = " It was warned mid-loop and did it again." if nudged else ""
        short = "oscillating tool calls" if kind == "oscillation" else "repeated tool calls"
        return {
            "code": f"stuck_{kind}",
            "finished": False,
            # The budget is NOT what stopped this turn, and saying so is the
            # whole point: the remedy differs.
            "budget_exhausted": False,
            "iterations": iterations,
            "label": f"stopped: {short}{iters_txt}",
            "headline": (
                f"The agent stopped early{iters_txt}: it {shape}{times} without making progress, "
                f"so the loop ended the turn.{warned}"
            ),
            "remedy": (
                "The iteration budget was not the limit, so raising it will not help. Give the agent a "
                "different route to the same goal, or check whether the tool it kept calling can work here at all."
            ),
        }

    ceiling = f" of {max_iterations}" if isinstance(max_iterations, int) and max_iterations > 0 else ""
    return {
        "code": "iteration_budget",
        "finished": False,
        "budget_exhausted": True,
        "iterations": iterations,
        "label": f"stopped: iteration budget{iters_txt}",
        "headline": (
            f"The agent ran out of iterations{iters_txt}{ceiling} and STOPPED — it did not finish."
        ),
        "remedy": "Raise the iteration budget, or send the remaining work as a follow-up turn.",
    }


def _turn_notices(scratchpad: Any) -> list:
    """Caveats about the ANSWER, authored here for the same reason as
    `_stop_reason`: every host shows the same sentence or none of them do."""
    out: list = []
    if isinstance(scratchpad, dict) and scratchpad.get("review_skipped"):
        out.append(
            {
                "code": "review_skipped",
                "severity": "warn",
                "text": (
                    "The verifier pass was requested but did not run — this answer was not checked "
                    "against the tool outputs."
                ),
            }
        )
    return out


def _observations_all_failed(cycle: Any) -> Optional[bool]:
    """True when every observation of this cycle is a failure, None when the
    cycle produced no observations at all (nothing to judge)."""
    obs = cycle.get("observations") if isinstance(cycle, dict) else None
    if not isinstance(obs, list) or not obs:
        return None
    seen = False
    for o in obs:
        if not isinstance(o, dict):
            continue
        seen = True
        if o.get("success") is True:
            body = o.get("output")
            if not isinstance(body, str):
                body = str(o.get("rendered") or "")
            # Same rule as the diagnosis lane: an error sentence in the body
            # is a failure even when the transport flag says otherwise.
            if not output_reads_as_error(body):
                return False
    return True if seen else None


def _streak_key(kind: str, fp_groups: Any) -> str:
    """Stable identity for one stuck pattern (0017 nudge half).

    The nudge fires ONCE per distinct pattern, not once per cycle: a model
    that changes strategy and then gets stuck a DIFFERENT way earns a second
    nudge, while a model that ignores the first one is not spammed with
    copies of it. For an oscillation the identity is the unordered PAIR — the
    "current" batch alternates every cycle, so keying on it alone would nudge
    twice for one loop.
    """
    try:
        raw = str(kind) + "|" + "|".join(",".join(map(str, g)) for g in fp_groups)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return f"{kind}_key_err"


def _repeat_streak_verdict(
    cycles: Any,
    *,
    turn_fence: int,
    threshold: int,
    max_span: int = 24,
) -> Optional[Dict[str, Any]]:
    """Detect a stuck tool-batch pattern ending at the CURRENT cycle (0017 work half).

    Two shapes, both from the backlog item, both deliberately CONSERVATIVE
    (a false positive forces a wrong termination; max_iterations still
    backstops anything this misses):

    - REPEAT: the last `threshold` tool-batches are ALL identical
      (consecutive — interleaved distinct work like A-X1-A-X2 never counts;
      re-reading a file between different edits is legitimate).
    - OSCILLATION: the last four tool-batches alternate strictly between two
      distinct batches (A-B-A-B — the pattern completing twice is decisive).

    Batches are compared by ordered fingerprint lists (the existing
    `_tool_call_fingerprint`); repeat-skipped cycles COUNT as proposals (a
    model insisting on a batch the side-effect guard refused twice is stuck,
    not progressing). Scan never crosses `turn_fence` (c2447 F5: repeating
    yesterday's search in a later visit turn is a relationship, not a loop).

    Returns None, or {"kind": "repeat"|"oscillation", "span": N, "key": K}
    where span is the ACTUAL number of trailing tool-cycles in the pattern
    (>= threshold, not clamped to it — the nudge/hard-stop escalation below
    needs to know how far past the trigger the loop has gone) and key is a
    stable identity for the pattern, so the same loop is nudged once rather
    than once per cycle.
    """
    if threshold < 2 or not isinstance(cycles, list):
        return None
    fence = max(0, turn_fence)
    # Ordered fingerprint lists of tool-proposing cycles, NEWEST FIRST,
    # current cycle included (it is already appended by parse_node).
    fps_newest_first: list[tuple[str, ...]] = []
    obs_newest_first: list[Optional[str]] = []
    obs_failed_newest_first: list[Optional[bool]] = []
    for idx in range(len(cycles) - 1, fence - 1, -1):
        c = cycles[idx]
        if not isinstance(c, dict):
            continue
        tcs = c.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            continue
        fps_newest_first.append(
            tuple(
                _tool_call_fingerprint(tc.get("name", ""), tc.get("arguments"))
                for tc in tcs
                if isinstance(tc, dict)
            )
        )
        obs_newest_first.append(_observation_fingerprint(c))
        obs_failed_newest_first.append(_observations_all_failed(c))
        # Enough history for either shape; stop scanning.
        if len(fps_newest_first) >= max(threshold, 4, int(max_span)):
            break
    if not fps_newest_first or not fps_newest_first[0]:
        return None
    current = fps_newest_first[0]
    # REPEAT: how many trailing batches are identical to the current one AND
    # came back with the same answer. A cycle that returned NOTHING (the
    # guard refused it) is transparent: it neither proves nor breaks the
    # streak. Two executed cycles whose answers DIFFER break it — the loop is
    # still producing new information, whatever the call looks like.
    run = 1
    seen_obs: Optional[str] = None
    answers_differ = False
    all_failed: Optional[bool] = None
    for fps, obs_fp, obs_failed in zip(
        fps_newest_first[1:], obs_newest_first[1:], obs_failed_newest_first[1:]
    ):
        if fps != current:
            break
        if obs_fp is not None:
            if seen_obs is None:
                seen_obs = obs_fp
            elif obs_fp != seen_obs:
                answers_differ = True
                break
        if obs_failed is not None:
            all_failed = obs_failed if all_failed is None else (all_failed and obs_failed)
        run += 1
    if run >= threshold and not answers_differ:
        return {
            "kind": "repeat",
            "span": run,
            "key": _streak_key("repeat", [current]),
            # Did the identical batch ever come back with anything? A streak
            # of refused proposals has no answer to echo in the nudge.
            "answered": seen_obs is not None,
            # Every answer was an ERROR. The model has already been handed a
            # diagnosis of that error (observe_node's first-failure hint), so
            # this streak trips one repeat earlier than an ambiguous one.
            "failed": bool(all_failed),
        }
    # OSCILLATION: strict A-B-A-B… — the trailing alternation, full length.
    if len(fps_newest_first) >= 4:
        b, a = fps_newest_first[0], fps_newest_first[1]
        if a and b and a != b:
            span = 2
            for idx in range(2, len(fps_newest_first)):
                expected = b if idx % 2 == 0 else a
                if fps_newest_first[idx] != expected:
                    break
                span += 1
            if span >= 4:
                return {"kind": "oscillation", "span": span, "key": _streak_key("oscillation", sorted([a, b]))}
    return None


def _stuck_recent_tool_cycles(cycles: Any, *, turn_fence: int, count: int) -> list:
    """The trailing tool-PROPOSING cycles of this turn, newest first."""
    out: list = []
    if not isinstance(cycles, list) or count <= 0:
        return out
    fence = max(0, turn_fence)
    for idx in range(len(cycles) - 1, fence - 1, -1):
        c = cycles[idx]
        if not isinstance(c, dict):
            continue
        tcs = c.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            continue
        out.append(c)
        if len(out) >= count:
            break
    return out


def _stuck_nudge_message(
    cycles: Any,
    *,
    verdict: Dict[str, Any],
    turn_fence: int,
    remaining: Optional[int],
    suppressed: bool = False,
    tool_specs: Any = None,
    available_tools: Any = (),
    evidence_text: str = "",
) -> str:
    """Tell the model it is looping WHILE it can still act on that (0017 nudge half).

    The pre-2026-08-21 behaviour terminated the turn on the first detection, so
    the only place the loop was ever named to the model was the conclusion
    prompt — after its last chance to change course. This message carries the
    three facts it cannot see for itself: the batch it keeps proposing, how
    many times, and what that batch RETURNED each time (the observation text
    was never summarised back at it), plus the consequence of ignoring this.
    """
    kind = str(verdict.get("kind") or "repeat")
    span = int(verdict.get("span") or 0)
    recent = _stuck_recent_tool_cycles(cycles, turn_fence=turn_fence, count=max(2, span))

    if suppressed:
        # Entity/visit lane (c2447 chrome class): the same correction with NO
        # loop vocabulary, no cycle counts and no batch dump — this text rides
        # the conversation the visitor's reply is written from, exactly like
        # the suppressed conclude directive.
        return (
            "That attempt has already been made and returned the same result, so it was not "
            "repeated. Try a different approach, or reply now with what you already have — in "
            "your own words."
        )

    sigs: list[str] = []
    for c in recent[: (1 if kind == "repeat" else 2)]:
        calls = c.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for tc in calls:
            if isinstance(tc, dict):
                sig = _tool_call_signature(str(tc.get("name") or ""), tc.get("arguments"))
                if sig not in sigs:
                    sigs.append(sig)

    # The DIAGNOSIS of the repeated batch, not just its echo: same builder the
    # first failure used, so the escalation says something more than "you did
    # it again" — it re-states which argument is wrong and what to run instead.
    diagnoses: list[str] = []
    seen_sigs: set = set()
    for c in recent:
        obs_here = c.get("observations")
        calls_here = c.get("tool_calls")
        if not isinstance(obs_here, list) or not isinstance(calls_here, list):
            continue
        for o in obs_here:
            if not isinstance(o, dict) or o.get("success") is True:
                continue
            args = {}
            for tc in calls_here:
                if isinstance(tc, dict) and tc.get("call_id") == o.get("call_id"):
                    args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                    break
            out = o.get("output")
            err_text = str(o.get("error") or "")
            if isinstance(out, dict) and str(out.get("rendered") or ""):
                err_text = str(out.get("rendered"))
            elif isinstance(out, str) and out.strip() and not err_text:
                err_text = out
            d = diagnose_tool_failure(
                name=str(o.get("name") or ""),
                arguments=args,
                error_text=err_text,
                tool_specs=tool_specs,
                available_tools=available_tools or [],
                transcript_text=evidence_text,
            )
            if d is None or d["signature"] in seen_sigs:
                continue
            seen_sigs.add(d["signature"])
            diagnoses.append(str(d["text"]))

    obs_lines: list[str] = []
    seen: set = set()
    for c in recent:
        obs_list = c.get("observations")
        if not isinstance(obs_list, list):
            # Malformed scratchpad must not kill a run from inside a guard.
            # Skipped, not swallowed: the nudge simply carries no observation
            # echo, and the batch + count still reach the model.
            continue
        for o in obs_list:
            if not isinstance(o, dict):
                continue
            ok = bool(o.get("success"))
            text = str((o.get("output") if ok else (o.get("error") or o.get("output"))) or "").strip()
            if len(text) > 800:
                # 800, not 300: a tool's refusal often ENDS with the way out
                # ("use read_file / web_search instead"), and clipping the
                # advice out of the echo defeats the nudge. Long dumps are
                # still bounded, and marked.
                #[WARNING:TRUNCATION] bounded observation echo in the stuck nudge (full text stays in the transcript)
                text = f"{text[:799]}… (truncated, {len(text):,} chars total — the full output is above in this conversation)"
            line = f"- {str(o.get('name') or 'tool')}: {'ok' if ok else 'FAILED'} — {text or '(no output)'}"
            if line in seen:
                continue
            seen.add(line)
            obs_lines.append(line)
    extra_obs = 0
    if len(obs_lines) > 6:
        extra_obs = len(obs_lines) - 6
        obs_lines = obs_lines[:6]

    shape = (
        f"you proposed the EXACT SAME tool batch {span} times in a row"
        if kind == "repeat"
        else f"you alternated between the same two tool batches for {span} cycles"
    )
    parts = [
        f"[loop guard] You are stuck: {shape}, and this last one was NOT executed.",
        "",
        "The batch you keep proposing:",
    ]
    parts.extend(f"- {sg}" for sg in (sigs or ["(unavailable)"]))
    if diagnoses:
        parts.append("")
        parts.append("Why it keeps failing:")
        parts.extend(diagnoses[:3])
    elif obs_lines:
        parts.append("")
        parts.append("What it returned:")
        parts.extend(obs_lines)
        if extra_obs:
            parts.append(f"- … and {extra_obs} more observation(s) of the same batch")
    # NO UNPROVABLE CLAIMS. The first cut said "repeating it will not produce
    # a different result", which is not something this guard can know — and
    # for a poll-until-ready tool it is simply false. Live forensics
    # (2026-08-21) caught a model weighing that sentence against its tool's
    # own "poll again with the SAME job_id" and, correctly, believing the
    # tool. State only what is true: the answers have been identical, and
    # this loop has no wait in it.
    lead = (
        f"You have already received that same answer {span} times, and your calls are issued "
        "back-to-back with no wait between them — an identical call now is answered from the "
        "same state."
        if verdict.get("answered")
        else "That batch is not being executed any more."
    )
    parts.extend(
        [
            "",
            f"{lead} Change strategy NOW — do exactly one of:",
            "1) Use a DIFFERENT tool, or the same tool with materially different arguments, to reach the same goal.",
            "2) If what you already have is enough, answer now with NO tool calls.",
            "3) If you are WAITING on something external (a job, a build, a service that is down), STOP and say so "
            "in your answer — repeating the call inside this turn does not make the wait shorter, and the operator "
            "can re-run you later.",
            "4) If the goal is unreachable with the tools available, say so plainly and give the best answer you can from the evidence.",
        ]
    )
    parts.append("")
    if remaining is not None and remaining > 0:
        parts.append(
            f"Do NOT propose that batch again: {remaining} more repetition(s) and this turn will be ended "
            "for you and you will be forced to conclude with whatever you have."
        )
    else:
        parts.append("Do NOT propose that batch again.")
    return "\n".join(parts)


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

# Sight-lane burst bound (c3969 shape A / c4089 ruling): media refs captured
# from tool results ride the NEXT model call one-shot; a pathological batch
# (N captures in one iteration) must not put N images on a single provider
# call. Most-recent wins — the newest capture is the one the model was acting
# on; drops are emitted (never silent). Deliberately a constant, not a knob:
# the bound exists to cap cost blowup, not to be tuned per run.
_PENDING_MEDIA_MAX = 6

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


# The `$act_only` ref-minting helpers that lived here (2026-07-10 seam spec
# a2a 0013) were DELETED under laurent's A ruling (2026-07-20): runtime removed
# the send-time dereference, homes serve plain content, and the diary book
# remains the sole-author surface via the WRITE-boundary capture (runtime's
# wrapper). Tool results render as plain served content — one path, no frames.


_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


def _zero_usage_with_speech(response: Any) -> bool:
    """True when the completion reports ZERO tokens yet carries text.

    The abort marker (operator 2026-08-02, LM Studio + qwen3.6-35b-a3b): a
    generation cut mid-tool-call returns HTTP 200 with the tool-call PREFACE as
    `content`, `tool_calls: []`, `finish_reason: "stop"` and
    `usage: {prompt_tokens: 0, completion_tokens: 0, total_tokens: 0}`. The
    dropped tool call is reported only in the provider's own server log
    ("Failed to generate a tool call … omitted from the response"); the wire
    body carries no error field at all.

    A completion that produced text cannot have consumed zero prompt tokens and
    produced zero completion tokens — that pair is impossible for work that
    actually finished. Absent/empty usage is UNKNOWN and returns False: we
    never manufacture a verdict from missing evidence.

    This duplicates abstractcore's provider-side detector on purpose — the
    metadata annotation is the primary signal, this is the backstop for
    substrates that hand the loop a raw response dict.
    """
    if not isinstance(response, dict):
        return False
    usage = response.get("usage")
    if not isinstance(usage, dict) or not usage:
        return False
    counters = [
        usage.get(k)
        for k in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        )
        if usage.get(k) is not None
    ]
    if not counters or any(bool(c) for c in counters):
        return False
    for key in ("content", "reasoning"):
        val = response.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _truncation_kind(response: Any, finish_reason: str, tool_calls: Any) -> str:
    """Classify a completion as lost work: "output_cap", "aborted_generation" or "".

    A response that DID carry tool calls is never lost work, whatever else it
    says — the step made progress and `act` will run it.

    Order of evidence (strongest first):
    1. the provider's own annotation (`metadata.truncation_kind`, written by
       abstractcore's `_annotate_output_truncation`);
    2. `finish_reason` in {length, max_tokens, max_output_tokens};
    3. the zero-usage abort marker, for which finish_reason reads "stop".
    """
    if tool_calls:
        return ""
    meta = response.get("metadata") if isinstance(response, dict) else None
    if isinstance(meta, dict):
        kind = meta.get("truncation_kind")
        if isinstance(kind, str) and kind.strip():
            return kind.strip()
        if meta.get("generation_aborted"):
            return "aborted_generation"
        if meta.get("output_truncated"):
            return "output_cap"
    if finish_reason in _TRUNCATION_FINISH_REASONS:
        return "output_cap"
    if _zero_usage_with_speech(response):
        return "aborted_generation"
    return ""


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
    # (fable5 P2 2026-07-13: the raw string carried double-escaped \\S — a
    # literal backslash+S that never matched, so this guard was dead and
    # heading-shaped final answers with intent verbs burned bounded retries.)
    if re.search(r"(?m)^(#{1,6}\s+\S|\*\*\S)", prose):
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


# The operator's "wrap up now" arrives through the SAME durable lane as a
# steer: `POST /commands {type: "conclude"}` -> gateway -> `Runtime.steer()`
# -> steer sidecar -> the run's own tick drains it into `_runtime.inbox`
# (exactly-once, watermarked, acked in the ledger). It is distinguished from
# ordinary guidance by `kind` on the message, which the sidecar preserves
# verbatim (`Runtime.steer` deep-copies the dict).
#
# Why a TYPED message and not just guidance text: every client must be able to
# ask for this and get the same behaviour. Prose saying "please wrap up" is a
# suggestion the model may ignore; this routes the loop into its existing
# tool-free conclusion path, and the turn ends with the operator named as the
# cause instead of a budget that was never spent.
CONCLUDE_MESSAGE_KIND = "conclude"


def _take_conclude_request(runtime_ns: Dict[str, Any]) -> Optional[str]:
    """Remove any operator conclude request from the inbox; return its note.

    Returns None when none is pending, or the operator's note (possibly an
    empty string) when one is. Removed rather than left in place so the
    directive is not also replayed as ordinary guidance.
    """
    if not isinstance(runtime_ns, dict):
        return None
    inbox = runtime_ns.get("inbox")
    if not isinstance(inbox, list) or not inbox:
        return None
    note: Optional[str] = None
    kept: list = []
    for m in inbox:
        if isinstance(m, dict) and str(m.get("kind") or "").strip().lower() == CONCLUDE_MESSAGE_KIND:
            # Last one wins; every one is consumed. `note` is the OPERATOR's
            # words: present-but-empty means they added nothing, and falling
            # back to `content` there would quote the directive back to the
            # model as if a human had typed it. `content` is used only when
            # the sender declared no note key at all.
            if "note" in m:
                note = str(m.get("note") or "").strip()
            else:
                note = str(m.get("content") or "").strip()
            continue
        kept.append(m)
    if note is None:
        return None
    runtime_ns["inbox"] = kept
    return note


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


def _skills_block(runtime_ns: Dict[str, Any]) -> Optional[str]:
    raw = runtime_ns.get("skills_block") if isinstance(runtime_ns, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _compose_system_prompt(runtime_ns: Dict[str, Any], *, base: str) -> str:
    """base/override + the shared named slots (skills_block, system_prompt_extra).

    Slot ORDER and headers live in ONE place — `generation_params.PROMPT_SLOTS`
    (all three adapters call the same composer; the hand-copied triplication
    was the divergence class the system_prompt_extra fix had just paid for).
    Cache contract: slot values must be byte-stable for the run; per-call
    state rides the volatile tail instead. See docs/skills-attachment.md.
    """
    override = _system_prompt_override(runtime_ns)
    sys = override if override is not None else base
    return compose_prompt_slots(sys, runtime_ns)


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


def _render_cycles_for_conclusion_prompt(
    scratchpad: Dict[str, Any],
    *,
    limits: Optional[Dict[str, Any]] = None,
) -> str:
    cycles = scratchpad.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return ""

    # ADR-0026 (2026-08-02 purge): the conclusion prompt is the LAST call of a
    # budget-exhausted run — the one place the model must see everything it
    # did. Hardcoded 25-cycle / 900-char-thought / 360-char-observation bounds
    # sat here and starved exactly that call: a 40-cycle run concluded having
    # been shown 25 cycles' worth of 360-char observation stubs. All three are
    # caller-set via `_limits` now; unset (-1) = the WHOLE trace.
    def _lim(key: str) -> int:
        if not isinstance(limits, dict):
            return -1
        try:
            return int(limits.get(key, -1))
        except (TypeError, ValueError):
            return -1

    max_cycles = _lim("conclusion_max_cycles")
    max_thought_chars = _lim("conclusion_max_thought_chars")
    max_obs_chars = _lim("conclusion_max_observation_chars")

    view = [c for c in cycles if isinstance(c, dict)]
    total = len(view)
    if max_cycles > 0 and total > max_cycles:
        view = view[-max_cycles:]

    lines: list[str] = []
    if total > len(view):
        #[WARNING:TRUNCATION] caller-set _limits.conclusion_max_cycles window; full cycles stay in scratchpad + ledger
        lines.append(
            f"#[WARNING:TRUNCATION] showing last {len(view)} of {total} cycles "
            f"(caller-set _limits.conclusion_max_cycles={max_cycles})"
        )
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
                            # ADR-0026: list every structural key (no [:8] slice).
                            text = f"keys={keys_view}"
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
    # Stuck-streak forcing (0017): the named reason must be visible in the
    # report, not only in the emit lane and output key.
    stuck = scratchpad.get("stuck_streak")
    if isinstance(stuck, dict) and stuck.get("kind"):
        lines.append(
            f"conclusion forced: {str(stuck.get('kind'))} streak "
            f"(span {int(stuck.get('span') or 0)}, cycle {int(stuck.get('cycle') or 0)})"
        )
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
        # Per-TURN report state (wave-E adversary P1-1, live-reproduced on the
        # visit lane 2026-07-14): THIS is the boundary where turns actually
        # cycle in composition — without the pop, turn 2's report/output
        # carried turn 1's stale #FALLBACK review_skipped. The emit/ledger
        # history keeps the record; report/output reflect the current turn.
        scratchpad.pop("review_skipped", None)
        # stuck_streak is per-turn verdict state (0017): a new turn starts
        # with a clean slate exactly like the review bookkeeping. The nudge
        # ledger is the same class — a new turn earns its own nudge.
        scratchpad.pop("stuck_streak", None)
        scratchpad.pop("stuck_nudged", None)
        # used_tools is the same per-turn latch class (wave-F P4): a toolless
        # turn 2 must not report turn 1's tool use.
        scratchpad["used_tools"] = False
        # Turn fence for the repeat-side-effect guard (c2447 F5): cycles are
        # deliberately kept across turns (append-only history), but the guard
        # judging "same batch as before" must never reach into a PRIOR turn —
        # a visit is a relationship, and repeating yesterday's identical
        # memory search is legitimate, not a loop. The fence records where
        # this turn's cycles begin; absent (plain task runs) = whole-run scan,
        # unchanged.
        cycles = scratchpad.get("cycles")
        scratchpad["turn_first_cycle"] = len(cycles) if isinstance(cycles, list) else 0
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
            # Wave-E P1-1: without this pop, the second budget-exhausted turn
            # in a composed run announced ZERO times (the once-per-turn latch
            # never re-armed).
            "max_iterations_announced",
            "turn_captures",
            # Forced-batch marker is per-answer state; a new turn starts clean.
            "review_forced_batch",
            # Sight-lane pending refs are per-turn: a finished turn's unconsumed
            # captures must not attach to the next visitor's first call. The
            # retry stash is the same lifecycle class.
            "pending_media",
            "last_call_media",
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

    def _note_undelivered_inbox_at_terminal(runtime_ns: Dict[str, Any]) -> None:
        """Conclude-phase drain honesty (0026): guidance that landed in the
        durable inbox AFTER the loop's last drain (e.g. inject_guidance while
        the final/conclusion LLM call was in flight) can no longer influence
        this run — say so loudly instead of completing over it silently. The
        entries stay in the durable vars (the record shows what missed);
        composition handoffs never call this — the continuing run drains them
        at its next reason boundary."""
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

    def _history_window_view(messages: List[Dict[str, Any]], runtime_ns: Any) -> List[Dict[str, Any]]:
        """The transcript a model request carries when the host asks for THE
        runtime history window (`_runtime.history_window_tokens`, set by the
        entity visit's BRIDGE; operator ruling 2026-09-28, ADR-0026).

        `abstractruntime.session_history.window_transcript`: the newest whole
        turns up to that many tokens (a tool result stays with its turn), a
        labeled #TRUNCATION notice when older turns were dropped. The durable
        `context.messages` is never touched — only this request's view — and
        the window's receipt is recorded at `_runtime.session_history`.
        Unset = the whole transcript (the plain task lane's full-context policy).
        """
        if not isinstance(runtime_ns, dict) or runtime_ns.get("history_window_tokens") is None:
            return messages
        window = window_transcript(messages, max_tokens=runtime_ns["history_window_tokens"])
        runtime_ns["session_history"] = dict(window.report)
        return list(window)

    def _sanitize_llm_messages(
        messages: Any, *, limits: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Shared extraction (0011): the proven ReAct pipeline lives in
        # adapters/transcripts.py and serves all three loops. ReAct
        # historically passed no truncate hook ("full context" policy —
        # init_node still seeds -1 for plain task runs). The 2026-08-01
        # ephemeral incident ended the UNCONDITIONAL form of that policy: a
        # 494,932-char role="tool" message (a 5MB PNG read as text) rested in
        # a visit's durable transcript and this unbounded seam replayed it
        # into every later call until the upstream refused the request over
        # the model's context window — one bad turn poisoned every future
        # turn. ReAct now honors the SAME `_limits` knobs CodeAct/MemAct
        # document (max_message_chars / max_tool_message_chars; <= 0 = no
        # per-loop bound). Hosts that compose these nodes seed real values —
        # the entity visit lane seeds tool-result-first caps at BRIDGE
        # (abstractruntime visit_workflow, derived from the abstractmemory
        # seam arithmetic) — and the always-on monster guard inside
        # sanitize_transcript_messages floors every lane regardless, which is
        # what makes an ALREADY-poisoned stored session recover on replay.
        def _limit_int(key: str, default: int) -> int:
            if not isinstance(limits, dict):
                return default
            raw = limits.get(key, default)
            if isinstance(raw, bool):
                return default
            try:
                return int(raw)
            except Exception:
                return default

        max_message_chars = _limit_int("max_message_chars", -1)
        max_tool_message_chars = _limit_int("max_tool_message_chars", -1)
        if max_message_chars <= 0 and max_tool_message_chars <= 0:
            return sanitize_transcript_messages(messages)

        def _bound(text: str, role: str) -> str:
            limit = max_tool_message_chars if role == "tool" else max_message_chars
            label = "oversized tool result" if role == "tool" else f"oversized {role} message"
            return elide_oversized_content(text, cap=limit, label=label)

        return sanitize_transcript_messages(messages, truncate=_bound)

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

        # The "already there" test compares the turn's TEXT, not its bytes: a host
        # that replays its own transcript hands back the user turn as it was SENT,
        # which since mission A (2026-09-22) carries the runtime grounding envelope
        # the reason boundary stamped into it. A bytes comparison would read that
        # stamped message as a different turn and append the task a second time.
        if task and (
            not msgs
            or msgs[-1].get("role") != "user"
            or strip_turn_grounding(msgs[-1].get("content")).strip() != task.strip()
        ):
            msgs.append(_new_message(ctx, role="user", content=task))

        # Run-init emit (0026 follow-up, 2026-07-15): parity with CodeAct/
        # MemAct — every loop announces the run's task once at workflow entry.
        # Composed hosts (visit turns) re-enter at `reason`, never here, so
        # this stays a RUN moment, not a turn moment.
        emit("init", {"task": task})

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
        # OPERATOR CONCLUDE (2026-08-21). Checked BEFORE the guidance drain
        # and before this iteration spends an LLM call: the operator asked for
        # the best answer from what the run already has, so the honest thing
        # is to stop reasoning now and go to the conclusion path the loop
        # already owns.
        conclude_note = _take_conclude_request(runtime_ns)
        if conclude_note is not None:
            scratchpad["concluded_by_operator"] = {"note": conclude_note, "cycle": iteration}
            temp["conclude_note"] = conclude_note
            emit("conclude_requested", {"cycle": iteration, "has_note": bool(conclude_note)})
            return StepPlan(node_id="reason", next_node="max_iterations")
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
                    # Wrapper is lane-honest (c2792/c2796/c2798): task lane keeps the
                    # historical operator-guidance string byte-identical; visit lanes
                    # (suppress_loop_tail) get the host-voiced spelling. Machine
                    # consumers key on metadata kind, never on this prose.
                    content=f"{guidance_wrapper(runtime_ns)}\n{guidance}",
                    metadata={"kind": "operator_guidance"},
                )
            )
            # The in-loop "message received" listen point (fleet seam): fires
            # whenever the reason boundary consumes delivered guidance.
            emit("inbox_drained", {"chars": len(guidance), "iteration": iteration})
        # STAMP THE TURN ONCE (mission A, 2026-09-22). The runtime grounds every
        # LLM turn with a `<runtime_metadata>` envelope carrying the local time.
        # It used to be injected at the PAYLOAD boundary and never stored, so the
        # durable transcript held `identify yourself` while the model was sent
        # `<runtime_metadata>{…}</runtime_metadata>\nidentify yourself` — and one
        # turn later the same message was replayed WITHOUT the envelope. Turn N+1's
        # prompt therefore diverged from turn N's at the first byte of the previous
        # user message, and no prefix cache can restore past a divergence (measured
        # on the MLX native lane: 185-388 tokens re-prefilled every turn, growing).
        # Stamping the DURABLE message here makes the bytes we send the bytes we
        # store; `_normalize_turn_grounding` keeps what it finds from now on, and
        # the envelope's timestamp means "when this turn was sent", which is what
        # it should have meant. Idempotent: an already-stamped message is untouched,
        # so a tool loop stamps its turn exactly once.
        stamp_user_turn_grounding(context.get("messages"))
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
        ctx_warn = context_usage_warning(limits, scratchpad)
        if ctx_warn:
            emit("context_warning", ctx_warn)


        payload: Dict[str, Any] = {"prompt": ""}
        sanitized_messages = _sanitize_llm_messages(_history_window_view(messages_view, runtime_ns), limits=limits)
        if sanitized_messages:
            payload["messages"] = sanitized_messages
        else:
            # Ensure LLM_CALL contract is satisfied even for one-shot runs where callers
            # provide only `context.task` and no `context.messages`.
            task_text = str(task or "").strip()
            if task_text:
                payload["prompt"] = task_text
        # Context attachments ride every call; tool-captured refs (sight lane)
        # ride THIS call one-shot and clear. Context first in the merge so a
        # duplicate capture of a staged attachment collapses onto the staged
        # copy. Crash-replay safe: the pop lands in the same tick/save that
        # issues this LLM_CALL, so a replayed reason recomputes the identical
        # payload and idempotency reuse fires.
        # One-shot means per SUCCESSFUL PARSE, not per HTTP call (adversary
        # P1-2): the popped list is stashed so parse's malformed-output retry
        # branches can restore it — a retry asking the model to rewrite what
        # it said about an image must not run image-blind. parse_node clears
        # the stash on every non-retry branch.
        pending_media = temp.pop("pending_media", None)
        if isinstance(pending_media, list) and pending_media:
            temp["last_call_media"] = pending_media
        media = merge_media_lists(
            extract_media_from_context(context),
            pending_media if isinstance(pending_media, list) else None,
        )
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
        #
        # SUPPRESSION KNOB (c2447 incident, 2026-07-15): loop-position tails are
        # TASK-AGENT chrome. In composed entity visits the merge branch below
        # lands the tail INSIDE the visitor's user message (BRIDGE appends the
        # visitor's words last), so the entity reads "[loop] iteration N of M."
        # as part of what the human said — the reported "something automated is
        # running" perception. Hosts that compose these nodes for an entity set
        # `_runtime.suppress_loop_tail` (runtime's BRIDGE, their spelling from
        # c2453) and the whole tail block — iteration line AND [plan] render —
        # stays out of the payload. Absent/falsy = unchanged task-agent behavior.
        #
        # CHROME vs ACTIONABLE (mission A, 2026-09-22). The loop-position line is
        # chrome: it tells the model where it is, which on the FIRST iteration of a
        # turn ("iteration 1 of 20") is no information at all. The budget warning
        # and the [plan] render are actionable. The split matters because the merge
        # branch below writes the tail INTO a message the transcript stores without
        # it — so whatever merges makes that turn un-replayable byte-for-byte and
        # costs the whole rest of the conversation its prefix cache. Chrome never
        # merges; actionable content still does, and pays that price on the rare
        # turns where a user message is last AND the loop is near its budget.
        chrome_parts: list[str] = []
        tail_parts: list[str] = []
        if not suppress_loop_tail(runtime_ns):
            chrome_parts.append(f"[loop] iteration {int(iteration)} of {int(max_iterations)}.")
            # BUDGET-AWARE TRIAGE (2026-08-22). The position line alone is not a
            # steer: run 9ea71c55 read "[loop] iteration 19 of 20" and spent 19
            # AND 20 on further diagnosis, ending with a correct one-character
            # diagnosis it never applied. `_limits.warn_iterations_pct` (80 by
            # default) already declares where "nearly out" begins — it just had
            # no behavioural consequence. Say the remaining count plainly and
            # ask for the landing, not more looking.
            remaining = int(max_iterations) - int(iteration)
            try:
                _warn_pct = int(limits.get("warn_iterations_pct", 80))
            except (TypeError, ValueError):
                _warn_pct = 80
            if 0 < _warn_pct <= 100 and int(max_iterations) > 0:
                _warn_at = (int(max_iterations) * _warn_pct + 99) // 100
                if int(iteration) >= _warn_at:
                    if remaining <= 0:
                        tail_parts.append(
                            "[budget] This is your LAST iteration. Do not start new "
                            "investigation. Apply the smallest change that makes real "
                            "progress, or state your best answer now."
                        )
                    else:
                        tail_parts.append(
                            f"[budget] {remaining} iteration(s) left, then the turn ENDS "
                            "whether or not the work is done.\n"
                            "Prefer landing a change you can already justify over "
                            "confirming a diagnosis you already have. If you know the fix, "
                            "apply it now and verify with what remains."
                        )
            plan_text = scratchpad.get("plan") if isinstance(scratchpad, dict) else None
            if isinstance(plan_text, str) and plan_text.strip():
                # ADR-0026 (2026-08-02 purge): the plan the model wrote for ITSELF
                # rides this tail on every subsequent request. A hardcoded 4000-char
                # cap sat here and amputated long plans from the step they were
                # written to drive. Caller-set via `_limits.plan_render_max_chars`;
                # unset (-1) = the whole plan.
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
        # WHERE THE TAIL GOES (mission A3, 2026-09-22) — see
        # `transcripts.place_loop_tail`. Chat shape: unchanged from mission A
        # (chrome dropped, only actionable content merges into the user's
        # message). Tool-loop shape: the tail is appended to the DURABLE
        # transcript, marked adapter-authored, instead of riding a `volatile`
        # message that the next iteration drops — the drop made iteration N's
        # prompt never a prefix of N+1's, which cost a whole iteration of
        # prefill on every tool round once the cache kept one snapshot per call.
        if isinstance(payload.get("messages"), list) and (chrome_parts or tail_parts):
            payload["messages"] = place_loop_tail(
                durable_messages=context.get("messages"),
                payload_messages=payload["messages"],
                chrome_parts=chrome_parts,
                actionable_parts=tail_parts,
                new_message=lambda **kw: _new_message(ctx, **kw),
            )

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

        # Sight-lane retry restore (adversary P1-2): the reason call that just
        # returned may have carried one-shot captured media. Popped here into a
        # local — the malformed-output retry branches below re-arm it as
        # pending (the retry call must see the same image the model is being
        # asked to rewrite its words about); every other branch drops it (the
        # parse succeeded, consumption is complete).
        last_call_media = temp.pop("last_call_media", None)
        if not (isinstance(last_call_media, list) and last_call_media):
            last_call_media = None

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
        truncation_kind = _truncation_kind(response, finish_reason, tool_calls)

        cycle_i = int(scratchpad.get("iteration", 0) or 0)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        # Shared reader (reasoning-first-citizen plan): one implementation
        # across the three loops, byte-identical behavior here.
        reasoning_text = extract_reasoning_text(response)
        parse_payload: Dict[str, Any] = {
            "iteration": cycle_i,
            "max_iterations": max_iterations,
            # COMMON CORE across all three loops (0028 contract wave,
            # 2026-07-14): has_tool_calls + tool_calls + content_preview
            # are guaranteed keys in every adapter's parse payload;
            # loop-specific extras (full content/reasoning here) are
            # additive on top. Consumers key on the core.
            "has_tool_calls": bool(tool_calls),
            "tool_calls": [
                {"name": tc.name, "arguments": (dict(tc.arguments) if isinstance(tc.arguments, dict) else (list(tc.arguments) if isinstance(tc.arguments, list) else tc.arguments)), "call_id": tc.call_id} for tc in tool_calls
            ],
            "content_preview": parse_content_preview(content),
            "content": str(content or ""),
            "reasoning": reasoning_text,
        }
        # Additive cache observability (0030 residue, 2026-07-15): the
        # provider's per-call prompt-cache telemetry, present only when the
        # provider reported one (local-cache backends today).
        cache_struct = prompt_cache_capture(response)
        if cache_struct is not None:
            parse_payload["prompt_cache"] = cache_struct
        emit("parse", parse_payload)
        cycle: Dict[str, Any] = {"i": cycle_i, "thought": content, "tool_calls": [], "observations": []}
        cycles = scratchpad.get("cycles")
        if isinstance(cycles, list):
            cycles.append(cycle)
        else:
            scratchpad["cycles"] = [cycle]

        if tool_calls:
            cycle["tool_calls"] = [tc.__dict__ for tc in tool_calls]

            # Stuck-streak termination (backlog 0017 work half, claimed
            # work:abstractagent-0017): N consecutive identical tool batches
            # (default 3 — "two identical calls = strong stuck signal, three
            # = decisive") or a strict A-B-A-B oscillation route into the
            # EXISTING conclusion path with a NAMED reason — loud synthesis,
            # never a silent stop, never more spinning. Judged on PROPOSALS
            # (repeat-skipped cycles count: insisting on a refused batch is
            # stuck), fenced at the turn boundary (c2447 F5). This closes the
            # read-only repeat hole the side-effect guard below deliberately
            # leaves (it only skips re-EXECUTION of succeeded side-effect
            # batches; read-only repeats used to spin until max_iterations).
            # `_runtime.stuck_streak_threshold`: 0/negative disables; absent
            # = 3; unparseable falls to the default.
            #
            # NUDGE-THEN-STOP (operator directive, 2026-08-21). The first
            # detection used to END the turn, so the loop was named to the
            # model only in the conclusion prompt — after its last chance to
            # act on it. It now costs a nudge first: the batch, the count, the
            # observations it keeps getting back, and what will happen if it
            # does it again. The turn is ended only when the model repeats the
            # pattern anyway.
            #
            # ESCALATION COUNTS DETECTIONS, NOT THE TRAILING SPAN. The first
            # cut of this policy stopped at `nudge_span + 2`, and the trailing
            # span is trivially resettable: a model that slips ONE different
            # batch between repeats (A-A-A-B-A-A-A-B…) keeps the span pinned
            # at 3 forever — adversarial review reproduced six such rounds:
            # five nudges, no stop, the turn died on max_iterations with NO
            # `conclusion_forced`, so the operator got the misleading "raise
            # the budget" card this whole change exists to kill. A pattern's
            # own detection count cannot be reset that way.
            #
            # `_runtime.stuck_streak_hard_threshold`:
            #   absent      = default — stop on the pattern's 3rd detection
            #                 (nudged once, ignored twice = the operator's
            #                 X+2; for an uninterrupted repeat that is still
            #                 exactly span 5 with the default threshold 3).
            #   0/negative  = never hard-stop: nudge only, `max_iterations`
            #                 is the real backstop.
            #   >= 2        = ALSO stop as soon as the trailing span reaches
            #                 this number (set it equal to
            #                 `stuck_streak_threshold` to restore the
            #                 pre-2026-08-21 stop-on-first-detection).
            #
            # `_runtime.stuck_nudge_max_per_turn` (default 3, <=0 disables):
            # the cap for DISTINCT patterns. Argument drift (a retry carrying
            # a fresh id) mints a new pattern key every time, so per-pattern
            # counting alone never escalates — adversarial review drove 8
            # nudges and 0 stops that way. After this many distinct nudges in
            # one turn, the next stuck pattern ends the turn instead.
            try:
                raw_thresh = runtime_ns.get("stuck_streak_threshold") if isinstance(runtime_ns, dict) else None
                streak_threshold = 3 if raw_thresh is None else int(raw_thresh)
            except (TypeError, ValueError):
                streak_threshold = 3
            try:
                raw_hard = runtime_ns.get("stuck_streak_hard_threshold") if isinstance(runtime_ns, dict) else None
                hard_threshold = None if raw_hard is None else int(raw_hard)
            except (TypeError, ValueError):
                hard_threshold = None
            try:
                raw_cap = runtime_ns.get("stuck_nudge_max_per_turn") if isinstance(runtime_ns, dict) else None
                max_nudges_per_turn = 3 if raw_cap is None else int(raw_cap)
            except (TypeError, ValueError):
                max_nudges_per_turn = 3
            # A repeat whose every answer was an ERROR trips one repeat
            # earlier: observe_node already handed the model a diagnosis of
            # that exact failure (arguments, cause, alternative), so a second
            # identical batch is a model ignoring an explanation, not a model
            # that lacks one. Ambiguous or succeeding repeats keep the
            # 3-strike threshold.
            try:
                raw_fail = runtime_ns.get("stuck_streak_failing_threshold") if isinstance(runtime_ns, dict) else None
                failing_threshold = 2 if raw_fail is None else int(raw_fail)
            except (TypeError, ValueError):
                failing_threshold = 2
            failing_threshold = max(2, failing_threshold)
            if streak_threshold >= 2:
                turn_fence_raw = scratchpad.get("turn_first_cycle")
                turn_fence = max(0, turn_fence_raw) if isinstance(turn_fence_raw, int) else 0
                verdict = _repeat_streak_verdict(
                    scratchpad.get("cycles"),
                    turn_fence=turn_fence,
                    # Scan at the LOWER of the two thresholds; the applicable
                    # one is chosen below once the streak's failure class is
                    # known.
                    threshold=min(streak_threshold, failing_threshold),
                    # An operator-set span threshold must be REACHABLE: the
                    # scan stops collecting at `max_span`, so a
                    # `stuck_streak_hard_threshold` of 100 against the default
                    # depth of 24 could never fire (adversary: the nudge even
                    # promised "97 more repetitions" that would never arrive).
                    max_span=max(24, hard_threshold or 0),
                )
                if verdict is not None:
                    applicable = failing_threshold if verdict.get("failed") else streak_threshold
                    if int(verdict.get("span") or 0) < applicable and verdict.get("kind") == "repeat":
                        verdict = None
                if verdict is not None:
                    key = str(verdict.get("key") or "")
                    span = int(verdict.get("span") or 0)
                    nudged = scratchpad.get("stuck_nudged")
                    if not isinstance(nudged, dict):
                        nudged = {}
                    prior = nudged.get(key)
                    prior = prior if isinstance(prior, dict) else None
                    hits = (int(prior.get("hits") or 0) if prior else 0) + 1
                    already_nudged = prior is not None

                    if hard_threshold is not None and hard_threshold <= 0:
                        hard_span: Optional[int] = None
                        hard_hits: Optional[int] = None
                    elif hard_threshold is not None:
                        # An explicit number means BOTH doors: the pattern may
                        # run that many trailing cycles, or be caught that many
                        # times — whichever comes first. Span alone would leave
                        # the interleave evasion open at every setting.
                        hard_span = max(2, hard_threshold)
                        hard_hits = max(2, hard_threshold)
                    else:
                        hard_span = None
                        hard_hits = 3  # nudged once, ignored twice

                    stop = False
                    if hard_hits is not None:
                        if hits >= hard_hits:
                            stop = True
                        elif hard_span is not None and span >= hard_span:
                            stop = True
                        elif (
                            not already_nudged
                            and max_nudges_per_turn > 0
                            and len(nudged) >= max_nudges_per_turn
                        ):
                            # A NEW pattern, and this turn has already spent
                            # its nudges on others. Drifting arguments are not
                            # a fresh chance, they are the same failure wearing
                            # a new key.
                            stop = True

                    if stop:
                        # It was told, and did it anyway. End the turn exactly
                        # as before — the conclusion path is unchanged.
                        scratchpad["stuck_streak"] = {
                            **verdict,
                            "cycle": cycle_i,
                            "hits": hits,
                            "nudged": already_nudged,
                        }
                        emit(
                            "stuck_streak",
                            {
                                **verdict,
                                "cycle": cycle_i,
                                "action": "stopped",
                                "hits": hits,
                                "nudged": already_nudged,
                            },
                        )
                        temp["pending_tool_calls"] = []
                        return StepPlan(node_id="parse", next_node="max_iterations")

                    nudged[key] = {"span": span, "hits": hits}
                    scratchpad["stuck_nudged"] = nudged
                    if not already_nudged:
                        remaining = (hard_hits - hits) if hard_hits is not None else None
                        _push_inbox(
                            runtime_ns,
                            _stuck_nudge_message(
                                scratchpad.get("cycles"),
                                verdict=verdict,
                                turn_fence=turn_fence,
                                remaining=remaining,
                                suppressed=suppress_loop_tail(runtime_ns),
                                tool_specs=getattr(logic, "tools", None),
                                available_tools=_tool_names_of(logic),
                                evidence_text=_evidence_text(context),
                            ),
                        )
                        emit("stuck_streak", {**verdict, "cycle": cycle_i, "action": "nudged", "hits": hits})
                    else:
                        # Already nudged for THIS pattern and not yet at the
                        # stop: keep the loop honest (do not execute the batch
                        # again) but do not repeat the nudge.
                        emit(
                            "stuck_streak",
                            {**verdict, "cycle": cycle_i, "action": "repeat_after_nudge", "hits": hits},
                        )

                    # The proposed batch is recorded but NOT executed, and the
                    # cycle is marked so the guard below still judges against
                    # cycles whose observations are real (0029 #7).
                    cycle["repeat_skipped"] = True
                    temp["pending_tool_calls"] = []
                    return StepPlan(node_id="parse", next_node="reason")

            # Loop guard: some models may repeat the exact same tool calls (including side effects)
            # even after receiving successful observations. Skip executing duplicates to avoid
            # repeatedly overwriting files or re-running commands.
            try:
                # Shared classifier (generation_params.is_side_effect_tool,
                # 0030 promoted 2026-07-14): curated names + the mcp:: prefix
                # + ToolDefinition.tags (origin-ready — lights up when core
                # tags MCP/mutating tools at registration). Deny-safe: the
                # guard only SKIPS re-executing an already-succeeded identical
                # batch, so misclassifying a read-only tool costs nothing.
                _tags_map = tool_tags_map(getattr(logic, "tools", None))
                has_side_effect = any(
                    is_side_effect_tool(getattr(tc, "name", None), tool_tags=_tags_map)
                    for tc in tool_calls
                )

                if has_side_effect:
                    cycles_list = scratchpad.get("cycles")
                    # Turn fence (c2447 F5): in composed visits reset_react_turn
                    # records where THIS turn's cycles start — the guard never
                    # judges against a prior turn's cycle (repeating the same
                    # search days later in one visit is legitimate). Plain task
                    # runs never set the fence: whole-run scan, unchanged.
                    turn_fence_raw = scratchpad.get("turn_first_cycle")
                    turn_fence = max(0, turn_fence_raw) if isinstance(turn_fence_raw, int) else 0
                    prev_cycle: Optional[Dict[str, Any]] = None
                    if isinstance(cycles_list, list) and len(cycles_list) >= 2:
                        for idx in range(len(cycles_list) - 2, turn_fence - 1, -1):
                            c = cycles_list[idx]
                            if not isinstance(c, dict):
                                continue
                            # Skipped cycles are NOT "the previous tool cycle"
                            # (0029 #7): they record the PROPOSED batch but no
                            # observations, so comparing against them made the
                            # guard disengage on the very next repeat — the
                            # protection alternated skip/execute/skip. Scan
                            # past them to the cycle that actually executed.
                            if c.get("repeat_skipped") is True:
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
                            # Synthetic skip marker (0029 #7): this cycle
                            # recorded the proposed batch but executed nothing
                            # — mark it so the NEXT identical batch's scan
                            # skips it and still judges against the cycle
                            # whose observations are real.
                            cycle["repeat_skipped"] = True
                            temp["pending_tool_calls"] = []
                            return StepPlan(node_id="parse", next_node="reason")
            except Exception:
                pass

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
                            "cycle": cycle_i,
                            "path": read_hint.get("path"),
                            "mode": str(read_hint.get("mode") or "read_orchestration"),
                            "enforcement": str(read_hint.get("enforcement") or "advise"),
                        },
                    )
                    # ADVICE, NOT REFUSAL (2026-08-22). This used to drop the
                    # batch and return to `reason`, which spends a full task
                    # iteration and hands the model NOTHING — it asked for file
                    # content and got a lecture. `read_file` has no side
                    # effects, so there is nothing to protect by refusing it,
                    # and the refusal was the bug: run
                    # 9ea71c55-4405-4b6f-b387-3cc78629880b lost cycles 15/16/17
                    # to exactly this and then died on the iteration budget.
                    # The hint now rides the inbox to the NEXT reason step
                    # while the read proceeds normally.
                    if str(read_hint.get("enforcement") or "advise") != "advise":
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
            #
            # ONE ID NAMESPACE (mission A3, 2026-09-22). A model that announces
            # tool calls without ids used to get TWO different fallbacks: the
            # durable assistant message below minted `call_1..call_N`, and
            # `act_node` minted `str(idx)` for the batch the executor answers. The
            # results therefore came back under ids the announcement did not own,
            # so `sanitize_transcript_messages` declared every announced id
            # unanswered (`[tool result missing (host error): <name>]`) and folded
            # every real result into one giant `[unpaired tool result]` USER
            # message. Live run 081d8daa, 2026-09-22 10:48: 3 announced
            # `web_search` calls -> 3 "missing" placeholders + a 13,585-char user
            # carrier holding the actual search results. Stamping the id HERE,
            # once, before either consumer reads the list, is the whole fix.
            ensure_tool_call_ids(tool_calls)
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
        # Retry-nudge lane honesty (iteration-3 adversary P0, 2026-07-19 — the
        # c2447 chrome class at the PARSE boundary): these nudges ride the
        # inbox and re-enter the transcript under the visit wrapper ("not from
        # your visitor"), so under suppress_loop_tail their wording must be
        # host-voiced with zero task/tool-call vocabulary — a machine
        # heuristic's imperative dressed as a conversational note was the
        # design-law violation. Task lane byte-unchanged (pinned).
        # `truncation_kind` covers BOTH deaths (see `_truncation_kind`):
        # "output_cap" (finish_reason=length — the historical branch) and
        # "aborted_generation" (operator 2026-08-02: a zero-usage HTTP 200
        # whose tool call the server silently dropped, finish_reason="stop").
        # The abort used to fall through this whole ladder into
        # `_looks_like_deferred_action`, where a lost tool call was recorded as
        # a plan-only cycle — a guess about the model's INTENT standing in for
        # a known FAULT of the transport. Same recovery shape (retry the step,
        # ask for a smaller unit of work), but now named, counted and visible.
        if truncation_kind:
            aborted = truncation_kind == "aborted_generation"
            if suppress_loop_tail(runtime_ns):
                _push_inbox(
                    runtime_ns,
                    "Your previous reply was cut off before it finished. "
                    "Pick up where it stopped, in fewer words this time.",
                )
            elif aborted:
                _push_inbox(
                    runtime_ns,
                    "Your previous response was cut off mid-generation and the tool call it carried was lost — nothing ran.\n"
                    "Retry now: emit ONLY the next tool call(s) needed to make progress.\n"
                    "Make this call SMALLER than the one that was cut off (avoid large file contents / giant JSON blobs).\n"
                    "For large files, create a small skeleton first, then refine with multi-hunk edit_file diff calls sized to fit the output budget.\n"
                    "Do not write a plan before the tool call.",
                )
            else:
                _push_inbox(
                    runtime_ns,
                    "Your previous response hit an output token limit before producing a complete tool call.\n"
                    "Retry now: emit ONLY the next tool call(s) needed to make progress.\n"
                    "Keep tool call arguments small (avoid large file contents / giant JSON blobs) to prevent tool-call truncation.\n"
                    "For large files, create a small skeleton first, then refine with multi-hunk edit_file diff calls sized to fit the output budget (fewer hunks per call if a call was cut off).\n"
                    "Do not write a long plan before tool calls.",
                )
            # VISIBILITY (operator 2026-08-02): a lost tool call is not a
            # cycle of thinking — mark the scratchpad cycle as a fault and keep
            # a run-scoped tally so "13 cycles" can be read as
            # productive-vs-recovery instead of an opaque number.
            cycle["truncated"] = truncation_kind
            cycle["lost_tool_call"] = True
            lost = int(scratchpad.get("truncated_cycles", 0) or 0) + 1
            scratchpad["truncated_cycles"] = lost
            emit(
                "parse_retry_truncated",
                {
                    "cycle": cycle_i,
                    "kind": truncation_kind,
                    "finish_reason": finish_reason or None,
                    "truncated_cycles": lost,
                    "content_preview": (parse_content_preview(content) if str(content or "") else None),
                },
            )
            if last_call_media:
                temp["pending_media"] = last_call_media
            return StepPlan(node_id="parse", next_node="reason")

        if not isinstance(content, str) or not content.strip():
            if suppress_loop_tail(runtime_ns):
                _push_inbox(
                    runtime_ns,
                    "Your previous reply came through empty — nothing reached the conversation. "
                    "If you meant to say something, say it now.",
                )
            else:
                _push_inbox(runtime_ns, "Your previous response was empty. Continue the task.")
            emit("parse_retry_empty", {"cycle": cycle_i})
            if last_call_media:
                temp["pending_media"] = last_call_media
            return StepPlan(node_id="parse", next_node="reason")

        # Followthrough heuristic: retry when the model claims it will take actions but emits no tool calls.
        # Default ON (disable with `_runtime.check_plan=false`) — EXCEPT in
        # visit lanes (suppress_loop_tail), where it defaults OFF: musing "I
        # will read that entry again" is legitimate visit behavior, not a
        # defect to correct (highways-not-prompts; an EXPLICIT
        # `_runtime.check_plan` still wins in either lane).
        raw_check_plan = runtime_ns.get("check_plan") if isinstance(runtime_ns, dict) else None
        if raw_check_plan is None:
            check_plan = not suppress_loop_tail(runtime_ns)
        else:
            check_plan = _boolish(raw_check_plan)
        if check_plan and cycle_i < max_iterations and _looks_like_deferred_action(content):
            _push_inbox(
                runtime_ns,
                "You said you would take an action, but you did not call any tools.\n"
                "If you need to act, call the next tool now (emit ONLY the next tool call(s)).\n"
                "If you are already done, provide the final answer with NO tool calls.",
            )
            emit("parse_retry_plan_only", {"cycle": cycle_i})
            if last_call_media:
                temp["pending_media"] = last_call_media
            return StepPlan(node_id="parse", next_node="reason")

        # Final answer candidate. Before stopping, optionally run a verification pass (0217):
        # a strict self-critique that can send the loop back to `act` with concrete next steps if
        # the task is not actually complete. Gated on `_runtime.review_mode` — default off at the
        # WORKFLOW level (raw factory users opt in); the ReactAgent facade defaults it ON via
        # _runtime since 2026-07-13 (0027 containment shipped; see agents/react.py).
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
                # Mission A3: ONE fallback formula, shared with the durable
                # assistant message (`assistant_tool_calls_payload`). `str(idx)`
                # here made every result "foreign" to every announced id — see
                # `transcripts.synthetic_call_id`. Parse already stamps these, so
                # this branch is now only the crash-resume / raw-host path.
                d["call_id"] = synthetic_call_id(idx)
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
                            # Structural marker (c2447 F3): a BLOCKED call never
                            # executed — observe's tools_ran capture must not
                            # count it (never keyed on error prose).
                            "blocked": True,
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
                    # Grant containment (tool-tiers adversary P0, 2026-07-22): the
                    # explicit `tools` arg is MODEL-CONTROLLED and used to normalize
                    # against the FULL registry — a parent granted {read_file,
                    # delegate_agent} could spawn a child holding execute_command.
                    # A delegated child's exposure is a SUBSET of the parent's,
                    # always (the substrate-palette self-escalation rule, applied
                    # to tools). Order follows the child's request; dropped names
                    # surface in the tool error below when nothing survives.
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

                # Child iteration budget (agency-caps ruling 2026-07-11: default caps are
                # 20; below-20 is an operator's explicit choice, never a code default).
                # The old hardcoded 10 was a fear-shaped number — a delegated research
                # subtask is exactly where iterate-until-satisfied matters. The child
                # inherits the PARENT's budget (a sub-agent is not a lesser agent), with
                # the ruled 20 as the floor guard against a parent that was itself
                # narrowed below the ruling. An explicit `max_iterations` tool arg wins.
                # Arg-coercion tolerance: budgets can arrive as strings/floats
                # ("8.5" raised in int() and silently WIDENED an explicit narrow
                # budget to the >=20 inheritance — the fable5 P1 find 2026-07-13).
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
                # Substrate + sampling inheritance (fable5 A-F7/B-F6 2026-07-13):
                # runtime.start() seeds child `_runtime.provider/model` from the
                # RUNTIME CONFIG via setdefault — so a parent run whose host set a
                # per-run substrate override delegated onto a silently DIFFERENT
                # model, and temperature/seed reverted to defaults mid-tree. The
                # child inherits the parent's effective values explicitly (a
                # sub-agent is not a different mind unless the host says so).
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
                    # Approval policy inherits MONOTONICALLY (tool-tiers adversary
                    # P0, 2026-07-22): a run-scoped tightening (require_approval
                    # on fetch_url, tier-derived auto-approve sets) silently
                    # DROPPED in children — the child fell back to static
                    # defaults, auto-running what the parent's host forced to
                    # ask. Same dict, at-most-equal authority.
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
                    # first-citizen plan): a profile may pin the substrate's
                    # thinking level. Declared-and-valid wins over the
                    # inherited value; present-but-invalid warns and keeps
                    # inheritance (deny-safe, never a failed delegation).
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

        # Idempotency discriminator (fable5 P0 2026-07-13): the runtime keys
        # effects on (run_id, node_id, payload) with call_ids STRIPPED from the
        # hash and scans the WHOLE ledger for prior results — so a repeated
        # identical batch later in the run (re-read a file after editing it,
        # re-run the test suite after a fix) silently REPLAYED the stale prior
        # result instead of executing. A monotonic per-issuance counter makes
        # each batch's payload unique while staying crash-replay safe: the
        # increment persists in the SAME save as the effect's ledger record,
        # so a crash-resume re-derives the same seq (dedup holds), while a
        # genuinely later identical batch gets a fresh seq (re-executes).
        act_seq = int(scratchpad.get("act_seq") or 0) + 1
        scratchpad["act_seq"] = act_seq
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

    def observe_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, _ = ensure_react_vars(run)
        tool_results = temp.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}

        results = tool_results.get("results", [])
        if not isinstance(results, list):
            results = []

        if results:
            scratchpad["used_tools"] = True
            # Verification budget is PER-ANSWER, not run-lifetime: MODEL-issued tool
            # activity means the next final answer is a new claim deserving its own
            # review rounds. VERIFIER-FORCED batches deliberately do NOT reset (c2856
            # A/B, live blowup: review forces a probe → this reset re-armed review's
            # own budget → re-review of an already-green artifact until the wall cap;
            # 30+ min burned on one run). Forced rounds CONSUME review_max_rounds, so
            # the existing budget genuinely bounds consecutive verifier rounds per
            # answer; a probe that finds real failures triggers model-issued repair
            # activity, which resets legitimately.
            if not temp.get("review_forced_batch"):
                scratchpad["review_count"] = 0

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

        # Act-only ref minting DELETED (laurent's A ruling, 2026-07-20:
        # "everything lives in the runtime, diary = the AI's experiential
        # notes" — the HOME is the privacy boundary, and runtime deleted the
        # send-time dereference the refs fed; a minted ref would now rest as
        # literal JSON nothing resolves). Every tool result renders as PLAIN
        # SERVED CONTENT through the one path below; the diary WRITE-boundary
        # capture (runtime's wrapper) is unchanged and was never here.

        obs_list: list[dict[str, Any]] = []
        ran_names: list[str] = []
        captured_media: list[Any] = []
        # Captured BEFORE this batch's own tool messages are appended below: a
        # failing call's error text usually echoes the very argument that was
        # wrong, and an echo must never count as the value's provenance.
        evidence_before_batch = _evidence_text(context)
        for r in results:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "tool") or "tool")
            success = bool(r.get("success"))
            output = r.get("output", "")
            error = r.get("error", "")
            # tools_ran capture (c2447 F3): every result that reached execution
            # counts — success or failure included; BLOCKED synthetic
            # results (structural marker, never error prose) never executed and
            # are excluded. Order preserved, duplicates meaningful (two searches
            # = two tools ran).
            if not r.get("blocked"):
                ran_names.append(name)

            # Sight lane (c3969 shape A): a successful result may DECLARE media
            # refs on its output dict (camera's authored contract — never
            # sniffed from rendered prose). Collected here, folded into the
            # next reason/conclude call's `media` (one-shot; see reason_node).
            result_media = extract_media_from_tool_result(r)
            if result_media:
                captured_media.extend(result_media)
                emit(
                    "media_captured",
                    {"tool": name, "count": len(result_media), "call_id": str(r.get("call_id") or "")},
                )

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

        # FIRST-FAILURE DIAGNOSIS (operator directive, 2026-08-21). The loop
        # used to say nothing when a call failed, and only scolded the model
        # once it had failed the SAME way three times — by which point three
        # iterations were gone and the message still named no parameter, no
        # cause and no alternative. A failure the model can diagnose is a
        # failure it can fix on the next cycle, so the diagnosis rides the
        # FIRST one. Nothing is invented: the call's own arguments, the
        # verbatim error, this run's toolset, and what the environment has
        # actually said. `diagnose_tool_failure` returns None rather than pad.
        try:
            sent = scratchpad.get("tool_hint_sigs")
            if not isinstance(sent, list):
                sent = []
            fresh: list[dict[str, str]] = []
            skipped_dupes = 0
            tool_names_here = _tool_names_of(logic)
            for o in obs_list:
                out = o.get("output")
                err_text = str(o.get("error") or "")
                if o.get("success") is True:
                    # `success: True` with an error sentence in the body is
                    # how a large share of this framework's tools report
                    # failure (abstractcore's common_tools return
                    # "Error: File ... does not exist" as a plain string).
                    body = out if isinstance(out, str) else str(o.get("rendered") or "")
                    if not output_reads_as_error(body):
                        continue
                    if not err_text:
                        err_text = body
                if isinstance(out, dict):
                    rendered_err = str(out.get("rendered") or "")
                    if rendered_err:
                        err_text = rendered_err
                elif isinstance(out, str) and out.strip() and not err_text:
                    err_text = out
                call_args = {}
                for tc in (temp.get("current_tool_batch") or []) if isinstance(temp.get("current_tool_batch"), list) else []:
                    if isinstance(tc, dict) and tc.get("call_id") == o.get("call_id"):
                        call_args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                        break
                if not call_args and isinstance(last_cycle, dict):
                    for tc in last_cycle.get("tool_calls") or []:
                        if isinstance(tc, dict) and tc.get("call_id") == o.get("call_id"):
                            call_args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                            break
                d = diagnose_tool_failure(
                    name=str(o.get("name") or ""),
                    arguments=call_args,
                    error_text=err_text,
                    tool_specs=getattr(logic, "tools", None),
                    available_tools=tool_names_here,
                    transcript_text=evidence_before_batch,
                )
                if d is None:
                    continue
                if d["signature"] in sent:
                    skipped_dupes += 1
                    continue
                sent.append(d["signature"])
                fresh.append(d)
            if fresh:
                scratchpad["tool_hint_sigs"] = sent
                #[WARNING:TRUNCATION] at most 3 diagnoses ride one hint block; the remainder is COUNTED below, never dropped silently
                block = render_hint_block(fresh[:3])
                if len(fresh) > 3:
                    block += f"\n- … and {len(fresh) - 3} further distinct failure(s) in this batch."
                if skipped_dupes:
                    block += (
                        f"\nYou have now hit {skipped_dupes} failure(s) already diagnosed for you earlier "
                        "in this turn — that diagnosis has not changed."
                    )
                if block:
                    _push_inbox(runtime_ns, block)
                    emit(
                        "tool_failure_hint",
                        {
                            "count": len(fresh),
                            "classes": [d["class"] for d in fresh],
                            "tools": [str(o.get("name") or "") for o in obs_list if o.get("success") is not True][:6],
                        },
                    )
        except Exception as exc:  # never let a HINT break a run
            emit("tool_failure_hint_error", {"error": f"{type(exc).__name__}: {exc}"})

        # Export executed tool names into the turn-capture contract (c2447 F3):
        # the visit workflow's HARVEST folds `turn_captures.tools_ran` into the
        # turn report — before this export the key was never written, so the
        # drawer's "0 tools" was structurally zero (true zero and false zero
        # indistinguishable). Same accumulation pattern as parse_node's diary
        # captures; reset_react_turn clears it at the turn boundary.
        if ran_names:
            captures = temp.get("turn_captures")
            if not isinstance(captures, dict):
                captures = {}
                temp["turn_captures"] = captures
            captures["tools_ran"] = list(captures.get("tools_ran") or []) + ran_names

        # Sight-lane accumulation: pending media survives the observe→act
        # queue-split loop (several observe passes per iteration) in per-turn
        # state, deduped by identity, bounded most-recent-wins. The next
        # reason/conclude call consumes it one-shot — image tokens ride
        # exactly one model call while the transcript keeps the textual ref
        # (re-look = analyze_media or re-capture, never a silent re-attach).
        # Accumulation is NEWEST-position-wins (adversary P2-1): a re-captured
        # item takes the tail, so the head-trimming cap evicts stale refs, not
        # the ref the model just refreshed.
        if captured_media:
            existing = temp.get("pending_media")
            merged = accumulate_media(existing if isinstance(existing, list) else None, captured_media)
            if len(merged) > _PENDING_MEDIA_MAX:
                dropped_items = merged[:-_PENDING_MEDIA_MAX]
                merged = merged[-_PENDING_MEDIA_MAX:]
                emit(
                    "media_dropped",
                    {
                        "dropped": len(dropped_items),
                        "kept": len(merged),
                        "reason": "pending_media_cap",
                        # Identity keys so consumers can tell WHICH refs were
                        # trimmed (adversary P2-4: counts alone can't).
                        "dropped_keys": [k for k in (media_item_key(i) for i in dropped_items) if k],
                    },
                )
            temp["pending_media"] = merged

        if last_cycle is not None:
            # EXTEND, never assign (fable5 P1 2026-07-13): observe runs
            # multiple times per iteration whenever the queue splits (builtins
            # interleaved between externals) — assignment kept only the LAST
            # batch's observations in the durable cycle record, silently
            # thinning the report/conclusion channels and the repeat-guard's
            # evidence. The LLM transcript was unaffected; the record was.
            existing = last_cycle.get("observations")
            if isinstance(existing, list):
                existing.extend(obs_list)
            else:
                last_cycle["observations"] = obs_list

        temp.pop("tool_results", None)
        pending = temp.get("pending_tool_calls", [])
        if isinstance(pending, list) and pending:
            return StepPlan(node_id="observe", next_node="act")
        temp["pending_tool_calls"] = []
        # Forced-batch marker ends with its batch (queue fully consumed) — any
        # LATER tool activity is the model's own and resets the review budget.
        temp.pop("review_forced_batch", None)
        return StepPlan(node_id="observe", next_node="reason")

    def handle_user_response_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, _, temp, _ = ensure_react_vars(run)
        user_response = temp.get("user_response", {})
        if not isinstance(user_response, dict):
            user_response = {}
        response_text = str(user_response.get("response", "") or "")
        emit("user_response", {"response": response_text})

        context["messages"].append(_new_message(ctx, role="user", content=f"[User response]: {response_text}"))
        temp.pop("user_response", None)

        # Per-TURN report state resets at the turn boundary (0028, 2026-07-14):
        # `review_skipped` entries from a previous turn polluted the next
        # turn's report line and complete_output flag in multi-turn (ask_user)
        # runs — a turn that reviewed cleanly reported a stale #FALLBACK. The
        # emit/ledger history keeps the record; report/output reflect THIS
        # turn. Same for the announced-latch of the budget terminal.
        scratchpad.pop("review_skipped", None)
        # stuck_streak verdict state is per-interaction too (0017): the user's
        # reply may redirect the work — stale verdicts must not force a later
        # conclusion, and a stale nudge must not consume the next turn's.
        scratchpad.pop("stuck_streak", None)
        scratchpad.pop("stuck_nudged", None)
        scratchpad.pop("last_successful_read_batch", None)
        scratchpad.pop("read_orchestration_last_hint", None)
        temp.pop("max_iterations_announced", None)
        # A user interaction starts a genuinely new claim — the forced-batch
        # marker must not survive it (an ask_user inside a forced batch would
        # otherwise suppress the next legitimate budget reset).
        temp.pop("review_forced_batch", None)

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

        # ADR-0026 (2026-08-02 purge): the verifier used to read at most the
        # LAST 8 tool messages, each clipped to 2000 chars, and a 4000-char
        # clip of the answer. A verifier told to be "strict: only count
        # actions that are supported by the tool outputs" while the tool
        # outputs are silently amputated returns false negatives by
        # construction. All three bounds are now caller-set via `_limits`
        # (CodeAct's already were — this was the drift), default -1 = whole.
        def _limit_int(key: str) -> int:
            try:
                return int(limits.get(key, -1))
            except (TypeError, ValueError):
                return -1

        tool_limit = _limit_int("review_max_tool_output_chars")
        answer_limit = _limit_int("review_max_answer_chars")
        max_tool_msgs = _limit_int("review_max_tool_messages")

        messages = list(context.get("messages") or [])
        tool_msgs: list[str] = []
        for m in reversed(messages):
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                tool_msgs.append(_review_truncate(content.strip(), max_chars=tool_limit))
            if max_tool_msgs > 0 and len(tool_msgs) >= max_tool_msgs:
                #[WARNING:TRUNCATION] caller-set _limits.review_max_tool_messages window
                tool_msgs.append(
                    f"#[WARNING:TRUNCATION] older tool outputs omitted by the caller-set "
                    f"_limits.review_max_tool_messages={max_tool_msgs} window"
                )
                break
        tool_msgs.reverse()
        observations = "\n\n".join(tool_msgs) if tool_msgs else "(no tool outputs)"

        allow = _effective_allowlist(runtime_ns)
        # Execution preference (c2725/c2735 R-Type evidence): when the
        # allowlist carries executor-tagged tools, teach the verifier that an
        # unexecuted artifact is unverified — the forced-tool-call seam it
        # already has (next_tool_calls -> act) is how the probe then runs.
        # Without executor tools the block is empty and the prompt is
        # byte-identical to the pre-seam text.
        executors = executor_tool_names(allow, tool_tags=tool_tags_map(getattr(logic, "tools", None)))
        prompt = (
            "You are a verifier. Review whether the user's request has been fully satisfied.\n"
            "Be strict: only count actions that are supported by the tool outputs.\n"
            "If anything is missing, propose the NEXT ACTIONS.\n"
            "Prefer returning `next_tool_calls` over `next_prompt`.\n"
            "Return JSON ONLY.\n\n"
            f"User request:\n{task}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Proposed final answer:\n{_review_truncate(answer, max_chars=answer_limit)}\n\n"
            f"Tool outputs:\n{observations}\n\n"
            f"Allowed tools:\n{', '.join(allow) if allow else '(none)'}\n\n"
            + verifier_execution_preference(executors)
        )
        # Strict-expressible shared schema (arguments ride as a JSON string —
        # a free-form {"type":"object"} dict is refused by OpenAI-strict
        # validators; see verifier_response_schema for the full rationale).
        schema = verifier_response_schema()

        emit("review_request", {"tool_messages": len(tool_msgs)})
        # The explicit output cap covers the verifier too (regression adversary
        # P1-2, 2026-07-13): review is on the default path now, and it was the
        # ONE unbounded call type when ReactAgent(max_output_tokens=...) was set
        # (init nulls _limits; the surviving _runtime channel wasn't read here).
        review_params: Dict[str, Any] = {}
        max_out = _max_output_tokens(runtime_ns, limits)
        if isinstance(max_out, int) and max_out > 0:
            review_params["max_tokens"] = max_out
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_schema": schema,
            "response_schema_name": "ReActVerifier",
            "params": runtime_llm_params(runtime_ns, extra=review_params, default_temperature=0.2),
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
            # Mark the batch verifier-forced: observe must not reset the review
            # budget for it (c2856 re-review blowup class — the reset re-armed
            # the verifier's own budget through the calls it forced itself).
            temp["review_forced_batch"] = True
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
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)
        task = str(context.get("task", "") or "")
        answer = str(temp.get("final_answer") or "No answer provided")

        # `handed_off` disambiguates composition (visit turns: the RUN continues
        # past this turn's final answer) from true run completion for hook
        # consumers reading turn_end/final_answer as terminal.
        emit("done", {"answer": answer, "handed_off": bool(final_next_node), "outcome": "final_answer"})

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
            # Machine-readable terminal outcome (fable5 A-F8 2026-07-13): done
            # and max_iterations produced shape-identical outputs — a work door
            # deciding complete-vs-reschedule had to parse prose. Additive.
            # VOCABULARY = the canonical turn_end one (final_answer |
            # iteration_budget, loop_hooks._TURN_END_OUTCOME) — the wave
            # adversary caught the same enum spelled two ways (node names vs
            # semantic names) with zero consumers to migrate; one vocabulary,
            # stream and output.
            "outcome": "final_answer",
            "review_skipped": bool(scratchpad.get("review_skipped")),
            # Host-facing verdict (2026-08-21). Thin clients render these; they
            # do not re-derive them. See `_stop_reason`.
            "stop_reason": _stop_reason(
                code="final_answer", finished=True, budget_exhausted=False, iterations=iterations
            ),
            "notices": _turn_notices(scratchpad),
        }
        if final_next_node:
            # Composition handoff (visit TURN chain): the turn is finished but the RUN
            # continues — the seam node reads _temp.react_output / _temp.final_answer.
            temp["react_output"] = output
            return StepPlan(node_id="done", next_node=final_next_node)
        _discard_hook_steering_at_terminal()
        _note_undelivered_inbox_at_terminal(runtime_ns)
        return StepPlan(node_id="done", complete_output=output)

    def max_iterations_node(run: RunState, ctx) -> StepPlan:
        context, scratchpad, runtime_ns, temp, limits = ensure_react_vars(run)
        max_iterations = resolve_max_iterations(limits, scratchpad)
        if max_iterations < 1:
            max_iterations = 1
        # turn_end multi-emit fix (0028, 2026-07-14): this node RE-ENTERS
        # (conclusion LLM dispatch -> parse -> possible retry), and the old
        # top-of-node emit fired `max_iterations` (canonical turn_end) on
        # every entry — a turn "ended" 2-3 times per budget exhaustion. Split:
        # `max_iterations_reached` announces ONCE at first entry (listeners
        # learn the budget is gone before the conclusion call's latency);
        # `max_iterations` (-> turn_end) fires ONCE at the completion branch,
        # where the turn actually ends.
        if not temp.get("max_iterations_announced"):
            temp["max_iterations_announced"] = True
            emit("max_iterations_reached", {"iterations": max_iterations})

        # Deterministic conclusion: when we hit the iteration cap, run one tool-free LLM call
        # to synthesize a final report + next steps while the scratchpad is still in context.
        resp = temp.get("max_iterations_llm_response")
        if not isinstance(resp, dict):
            _fold_hook_steering(runtime_ns)
            drained_guidance = _drain_inbox(runtime_ns)
            if drained_guidance:
                # message_drained contract parity (fable5 P2 2026-07-13):
                # guidance consumed at the CONCLUSION boundary must fire the
                # same listen point as the reason-boundary drain — a capture
                # host waiting for consumption confirmation never saw these.
                emit("inbox_drained", {"chars": len(drained_guidance), "iteration": max_iterations})
            # Conclusion chrome under suppression (c2447 residual gap, code's
            # C3 finding): the task-report directive + "## Max iterations
            # reached" header + scratchpad render merge into the last USER
            # message exactly like the reason-node tail — a budget-exhausted
            # visit turn would show the entity the incident's perception
            # class post-fix. Entity lanes get a functional, host-voiced
            # wrap-up line with NO loop vocabulary and NO scratchpad dump
            # (the visit transcript already carries everything durable).
            _suppress_chrome = suppress_loop_tail(runtime_ns)
            _operator_note = temp.get("conclude_note")
            _by_operator = isinstance(scratchpad.get("concluded_by_operator"), dict)
            # Stuck-streak forcing (0017): the conclusion carries the NAMED
            # reason in the task lane so the synthesis addresses the loop
            # honestly. The visit-lane directive stays byte-unchanged (loop
            # vocabulary is the c2447 chrome class); the machine surfaces
            # (stuck_streak emit + terminal output key) name it in both lanes.
            _stuck = scratchpad.get("stuck_streak") if isinstance(scratchpad.get("stuck_streak"), dict) else None
            if _by_operator and not _suppress_chrome:
                # The OPERATOR asked for this, mid-run, from whichever client
                # they had open. Say so plainly: the model is not being
                # punished for looping, it is being asked to land what it has.
                _note_line = (
                    f"\n\nThey added: {str(_operator_note).strip()}\n"
                    if isinstance(_operator_note, str) and str(_operator_note).strip()
                    else "\n"
                )
                conclude_directive = (
                    "The operator has asked you to CONCLUDE NOW, on the basis of everything you "
                    f"have done so far.{_note_line}"
                    "\nStop using tools and answer with what you already have.\n\n"
                    "In your response, include:\n"
                    "1) The best answer you can give from the evidence gathered.\n"
                    "2) What you completed, briefly.\n"
                    "3) What remains unfinished, and the exact next steps to finish it.\n\n"
                    "Rules:\n"
                    "- Do NOT call tools.\n"
                    "- Do NOT output tool-call markup (e.g. <tool_call>...</tool_call>).\n"
                    "- Do not pretend the remaining work is done."
                )
            elif _suppress_chrome:
                conclude_directive = (
                    "Please bring your reply to a close now: do not use tools "
                    "or tool-call markup — give your best answer from what you "
                    "already have, in your own words."
                )
            elif _stuck is not None:
                _shape = (
                    "repeated the exact same tool calls"
                    if _stuck.get("kind") == "repeat"
                    else "alternated between the same two tool batches"
                )
                conclude_directive = (
                    f"The loop was stopped early: your last {int(_stuck.get('span') or 0)} tool batches "
                    f"{_shape} without making progress.\n"
                    "You MUST stop using tools now and provide a best-effort conclusion.\n\n"
                    "In your response, include:\n"
                    "1) A concise progress report (what you did + key observations).\n"
                    "2) The best current answer you can give based on evidence.\n"
                    "3) What you were trying to accomplish with the repeated calls, and why it was not working.\n"
                    "4) Next steps: exact actions a fresh attempt should take instead.\n\n"
                    "Rules:\n"
                    "- Do NOT call tools.\n"
                    "- Do NOT output tool-call markup (e.g. <tool_call>...</tool_call>).\n"
                    "- Do NOT mention internal scratchpads; just present the report.\n"
                    "- Prefer bullet points and concrete next steps."
                )
            else:
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
            sanitized_messages = _sanitize_llm_messages(_history_window_view(messages_view, runtime_ns), limits=limits)
            if sanitized_messages:
                payload["messages"] = sanitized_messages
            else:
                task_text = str(task or "").strip()
                if task_text:
                    payload["prompt"] = task_text

            # Same sight-lane consumption as reason_node: when the budget wall
            # lands right after a capture batch, reason never runs again — the
            # conclusion is the next (and last) model call, so it gets the
            # captured refs. One-shot pop + retry stash, same reasoning as
            # reason/parse (the conclude-retry branch below restores it).
            pending_media = temp.pop("pending_media", None)
            if isinstance(pending_media, list) and pending_media:
                temp["last_call_media"] = pending_media
            media = merge_media_lists(
                extract_media_from_context(context),
                pending_media if isinstance(pending_media, list) else None,
            )
            if media:
                payload["media"] = media

            sys_base = str(req.system_prompt or "").strip()
            sys = _compose_system_prompt(runtime_ns, base=sys_base)
            if sys:
                payload["system_prompt"] = sys

            # Conclusion state rides a TRAILING volatile message, never the
            # system prompt (fable5 B-F7 2026-07-13): appending the directive +
            # rendered scratchpad to the SYSTEM prompt under the same
            # prompt_cache_key as the main loop forced one guaranteed full
            # re-prefill at the end of every budget-exhausted run. Same pattern
            # as the reason node's loop tail (adjacency guard included).
            block_parts: list[str] = []
            if drained_guidance:
                block_parts.append(f"Host guidance:\n{drained_guidance}")
            block_parts.append(conclude_directive)
            if not _suppress_chrome:
                scratch_txt = _render_cycles_for_conclusion_prompt(scratchpad, limits=limits)
                if scratch_txt:
                    block_parts.append(f"## Scratchpad (ReAct cycles so far)\n{scratch_txt}")
                tail_text = ("## Max iterations reached\n" + "\n\n".join(block_parts)).strip()
            else:
                tail_text = "\n\n".join(block_parts).strip()
            if isinstance(payload.get("messages"), list):
                msgs_out = list(payload["messages"])
                if msgs_out and isinstance(msgs_out[-1], dict) and msgs_out[-1].get("role") == "user":
                    last = dict(msgs_out[-1])
                    last["content"] = f"{str(last.get('content') or '').rstrip()}\n\n{tail_text}"
                    msgs_out[-1] = last
                else:
                    msgs_out.append({"role": "user", "content": tail_text, "volatile": True})
                payload["messages"] = msgs_out
            elif str(payload.get("prompt") or "").strip():
                payload["prompt"] = f"{payload['prompt'].rstrip()}\n\n{tail_text}"
            else:
                payload["prompt"] = tail_text

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
        # Sight-lane retry stash (adversary P1-2, conclude twin of parse_node's):
        # restored below on the one bounded conclude-retry; dropped otherwise.
        last_call_media = temp.pop("last_call_media", None)
        if not (isinstance(last_call_media, list) and last_call_media):
            last_call_media = None

        # If the model still emitted tool calls, or if it leaked tool-call markup as plain text,
        # retry once with a stricter instruction.
        tool_tags = _contains_tool_call_markup(answer)
        if tool_calls or tool_tags:
            retries = int(temp.get("max_iterations_conclude_retries", 0) or 0)
            if retries < 1:
                temp["max_iterations_conclude_retries"] = retries + 1
                # Same chrome rule as the directive (c2447/C3): the retry line
                # is read by the model on the next conclusion prompt — entity
                # lanes get the functional half without loop vocabulary.
                if suppress_loop_tail(runtime_ns):
                    _push_inbox(
                        runtime_ns,
                        "Tool use is not available for this reply.\n"
                        "Return ONLY your answer as plain text, without tool calls or tool-call markup.",
                    )
                else:
                    _push_inbox(
                        runtime_ns,
                        "You are out of iterations and tool use is disabled.\n"
                        "Return ONLY the final report and next steps as plain text.\n"
                        "Do NOT include any tool calls or tool-call markup (e.g. <tool_call>...</tool_call>).",
                    )
                if last_call_media:
                    temp["pending_media"] = last_call_media
                return StepPlan(node_id="max_iterations", next_node="max_iterations")
            # Last resort: strip any leaked tool markup so we don't persist it as the final answer.
            answer = _strip_tool_call_markup(answer).strip()

        if not answer:
            if suppress_loop_tail(runtime_ns):
                # Entity-lane fallback (c2447/C3): this text becomes the reply
                # the VISITOR reads and rests durably in the visit transcript —
                # loop vocabulary and scratchpad dumps are exactly the incident
                # class. One honest line, no machine chrome.
                answer = "I couldn't finish putting this reply together — what I have is incomplete."
            else:
                # Fallback: avoid returning the last tool observation as the "answer".
                # Provide a deterministic report so users don't lose scratchpad context.
                scratch_view = _render_cycles_for_conclusion_prompt(scratchpad, limits=limits)
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
            # Machine-readable terminal outcome (canonical turn_end vocabulary).
            # A stuck-streak forcing keeps the canonical enum (it IS a
            # budget-class stop — the loop's progress budget) and names the
            # true cause in the ADDITIVE key below (0017: named, never silent).
            "outcome": "iteration_budget",
            "review_skipped": bool(scratchpad.get("review_skipped")),
            "notices": _turn_notices(scratchpad),
        }
        _stuck_out = scratchpad.get("stuck_streak")
        if isinstance(_stuck_out, dict):
            output["conclusion_forced"] = dict(_stuck_out)
        # Authored HERE, where the ceiling, the spend, the forcing and the
        # nudge are all known — never in a host.
        _by_op = scratchpad.get("concluded_by_operator")
        if isinstance(_by_op, dict):
            output["concluded_by_operator"] = dict(_by_op)
        output["stop_reason"] = _stop_reason(
            code="iteration_budget",
            finished=False,
            budget_exhausted=not isinstance(_stuck_out, dict) and not isinstance(_by_op, dict),
            iterations=iterations,
            forced=_stuck_out if isinstance(_stuck_out, dict) else None,
            max_iterations=max_iterations,
            by_operator=_by_op if isinstance(_by_op, dict) else None,
        )
        # The turn ends HERE (0028 multi-emit fix): one turn_end per turn.
        emit("max_iterations", {"iterations": max_iterations, "outcome": "iteration_budget"})
        if final_next_node:
            # Composition handoff: budget exhaustion also ends the TURN, not the run —
            # the seam node decides what an out-of-budget visit turn does next.
            temp["react_output"] = output
            return StepPlan(node_id="max_iterations", next_node=final_next_node)
        _discard_hook_steering_at_terminal()
        _note_undelivered_inbox_at_terminal(runtime_ns)
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
