"""Shared transcript helpers: durable tool_calls preservation + payload-boundary repair.

Backlog 0011 (C2): ReAct preserved assistant `tool_calls` metadata in the durable
transcript and repaired orphaned tool messages at the payload boundary; CodeAct and
MemAct never received that wave — their durable histories emit `role:"tool"` messages
whose ids no assistant `tool_calls` entry announces, and native OpenAI rejects the
WHOLE request ("assistant tool_calls must be followed by matching tool messages" /
tool message without a preceding tool_calls run -> 400 at iteration 2).

This module is the 0011-preferred EXTRACTION (a shared implementation, not a third
copy): the pipeline below is ReAct's proven shape, parameterized for the loops'
one real difference (CodeAct/MemAct bound message sizes at this boundary; ReAct
does not). ReAct's adapter delegates here unchanged in behavior — its byte-stable
prompt-prefix pins are the regression harness for the extraction.

Pipeline (payload boundary ONLY — durable history is never mutated):
1. field filtering (drop runtime metadata providers reject),
2. optional marked truncation (ADR-0026: any lossy bound keeps a marker),
   plus the always-on oversized-message floor (the 2026-08-01 monster
   guard — see OVERSIZED_MESSAGE_CLAMP_CHARS),
3. assistant `tool_calls` sanitization (function entries, stable synthetic ids),
4. orphan repair BOTH directions (unanswered tool_calls ids get deterministic
   synthetic tool results; unpaired tool messages fold into inert user notes),
5. adjacent-user merge (alternation-strict chat templates).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from abstractcore.tools import ToolCall

# The runtime owns this marker (it also strips it before the provider call) —
# abstractagent declares what it synthesized, abstractruntime's grounding pass
# refuses to treat a declared carrier as "the turn". See
# `abstractruntime.turn_grounding` for the incident this closes (mission A3).
from abstractruntime.turn_grounding import SYNTHETIC_MESSAGE_KEY, SYNTHETIC_TOOL_RESULT

# Interactive builtins resolve through waits + user messages BY DESIGN; only their
# unanswered ids may honestly be labeled "handled interactively". Anything else
# unanswered is a genuinely lost result and must SAY so (a repair that papers over
# real loss with a false claim actively misleads the model).
INTERACTIVE_BUILTIN_NAMES = frozenset({"ask_user"})

# THE MONSTER GUARD (operator incident 2026-08-01, entity "ephemeral",
# driver-lane visit: a 5MB screenshot read as text poisoned the session —
# a 494,932-char `role:"tool"` message rested in the durable transcript and
# rode EVERY subsequent packing pass, until a 48-message, 722,453-char
# request was refused upstream over the model's context window; the session
# was wedged permanently because the transcript is durable and replayed
# whole). This function is the ONE payload seam every loop passes (ReAct,
# CodeAct, MemAct — and entity visits, which are ReAct composed under
# runtime's visit graph), so a floor HERE is what lets an already-poisoned
# STORED session recover on its next packing pass: durable history is never
# mutated, but no single replayed message may enter a payload unbounded.
#
# Cap provenance: 200,000 chars is the stack's largest sanctioned budget
# for an ENTIRE replayed session history (abstractgateway bundle_host
# session seeding: 24k chars default, `min(200000, …)` hard ceiling). A
# single message larger than the largest whole-history budget in the stack
# is structurally a monster in every lane, never legitimate content. Loop
# hooks (CodeAct/MemAct `_limits` bounds, ReAct's `_limits`-driven hook,
# the entity visit lane's BRIDGE-seeded caps) bound FIRST and tighter;
# this guard is only the floor under all of them. ADR-0026: the cut is
# marked in-text and the full content stays durable (run vars + ledger).
OVERSIZED_MESSAGE_CLAMP_CHARS = 200_000


def elide_oversized_content(text: str, *, cap: int, label: str) -> str:
    """Head + labeled stub for content too large to re-send whole.

    Deterministic (byte-stable across packing passes — the prompt-prefix
    cache property survives because the same stored message always clamps
    to the same bytes) and never silent (ADR-0026). `cap <= 0` = no bound.
    """
    s = str(text or "")
    if cap <= 0 or len(s) <= cap:
        return s
    head = s[:cap]
    #[WARNING:TRUNCATION] payload-boundary clamp; durable history keeps the full text
    return (
        head
        + f"\n… [{len(s) - len(head):,} chars elided: {label} - clamped at the LLM "
        "payload boundary; the full text remains in the durable record]"
    )


# The common-core `parse` payload's content preview bound (0028 contract).
PARSE_CONTENT_PREVIEW_CHARS = 200


def parse_content_preview(content: Any, *, cap: int = PARSE_CONTENT_PREVIEW_CHARS) -> str:
    """The `parse` hook payload's bounded view of the model's reply.

    ADR-0026 §1: the key name says "preview", but a bare slice left every
    reader — observability surfaces, monitors, replay tooling — unable to
    tell a 200-char reply from a 20,000-char one that got cut. Name the
    loss; the full content stays in the durable transcript.
    #[WARNING:TRUNCATION] parse-hook content preview; full reply in the transcript
    """
    s = str(content or "")
    if not s:
        return "(no content)"
    if cap <= 0 or len(s) <= cap:
        return s
    return s[:cap] + f"… [#TRUNCATION: {cap} of {len(s):,} chars; full reply in the transcript]"


def extract_reasoning_text(response: Any) -> str:
    """Return the model's separated reasoning text from an LLM result, or "".

    One shared reader for all three loop adapters (reasoning-first-citizen
    plan, agent section: reasoning extraction was ReAct-only, which is the
    drift class backlog 0021 exists for). Providers report the separated
    thinking channel as `reasoning` (core-normalized) or `reasoning_content`
    (some OpenAI-compatible servers); absent means the model interleaved or
    withheld it — "" is honest, never a placeholder.
    """
    try:
        if isinstance(response, dict):
            rc = response.get("reasoning")
            if rc is None:
                rc = response.get("reasoning_content")
            return str(rc or "")
    except Exception:
        pass
    return ""


def synthetic_call_id(index: int) -> str:
    """The ONE fallback id formula for a tool call the model announced without one.

    MISSION A3 (2026-09-22). Two places used to invent this id independently and
    disagreed: `assistant_tool_calls_payload` minted `call_{i+1}` for the DURABLE
    assistant message, while the loops' `act_node` minted `str(idx)` for the batch
    the executor answers (react_runtime ~2608, codeact_runtime ~890,
    memact_runtime ~717). The tool results therefore came back with ids the
    announced calls did not own, so EVERY announced id was "unanswered" and EVERY
    real result was "foreign": `sanitize_transcript_messages` replaced the results
    with `[tool result missing (host error): <name>]` placeholders and folded the
    real output into one giant `[unpaired tool result]` USER message. Measured on
    the operator's live run 081d8daa (2026-09-22 10:48): 3 announced `web_search`
    calls -> 3 placeholders + one 13,585-char user carrier; 5 `fetch_url` calls ->
    5 placeholders + one 8,423-char carrier. The model was told its tools failed
    while their output rode a user turn, and that user turn then attracted the
    runtime grounding envelope and broke the prompt cache (see
    `abstractruntime.turn_grounding`).

    `ensure_tool_call_ids` below stamps this ONCE, at parse time, onto the
    ToolCall objects both consumers read.
    """
    return f"call_{int(index) + 1}"


def ensure_tool_call_ids(tool_calls: Any) -> Any:
    """Give every announced tool call a stable id, IN PLACE, once (mission A3).

    Accepts the loops' two shapes (`ToolCall` objects and plain dicts) and fills
    only the EMPTY ones, so a provider that supplies real ids is untouched. Must
    be called at the parse boundary — before the durable assistant message is
    built and before the pending batch is queued — so the announcement and the
    answers share one id namespace.
    """
    if not isinstance(tool_calls, list):
        return tool_calls
    for i, tc in enumerate(tool_calls):
        if isinstance(tc, ToolCall):
            current = str(tc.call_id).strip() if tc.call_id is not None else ""
            if not current:
                tc.call_id = synthetic_call_id(i)
        elif isinstance(tc, dict):
            current = str(tc.get("call_id") or "").strip()
            if not current:
                tc["call_id"] = synthetic_call_id(i)
    return tool_calls


LOOP_TAIL_MARKER = "loop_tail"


def place_loop_tail(
    *,
    durable_messages: Any,
    payload_messages: List[Dict[str, Any]],
    chrome_parts: List[str],
    actionable_parts: List[str],
    new_message: Callable[..., Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Place a loop's per-iteration tail so the prompt stays an EXACT prefix extension.

    MISSION A3 (2026-09-22). The tail (`[loop] iteration N of M.`, plus the
    actionable `[budget]` / `[plan]` parts) used to ride a trailing `volatile`
    message that existed in iteration N's payload and was GONE from iteration
    N+1's. So iteration N's end-of-prompt snapshot was never a prefix of N+1's
    prompt, by construction. That used to be hidden by mlx-vlm's 256-token
    intermediate checkpoints (~88 tokens lost per iteration). With one snapshot
    per call (the shape that keeps a conversation's lineage alive across a tool
    loop, see abstractcore `NativeSession.prompt_cache`), it cost the WHOLE
    iteration. Measured on the hermetic gateway, basic-agent@0.0.5, real
    `web_search`/`fetch_url`: iterations 3-4 restored only the run's first call
    (2,703 of 5,867-8,556 tokens). And being the last user message, the tail
    also attracted the grounding envelope on every iteration.

    Rule, the same one mission A applied to the user's turn: the bytes sent are
    the bytes stored. In the TOOL-LOOP shape (the payload ends with assistant/tool
    messages), the tail is appended to the DURABLE transcript, once, as a user
    message marked adapter-authored (`SYNTHETIC_MESSAGE_KEY = "loop_tail"`,
    `metadata.kind = "loop_tail"`). The next iteration re-renders it byte for
    byte where it was sent, so iteration N's prompt is an exact prefix of N+1's.
    The marker keeps it from ever being taken for "the turn": the grounding stamp
    and the runtime's grounding pass both skip it, and the runtime strips the
    key before the provider call.

    In the CHAT shape (the payload ends with the user's own durable message) the
    mission A rule is unchanged: chrome is dropped, and only actionable content
    merges into that message.
    """
    msgs_out = list(payload_messages or [])
    chrome = [str(p).strip() for p in (chrome_parts or []) if str(p or "").strip()]
    actionable = [str(p).strip() for p in (actionable_parts or []) if str(p or "").strip()]
    if msgs_out and isinstance(msgs_out[-1], dict) and msgs_out[-1].get("role") == "user":
        merge_text = "\n\n".join(actionable).strip()
        if merge_text:
            last = dict(msgs_out[-1])
            last["content"] = f"{str(last.get('content') or '').rstrip()}\n\n{merge_text}"
            msgs_out[-1] = last
        return msgs_out
    tail_text = "\n\n".join(chrome + actionable).strip()
    if not tail_text:
        return msgs_out
    durable = new_message(role="user", content=tail_text, metadata={"kind": LOOP_TAIL_MARKER})
    durable[SYNTHETIC_MESSAGE_KEY] = LOOP_TAIL_MARKER
    if isinstance(durable_messages, list):
        durable_messages.append(durable)
    msgs_out.append({"role": "user", "content": tail_text, SYNTHETIC_MESSAGE_KEY: LOOP_TAIL_MARKER})
    return msgs_out


def assistant_tool_calls_payload(tool_calls: List[ToolCall]) -> list[dict[str, Any]]:
    """OpenAI-shaped `tool_calls` metadata for a durable assistant message."""
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
            call_id_str = synthetic_call_id(i)
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        tc_payload.append(
            {
                "type": "function",
                "id": call_id_str,
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return tc_payload


def sanitize_tool_calls_field(raw: Any) -> Optional[list[dict[str, Any]]]:
    """Sanitize a durable message's `tool_calls` field for the wire (None = drop)."""
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
            call_id_str = synthetic_call_id(i)
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


def sanitize_transcript_messages(
    messages: Any,
    *,
    truncate: Optional[Callable[[str, str], str]] = None,
    interactive_builtins: frozenset = INTERACTIVE_BUILTIN_NAMES,
) -> List[Dict[str, Any]]:
    """Runtime-owned message dicts -> provider-safe OpenAI-style payload messages.

    `truncate(text, role) -> text` is the loop's optional marked-truncation hook
    (CodeAct/MemAct bound message sizes here; ReAct passes None). Durable history
    is never mutated — every repair below is payload-boundary only.
    """
    if not isinstance(messages, list) or not messages:
        return []
    out: List[Dict[str, Any]] = []

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        if not role:
            continue
        content = m.get("content")
        content_str = "" if content is None else str(content)
        tool_calls = sanitize_tool_calls_field(m.get("tool_calls"))

        # Assistant tool-calls messages may legitimately have empty content, but must still be included.
        if not content_str.strip() and not (role == "assistant" and tool_calls):
            continue

        if truncate is not None:
            content_str = truncate(content_str, role)
        # Structural floor under every lane and hook (see the monster-guard
        # note on OVERSIZED_MESSAGE_CLAMP_CHARS): tool results are the aimed-
        # at class (the 2026-08-01 poison was a tool message), but a monster
        # is a monster whatever role smuggles it — the same guard applies to
        # all, at a bound no honest message of ANY role reaches.
        content_str = elide_oversized_content(
            content_str,
            cap=OVERSIZED_MESSAGE_CLAMP_CHARS,
            label=("oversized tool result" if role == "tool" else f"oversized {role} message"),
        )

        entry: Dict[str, Any] = {"role": role, "content": content_str}
        # An adapter-authored DURABLE message (the loop tail, mission A3) keeps
        # its marker on the payload so the runtime's grounding pass never takes
        # it for the turn; the runtime strips the key before the provider call.
        if m.get(SYNTHETIC_MESSAGE_KEY):
            entry[SYNTHETIC_MESSAGE_KEY] = m.get(SYNTHETIC_MESSAGE_KEY)
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
    # OpenAI reject the WHOLE request -> 400, run fails. The durable history legitimately
    # produces this shape for interactive builtins (ask_user resolves via a wait + a user
    # message, never a tool message). Repaired here at the payload boundary: synthesize a
    # deterministic tool result right after the assistant turn (adjacency is part of the
    # provider contract). The synthetic text is stable so the cached prefix stays
    # byte-identical across iterations (0212).
    repaired: List[Dict[str, Any]] = []
    i = 0
    while i < len(out):
        entry = out[i]
        if entry.get("role") == "tool":
            # Orphan TOOL message (no immediately-preceding assistant tool_calls run —
            # e.g. a compaction/trim cut, or CodeAct's fenced-code execution whose
            # assistant turn carried a code FENCE, not a tool call): strict providers
            # 400 on it. Fold it into an inert user-visible note instead.
            repaired.append({
                "role": "user",
                "content": f"[unpaired tool result]: {str(entry.get('content') or '')}",
                SYNTHETIC_MESSAGE_KEY: SYNTHETIC_TOOL_RESULT,
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
        foreign: List[Dict[str, Any]] = []
        while i < len(out) and out[i].get("role") == "tool":
            tid = str(out[i].get("tool_call_id") or "")
            if tid and tid in want:
                answered.add(tid)
                repaired.append(out[i])
            else:
                # Foreign id inside an answering run (latent in the original
                # ReAct pipeline, caught by the 0011 extraction pins): passing
                # it through 400s strict providers ("tool_call_id not found in
                # tool_calls"). Hold it; fold it AFTER the announced block so
                # the answered run keeps provider-required adjacency.
                foreign.append(out[i])
            i += 1
        for tid, name in want.items():
            if tid in answered:
                continue
            if name in interactive_builtins:
                content = "[handled interactively; see the following conversation messages]"
            else:
                content = f"[tool result missing (host error): {name or 'unknown tool'}]"
            repaired.append({"role": "tool", "tool_call_id": tid, "content": content})
        for stray in foreign:
            repaired.append({
                "role": "user",
                "content": f"[unpaired tool result]: {str(stray.get('content') or '')}",
                SYNTHETIC_MESSAGE_KEY: SYNTHETIC_TOOL_RESULT,
            })

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
            # A merge that swallows a synthesized carrier yields a message that is
            # still PART adapter-synthesized, and a synthesized part is re-derived
            # from the durable transcript on every iteration. The marker therefore
            # survives the merge (mission A3): the runtime's grounding pass must
            # never choose such a message as "the turn", or the envelope hops onto
            # it and every iteration diverges thousands of tokens before the end.
            if entry.get(SYNTHETIC_MESSAGE_KEY) and not prev.get(SYNTHETIC_MESSAGE_KEY):
                prev[SYNTHETIC_MESSAGE_KEY] = entry.get(SYNTHETIC_MESSAGE_KEY)
            merged[-1] = prev
            continue
        merged.append(entry)
    return merged
