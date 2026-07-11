"""Prefix-reuse measurement over captured LLM_CALL payloads.

Frozen visit seam spec (a2a 0013), fixture-home A/B criterion 3: "PREFIX REUSE
measured and reported (stable-head bytes identical across iterations; % reuse in
the report — target is evidence, not a pass bar)."

This module is EVIDENCE TOOLING: it measures what a sequence of request payloads
actually reuses byte-for-byte, mirroring how provider prompt caches key requests
(a cache hit requires an identical serialized prefix). It never mutates payloads
and holds no policy — the numbers are the report.

Model of a request, matching the adapters' LLM_CALL payload shape:
- head: `system_prompt` (str) + `tools` (list) — reusable only if BYTE-IDENTICAL
  to the previous request's head;
- message lane: `messages` (list) — the reusable part is the longest common
  PREFIX of byte-identical messages against the previous request.

Bytes are computed over deterministic JSON (sorted keys, no whitespace) so the
measurement is stable across dict orderings and matches the byte-discipline the
prefix-stability tests pin.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _canon_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    except Exception:
        return len(str(value))


def _canon(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value)


def measure_prefix_reuse(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-request reuse measurements for a sequence of LLM_CALL payloads.

    Returns one entry per request. The first request has nothing to reuse
    (`reused_bytes=0`); each subsequent entry reports, against the PREVIOUS
    request: whether the head (system_prompt + tools) is byte-identical, how many
    leading messages are byte-identical, and the resulting reused/total byte
    counts with `reuse_ratio` in [0, 1].
    """
    out: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for i, payload in enumerate(payloads):
        payload = payload if isinstance(payload, dict) else {}
        head_sys = str(payload.get("system_prompt") or "")
        tools = payload.get("tools")
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []

        head_bytes = len(head_sys) + _canon_bytes(tools)
        msg_canons = [_canon(m) for m in messages]
        total_bytes = head_bytes + sum(len(c) for c in msg_canons)

        entry: Dict[str, Any] = {
            "request_index": i,
            "total_bytes": total_bytes,
            "head_bytes": head_bytes,
            "message_count": len(messages),
        }

        if prev is None:
            entry.update(
                {
                    "head_identical": False,
                    "reused_message_count": 0,
                    "reused_bytes": 0,
                    "reuse_ratio": 0.0,
                }
            )
        else:
            head_identical = head_sys == prev["head_sys"] and _canon(tools) == prev["tools_canon"]
            reused_msgs = 0
            reused_msg_bytes = 0
            for cur, old in zip(msg_canons, prev["msg_canons"]):
                if cur != old:
                    break
                reused_msgs += 1
                reused_msg_bytes += len(cur)
            # Provider caches key on the WHOLE serialized prefix: a changed head
            # invalidates everything after it, so message reuse only counts when
            # the head held.
            reused_bytes = (head_bytes + reused_msg_bytes) if head_identical else 0
            entry.update(
                {
                    "head_identical": head_identical,
                    "reused_message_count": reused_msgs if head_identical else 0,
                    "reused_bytes": reused_bytes,
                    "reuse_ratio": round(reused_bytes / total_bytes, 4) if total_bytes else 0.0,
                }
            )

        out.append(entry)
        prev = {"head_sys": head_sys, "tools_canon": _canon(tools), "msg_canons": msg_canons}
    return out


def prefix_reuse_report(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate report for the A/B evidence row.

    `overall_reuse_ratio` = reused bytes across requests 2..N over their total
    bytes (request 1 has no cache to hit and is excluded from the denominator —
    the number answers "of what we sent after warm-up, how much was reusable").
    """
    per_request = measure_prefix_reuse(payloads)
    tail = per_request[1:]
    tail_total = sum(e["total_bytes"] for e in tail)
    tail_reused = sum(e["reused_bytes"] for e in tail)
    return {
        "requests": len(per_request),
        "heads_identical": all(e["head_identical"] for e in tail) if tail else False,
        "overall_reuse_ratio": round(tail_reused / tail_total, 4) if tail_total else 0.0,
        "per_request": per_request,
    }
