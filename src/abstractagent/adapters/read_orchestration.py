"""Helpers for package-owned read orchestration nudges.

This module stays intentionally policy-light. It only detects conservative
same-file reread patterns where the package can nudge the model before it burns
another turn on obviously inefficient read orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "").strip()
    return str(getattr(call, "name", "") or "").strip()


def _call_args(call: Any) -> Dict[str, Any]:
    if isinstance(call, dict):
        args = call.get("arguments")
    else:
        args = getattr(call, "arguments", None)
    return dict(args) if isinstance(args, dict) else {}


def _effective_end(span: Dict[str, Any]) -> int:
    try:
        return int(span.get("end")) if span.get("end") is not None else int(span.get("start") or 1)
    except Exception:
        return int(span.get("start") or 1)


def _ranges_are_nearby(left: Dict[str, Any], right: Dict[str, Any], *, max_gap_lines: int = 20) -> bool:
    l0 = int(left.get("start") or 1)
    l1 = _effective_end(left)
    r0 = int(right.get("start") or 1)
    r1 = _effective_end(right)
    if l0 > l1:
        l0, l1 = l1, l0
    if r0 > r1:
        r0, r1 = r1, r0
    if not (l1 < r0 or r1 < l0):
        return True
    gap = (r0 - l1) if l1 < r0 else (l0 - r1)
    return gap <= max_gap_lines


def normalize_read_file_call(call: Any) -> Optional[Dict[str, Any]]:
    if _call_name(call) != "read_file":
        return None
    args = _call_args(call)
    path = args.get("path") if isinstance(args.get("path"), str) else args.get("file_path")
    if not isinstance(path, str) or not path.strip():
        return None

    should_entire = args.get("should_read_entire_file", True)
    start = args.get("start_line")
    if start is None:
        start = args.get("start_line_one_indexed", args.get("start", 1))
    end = args.get("end_line")
    if end is None:
        end = args.get("end_line_one_indexed_inclusive", args.get("end"))

    try:
        start_i = int(start or 1)
    except Exception:
        start_i = 1
    try:
        end_i = int(end) if end is not None else None
    except Exception:
        end_i = None

    is_slice = end is not None or start_i != 1 or bool(should_entire) is False
    return {
        "path": path.strip(),
        "start": start_i,
        "end": end_i,
        "is_slice": bool(is_slice),
    }


def summarize_successful_read_batch(batch: Any, results: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(batch, list) or not batch or not isinstance(results, list) or len(results) != len(batch):
        return None
    items: List[Dict[str, Any]] = []
    for call, result in zip(batch, results):
        if not isinstance(result, dict) or result.get("success") is not True:
            return None
        item = normalize_read_file_call(call)
        if item is None:
            return None
        items.append(item)
    if not items:
        return None
    paths = {str(i.get("path") or "") for i in items}
    return {
        "count": len(items),
        "items": items,
        "same_path": len(paths) == 1,
        "only_slices": all(bool(i.get("is_slice")) for i in items),
        "non_read_batches_since": 0,
    }


def advance_last_successful_read_batch(previous_batch: Any, current_calls: Any, results: Any) -> Optional[Dict[str, Any]]:
    read_batch = summarize_successful_read_batch(current_calls, results)
    if read_batch is not None:
        return read_batch
    if isinstance(current_calls, list) and any(normalize_read_file_call(call) is not None for call in current_calls):
        return None
    if not isinstance(previous_batch, dict):
        return None
    preserved = dict(previous_batch)
    preserved["non_read_batches_since"] = int(previous_batch.get("non_read_batches_since") or 0) + 1
    return preserved


def _single_same_file_read_pair(previous_batch: Any, current_calls: Any) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(previous_batch, dict):
        return None
    current_batch = summarize_successful_read_batch(current_calls, [{"success": True} for _ in current_calls]) if isinstance(current_calls, list) else None
    if not isinstance(current_batch, dict):
        return None
    if int(previous_batch.get("count") or 0) != 1 or int(current_batch.get("count") or 0) != 1:
        return None
    prev_items = previous_batch.get("items")
    cur_items = current_batch.get("items")
    if not (isinstance(prev_items, list) and prev_items and isinstance(cur_items, list) and cur_items):
        return None
    prev = prev_items[0]
    cur = cur_items[0]
    if str(prev.get("path") or "") != str(cur.get("path") or ""):
        return None
    return prev, cur


def detect_nearby_same_file_staircase(
    previous_batch: Any,
    current_calls: Any,
    *,
    previously_warned_signature: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    pair = _single_same_file_read_pair(previous_batch, current_calls)
    if pair is None:
        return None
    prev, cur = pair
    if not (bool(prev.get("is_slice")) and bool(cur.get("is_slice"))):
        return None
    if not _ranges_are_nearby(prev, cur):
        return None

    signature = f"{cur.get('path')}:{cur.get('start')}:{cur.get('end')}"
    if isinstance(previously_warned_signature, str) and previously_warned_signature == signature:
        return None

    path = str(cur.get("path") or "")
    prev_start = int(prev.get("start") or 1)
    prev_end = _effective_end(prev)
    cur_start = int(cur.get("start") or 1)
    cur_end = _effective_end(cur)
    return {
        "mode": "wider_same_file_read",
        "signature": signature,
        "path": path,
        "message": (
            "You just read a nearby slice of the same file successfully.\n"
            f"For `{path}`, avoid walking the file in another narrow slice ({prev_start}-{prev_end} then {cur_start}-{cur_end}).\n"
            "If you need more nearby context, request ONE wider contiguous read_file range.\n"
            "If you actually need distant passages, batch the read_file calls in ONE response."
        ),
    }


def detect_redundant_same_file_full_reread(
    previous_batch: Any,
    current_calls: Any,
    *,
    previously_warned_signature: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(previous_batch, dict):
        return None
    if int(previous_batch.get("non_read_batches_since") or 0) <= 0:
        return None
    if int(previous_batch.get("count") or 0) != 1:
        return None
    prev_items = previous_batch.get("items")
    if not (isinstance(prev_items, list) and prev_items):
        return None
    prev = prev_items[0]
    if bool(prev.get("is_slice")):
        return None
    if not isinstance(current_calls, list):
        return None

    path = str(prev.get("path") or "")
    matched = False
    for call in current_calls:
        cur = normalize_read_file_call(call)
        if cur is None:
            continue
        if bool(cur.get("is_slice")):
            continue
        if str(cur.get("path") or "") != path:
            continue
        matched = True
        break
    if not matched:
        return None

    signature = f"{path}:full_reread"
    if isinstance(previously_warned_signature, str) and previously_warned_signature == signature:
        return None

    return {
        "mode": "avoid_full_file_reread",
        "signature": signature,
        "path": path,
        "message": (
            "You already read this file successfully.\n"
            f"For `{path}`, do not request another full-file read_file call right now.\n"
            "Use the content already in context, or ask for ONE bounded range only if you need a precise confirmation."
        ),
    }
