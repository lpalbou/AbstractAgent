"""Helpers for attachment/media plumbing in runtime-backed agents.

Two media entry lanes share one normalization (they must never drift):

- RUN-START attachments: ``context["attachments"]`` / ``context["media"]``
  (`extract_media_from_context`) — media the caller staged before the run.
- MID-LOOP tool results (sight lane, commons c3969 shape A / c4089 ruling):
  a successful tool result whose dict ``output`` carries a handler-AUTHORED
  ``media`` list (`extract_media_from_tool_result`) — bare paths, or
  ``{"$artifact": id}`` refs when an artifact store is present. The field is
  declared at the producing tool (camera's contract), never sniffed from
  rendered prose; the runtime llm_client resolves ``$artifact`` refs into
  provider-ready content downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_media_items(raw: Any) -> Optional[List[Any]]:
    """Normalize a raw media list into wire-ready items (or None).

    Accepted item shapes (everything else is dropped silently — the field is
    a declared contract, not a guess surface):
    - non-empty strings (file paths / URLs), stripped
    - dicts carrying ``$artifact`` or ``artifact_id`` (kept whole, copied)
    """
    if isinstance(raw, tuple):
        items = list(raw)
    else:
        items = raw

    if not isinstance(items, list) or not items:
        return None

    out: List[Any] = []
    for item in items:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue

        if isinstance(item, dict):
            # Prefer artifact refs; accept both {"$artifact": "..."} and {"artifact_id": "..."}.
            aid = item.get("$artifact")
            if not (isinstance(aid, str) and aid.strip()):
                aid = item.get("artifact_id")
            if isinstance(aid, str) and aid.strip():
                out.append(dict(item))
            continue

    return out or None


def media_item_key(item: Any) -> Optional[str]:
    """Stable identity key for dedup: artifact id wins, else the path string."""
    if isinstance(item, dict):
        for k in ("$artifact", "artifact_id"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return f"artifact:{v.strip()}"
        return None
    if isinstance(item, str) and item.strip():
        return f"path:{item.strip()}"
    return None


def extract_media_from_context(context: Dict[str, Any]) -> Optional[List[Any]]:
    """Return a normalized `media` list from a runtime `context` dict.

    Supported keys (best-effort):
    - `context["attachments"]`: preferred (artifact refs)
    - `context["media"]`: legacy/alternate
    """
    raw = context.get("attachments")
    if raw is None:
        raw = context.get("media")
    return normalize_media_items(raw)


def merge_media_lists(*lists: Optional[List[Any]]) -> Optional[List[Any]]:
    """Merge media lists in order, deduplicating by identity key.

    First occurrence wins (context attachments keep their position ahead of
    later captures); unkeyable items pass through unchanged — dropping them
    would be silent data loss on shapes this helper does not understand.
    """
    out: List[Any] = []
    seen: set[str] = set()
    for lst in lists:
        if not lst:
            continue
        for item in lst:
            key = media_item_key(item)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            out.append(item)
    return out or None


def extract_media_from_tool_result(result: Dict[str, Any]) -> Optional[List[Any]]:
    """Return declared media refs from ONE executed tool result (or None).

    Contract (sight lane, c3969 A): only a SUCCESSFUL result whose dict
    ``output`` declares a ``media`` list carries sight — a failed capture's
    media claim is not evidence a file landed, and string outputs have no
    declared field to read (authored-only, never prose-sniffed).
    """
    if not isinstance(result, dict) or not result.get("success"):
        return None
    output = result.get("output")
    if not isinstance(output, dict):
        return None
    return normalize_media_items(output.get("media"))
