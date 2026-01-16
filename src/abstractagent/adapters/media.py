"""Helpers for attachment/media plumbing in runtime-backed agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def extract_media_from_context(context: Dict[str, Any]) -> Optional[List[Any]]:
    """Return a normalized `media` list from a runtime `context` dict.

    Supported keys (best-effort):
    - `context["attachments"]`: preferred (artifact refs)
    - `context["media"]`: legacy/alternate
    """
    raw = context.get("attachments")
    if raw is None:
        raw = context.get("media")

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

