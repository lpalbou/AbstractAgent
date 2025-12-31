from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


_DELTA_FENCES = {
    "```active_memory_delta",
    "```active-memory-delta",
    "```memory_delta",
    "```memory-delta",
}


def extract_active_memory_delta(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extract an `active_memory_delta` fenced JSON block from assistant content.

    Returns:
      (clean_text, delta_dict_or_none)

    Notes:
    - Uses the LAST matching fence in the message.
    - Removes the entire fenced block from the returned text.
    - If JSON parsing fails, returns the original text and None.
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return raw, None

    lines = raw.splitlines()
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip().lower() in _DELTA_FENCES:
            start_idx = i
    if start_idx is None:
        return raw, None

    end_idx: Optional[int] = None
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip() == "```":
            end_idx = j
            break
    if end_idx is None:
        return raw, None

    payload = "\n".join(lines[start_idx + 1 : end_idx]).strip()
    if not payload:
        delta: Any = {}
    else:
        try:
            delta = json.loads(payload)
        except Exception:
            return raw, None
    if not isinstance(delta, dict):
        return raw, None

    new_lines = lines[:start_idx] + lines[end_idx + 1 :]
    cleaned = "\n".join(new_lines).rstrip()
    return cleaned, dict(delta)

