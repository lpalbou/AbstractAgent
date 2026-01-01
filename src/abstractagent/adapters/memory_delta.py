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
    - Removes ALL matching delta fenced blocks from the returned text (never shown to user).
    - Parses the LAST valid JSON dict among the removed blocks (best-effort).
    - If no valid dict parses, returns (clean_text, None).
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return raw, None

    lines = raw.splitlines()
    cleaned_lines: list[str] = []

    last_valid: Optional[Dict[str, Any]] = None

    i = 0
    while i < len(lines):
        if lines[i].strip().lower() not in _DELTA_FENCES:
            cleaned_lines.append(lines[i])
            i += 1
            continue

        # Found a delta fence. Consume until closing ``` or EOF.
        j = i + 1
        while j < len(lines) and lines[j].strip() != "```":
            j += 1

        payload_lines = lines[i + 1 : j]
        payload = "\n".join(payload_lines).strip()
        if payload:
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    last_valid = dict(parsed)
            except Exception:
                # Ignore parsing errors; we still strip the block from user-visible content.
                pass

        # Skip the closing fence line if present.
        i = j + 1 if j < len(lines) else len(lines)

    cleaned = "\n".join(cleaned_lines).rstrip()
    return cleaned, last_valid
