"""Self-improvement logging tool.

This tool intentionally writes to a user-local JSONL file (not the source tree)
so it works both in monorepo dev and in installed environments.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from abstractcore.tools import tool


def _default_improvements_path() -> Path:
    configured = os.getenv("ABSTRACTFRAMEWORK_IMPROVEMENTS_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".abstractframework" / "improvements.jsonl"


@tool(
    name="self_improve",
    description="Log an improvement suggestion (JSONL) for later review and prioritization.",
    when_to_use="When you notice a bug, UX issue, missing feature, or an idea that should be tracked for future work",
)
def self_improve(
    suggestion: str,
    target: str = "unknown",
    category: str = "general",
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Append a self-improvement entry to a local JSONL file."""
    try:
        out_path = _default_improvements_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "target": target,
            "suggestion": suggestion,
            "tags": tags or {},
            "cwd": os.getcwd(),
        }

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return f"Logged improvement suggestion to {str(out_path)}"
    except Exception as e:
        return f"Error: {e}"

