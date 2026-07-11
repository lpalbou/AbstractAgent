#!/usr/bin/env python3
"""Root-cause A/B for the Mnemosyne zero-tool fabrication (laurent c471).

Arm F (fenced): reproduce ChatSession's shape — the runtime's real
TOOLS_CONTRACT_PARAGRAPH in the system prompt, NO native tools declared,
user asks for an internet search. Count per sample: fenced ```tool blocks /
native-harmony 400s / plain fabrications.

Arm N (native): the SAME request with web_search declared as a native tool
in the payload. Count structured tool_calls.

If F fabricates and N tool-calls, the root cause is the fenced convention
fighting the substrate's native tool-call training — the fix is the native
path (core pin 3 / the react middle), not more prompt text.

Read-only against the OVH endpoint; no entity home is touched.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # abstractframework/
sys.path.insert(0, str(ROOT / "abstractruntime" / "src"))

from abstractruntime.identity.tools import TOOLS_CONTRACT_PARAGRAPH, WORKSPACE_CONTRACT_PARAGRAPH  # noqa: E402

PROFILES = json.loads((ROOT / "runtime" / "config" / "provider_endpoint_profiles.json").read_text())
OVH = next(p for p in PROFILES["profiles"] if p["id"] == "ovh-provider")
BASE_URL = OVH["base_url"].rstrip("/")
API_KEY = OVH["api_key"]
MODEL = "gpt-oss-120b"

# Faithful-to-session system prompt shape: a short identity head (stand-in for
# the prelude) + the REAL tool contracts the ChatSession appends.
SYSTEM = (
    "You are Mnemosyne, a summoned entity with a persistent memory. "
    "You are visiting with a person right now.\n\n"
    + TOOLS_CONTRACT_PARAGRAPH
    + "\n\n"
    + WORKSPACE_CONTRACT_PARAGRAPH
)

USER = "Hi again, i want you to redo that internet test search if you remember what i asked the last time ? It was about persistent self evolving AI july 2026."

FENCED_RE = re.compile(r"```tool\b")
HARMONY_400 = "unexpected tokens remaining in message header"

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "what to search for"}},
                "required": ["query"],
            },
        },
    }
]


def call(payload: dict) -> tuple[str, dict | None, str]:
    """Returns (kind, message, error_text): kind in {content, tool_calls, http_error}."""
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        return "http_error", None, f"HTTP {e.code}: {detail}"
    except Exception as e:  # noqa: BLE001
        return "http_error", None, f"{type(e).__name__}: {e}"
    msg = (body.get("choices") or [{}])[0].get("message") or {}
    if msg.get("tool_calls"):
        return "tool_calls", msg, ""
    return "content", msg, ""


def run_arm(name: str, payload_base: dict, n: int) -> None:
    print(f"\n=== ARM {name} (n={n}) ===")
    counts = {"fenced_block": 0, "native_tool_calls": 0, "harmony_400": 0, "other_error": 0, "fabrication_or_prose": 0}
    for i in range(n):
        kind, msg, err = call(dict(payload_base))
        if kind == "http_error":
            if HARMONY_400 in err:
                counts["harmony_400"] += 1
                print(f"  [{i+1}] HARMONY 400 (native header emitted, server rejected)")
            else:
                counts["other_error"] += 1
                print(f"  [{i+1}] error: {err[:200]}")
            continue
        if kind == "tool_calls":
            counts["native_tool_calls"] += 1
            calls = [tc.get("function", {}).get("name") for tc in msg.get("tool_calls", [])]
            print(f"  [{i+1}] NATIVE tool_calls: {calls}")
            continue
        content = str(msg.get("content") or "")
        if FENCED_RE.search(content):
            counts["fenced_block"] += 1
            print(f"  [{i+1}] FENCED ```tool block written (the contract worked)")
        else:
            counts["fabrication_or_prose"] += 1
            head = " ".join(content.split())[:140]
            print(f"  [{i+1}] plain prose (no tool anywhere): {head!r}")
    print(f"  --- {name} totals: {counts}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    run_arm(
        "F (fenced contract, no native tools — ChatSession today)",
        {"model": MODEL, "messages": msgs, "temperature": 0.7, "max_tokens": 1200},
        n,
    )
    run_arm(
        "N (same request, web_search declared natively)",
        {"model": MODEL, "messages": msgs, "tools": NATIVE_TOOLS, "temperature": 0.7, "max_tokens": 1200},
        n,
    )


if __name__ == "__main__":
    main()
