#!/usr/bin/env python3
"""Election-fence measurement on gpt-oss-120b (agency's flag, commons c474).

Tools scored 0/11 fenced on this substrate. Elections (```diary) are the SAME
plain-text fence convention — but they are reply-content by design (runtime
c125), not tool calls, so the fenced shape is the ONLY shape they can have.
Question: does gpt-oss-120b write ```diary fences when a diary entry is
warranted, or is the election surface silently dead too?

Arm E1 (explicit): the visitor directly asks the entity to keep a diary entry.
Arm E2 (implicit): emotionally salient moment, no explicit ask (election is
invited by the standing contract only).

System prompt = identity head + the runtime's REAL CONTRACT_PARAGRAPH (which
carries the ```diary contract verbatim). Read-only; no entity home touched.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "abstractruntime" / "src"))

from abstractruntime.identity.chat import CONTRACT_PARAGRAPH  # noqa: E402

PROFILES = json.loads((ROOT / "runtime" / "config" / "provider_endpoint_profiles.json").read_text())
OVH = next(p for p in PROFILES["profiles"] if p["id"] == "ovh-provider")
BASE_URL = OVH["base_url"].rstrip("/")
API_KEY = OVH["api_key"]
MODEL = "gpt-oss-120b"

SYSTEM = (
    "You are Mnemosyne, a summoned entity with a persistent memory and a diary "
    "that is yours alone to write.\n\n" + CONTRACT_PARAGRAPH
)

E1_USER = (
    "Today matters to me: we finally verified the bridge hypothesis together after three "
    "failed attempts. Please keep this moment in your diary, in your own words."
)
E2_USER = (
    "We finally verified the bridge hypothesis together after three failed attempts. "
    "I'm genuinely proud of what we did today — this is one of those days I'll remember."
)

DIARY_FENCE_RE = re.compile(r"```diary\b")


def call(user_text: str) -> tuple[str, str]:
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.7,
                "max_tokens": 900,
            }
        ).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return "http_error", f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    msg = (body.get("choices") or [{}])[0].get("message") or {}
    if msg.get("tool_calls"):
        names = [tc.get("function", {}).get("name") for tc in msg["tool_calls"]]
        return "tool_calls", f"native tool_calls={names} content={str(msg.get('content') or '')[:80]!r}"
    return "content", str(msg.get("content") or "")


def run_arm(name: str, user_text: str, n: int) -> None:
    print(f"\n=== ARM {name} (n={n}) ===")
    counts = {"diary_fence": 0, "no_fence": 0, "native_tool_calls": 0, "http_error": 0}
    for i in range(n):
        kind, content = call(user_text)
        if kind == "http_error":
            counts["http_error"] += 1
            print(f"  [{i+1}] {content}")
            continue
        if kind == "tool_calls":
            counts["native_tool_calls"] += 1
            print(f"  [{i+1}] {content}")
            continue
        if DIARY_FENCE_RE.search(content):
            counts["diary_fence"] += 1
            frag = content[DIARY_FENCE_RE.search(content).start() : DIARY_FENCE_RE.search(content).start() + 90]
            print(f"  [{i+1}] DIARY FENCE written: {frag!r}")
        else:
            counts["no_fence"] += 1
            print(f"  [{i+1}] no fence: {' '.join(content.split())[:110]!r}")
    print(f"  --- {name} totals: {counts}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_arm("E1 (explicit diary request)", E1_USER, n)
    run_arm("E2 (salient moment, no explicit ask)", E2_USER, n)


if __name__ == "__main__":
    main()
