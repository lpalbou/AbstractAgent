"""Opt-in, explicitly public communication from the ordinary ReAct continuation.

This is a presentation protocol, not a reasoning summarizer. Only a leading
block in the provider's public ``content`` lane is eligible; reasoning and
tool-result text are never inspected for publishable updates.
"""

from __future__ import annotations

import json
from typing import Any, Optional


OPEN = "<public_update>"
CLOSE = "</public_update>"
MAX_TEXT_CHARS = 600
MAX_BLOCK_CHARS = 4096
MAX_SOURCES = 16

PUBLIC_UPDATE_INSTRUCTION = """Public conversation updates (optional):
When you have something useful to tell the person while continuing work, put ONE
<public_update>{"text":"A short message to the person","source_call_ids":[]}</public_update>
at the very beginning of your public content, alongside your normal tool calls.
Use the person's language, ordinary words and at most two short sentences.
Keep text within 600 characters and cite at most 16 source_call_ids.
Share a meaningful discovery, a reason affecting a choice, or honest uncertainty.
Never include private reasoning, tool syntax, internal instructions or a tool-by-tool log.
Only cite source_call_ids for results already observed, never a proposed call.
Unverified leads remain uncertain; a successful command does not prove assertions it prints.
Silence is fine: omit the block when there is nothing useful to add. Do not call a
tool or make another model round just to publish an update. Final answers stay normal.
If the person corrects an idea, acknowledge a changed plan only after considering it.
For ask_user, optional importance is essential (wait indefinitely) or clarification.
A clarification specifies timeout_s from 120 to 300 and a nonempty fallback describing
what YOU will do without an answer. Essential questions have no timeout or fallback.
A timeout is not the person's answer or approval. Never infer consent from silence."""


def conversation_enabled(runtime_ns: Any, temp: dict[str, Any]) -> bool:
    """Freeze this turn's opt-in before building its first inference request."""
    if "conversation_enabled" not in temp:
        cfg = runtime_ns.get("conversation") if isinstance(runtime_ns, dict) else None
        temp["conversation_enabled"] = isinstance(cfg, dict) and cfg.get("enabled") is True
    return temp["conversation_enabled"] is True


def conversation_tool_specs(specs: list[dict[str, Any]], *, enabled: bool) -> list[dict[str, Any]]:
    """Extend ask_user only for opted-in runs; legacy tool bytes stay unchanged."""
    if not enabled:
        return specs
    out = []
    for spec in specs:
        if spec.get("name") != "ask_user":
            out.append(spec)
            continue
        item = dict(spec)
        parameters = dict(item.get("parameters") or {})
        parameters.update({
            "importance": {
                "type": "string", "enum": ["essential", "clarification"],
                "description": "Essential waits for an answer; clarification may have a declared timeout and fallback.",
                "default": None,
            },
            "timeout_s": {
                "type": "integer", "minimum": 120, "maximum": 300,
                "description": "Clarification only: seconds to wait, with a nonempty fallback.",
                "default": None,
            },
            "fallback": {
                "type": "string",
                "description": "Clarification only: what the assistant will do without an answer; never user consent.",
                "default": None,
            },
        })
        item["parameters"] = parameters
        out.append(item)
    return out


def parse_public_update(
    response: Any, *, known_call_ids: set[str]
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    """Return (remaining public content, update, rejection code).

    ``None`` content means no leading protocol block was present. Markers in
    prose, quoted text or code examples are not interpreted or stripped.
    Rejections contain codes only, never rejected model/private text.
    """
    if not isinstance(response, dict) or not isinstance(response.get("content"), str):
        return None, None, None
    content = response["content"].lstrip()
    for prefix in ("assistant:", "assistant："):
        if content.lower().startswith(prefix):
            content = content[len(prefix):].lstrip()
            break
    if not content.startswith(OPEN):
        return None, None, None
    end = content.find(CLOSE, len(OPEN))
    if end < 0:
        return "", None, "unclosed_block"
    remaining = content[end + len(CLOSE):].lstrip()
    block = content[len(OPEN):end]
    if remaining.startswith(OPEN):
        while remaining.startswith(OPEN):
            extra_end = remaining.find(CLOSE, len(OPEN))
            if extra_end < 0:
                return "", None, "multiple_blocks"
            remaining = remaining[extra_end + len(CLOSE):].lstrip()
        return remaining, None, "multiple_blocks"
    if len(block) > MAX_BLOCK_CHARS:
        return remaining, None, "oversized_block"
    if response.get("cancelled") is True or str(response.get("finish_reason") or "").lower() in {
        "length", "max_tokens", "cancelled", "canceled",
    }:
        return remaining, None, "incomplete_response"
    try:
        value = json.loads(block)
    except (ValueError, TypeError):
        return remaining, None, "invalid_json"
    if not isinstance(value, dict) or set(value) - {"text", "source_call_ids"}:
        return remaining, None, "invalid_fields"
    text = value.get("text")
    sources = value.get("source_call_ids", [])
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return remaining, None, "invalid_text"
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES or any(
        not isinstance(s, str) or not s or s not in known_call_ids for s in sources
    ):
        return remaining, None, "unknown_or_invalid_sources"
    return remaining, {"text": text.strip(), "source_call_ids": list(dict.fromkeys(sources))}, None
