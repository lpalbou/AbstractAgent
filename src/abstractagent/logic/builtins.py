"""Built-in tool specs used by agents.

These are tool *definitions* (schemas), not executable tool callables.
"""

from __future__ import annotations

from abstractcore.tools import ToolDefinition

ASK_USER_TOOL = ToolDefinition(
    name="ask_user",
    description=(
        "Ask the user a question when you need clarification or input. "
        "Use this when the task is ambiguous or you need the user to make a choice."
    ),
    parameters={
        "question": {
            "type": "string",
            "description": "The question to ask the user (required)",
        },
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of choices for the user to pick from",
        },
    },
    when_to_use="When the task is ambiguous or you need user input to proceed",
)

RECALL_MEMORY_TOOL = ToolDefinition(
    name="recall_memory",
    description=(
        "Recall original memory from archived spans with provenance. "
        "Use this to reconstruct details after compaction or when you need exact prior context. "
        "Prefer recalling by span_id when available."
    ),
    parameters={
        "span_id": {
            "type": "string",
            "description": (
                "Optional span identifier (artifact id) or 1-based index into archived spans. "
                "If a summary includes span_id=..., use that exact value."
            ),
        },
        "query": {
            "type": "string",
            "description": "Optional keyword query (topic/person/etc). Performs metadata-first search with bounded deep scan over archived messages.",
        },
        "since": {
            "type": "string",
            "description": "Optional ISO8601 start timestamp for time-range filtering.",
        },
        "until": {
            "type": "string",
            "description": "Optional ISO8601 end timestamp for time-range filtering.",
        },
        "tags": {
            "type": "object",
            "description": "Optional metadata tag filters (e.g., {\"topic\":\"api\",\"person\":\"alice\"}).",
        },
        "limit_spans": {
            "type": "integer",
            "description": "Maximum number of spans to return (default 5).",
            "default": 5,
        },
        "connected": {
            "type": "boolean",
            "description": "If true, also include connected spans (time neighbors and shared-tag neighbors).",
            "default": False,
        },
        "neighbor_hops": {
            "type": "integer",
            "description": "When connected=true, include up to this many neighbor spans on each side (default 1).",
            "default": 1,
        },
        "max_messages": {
            "type": "integer",
            "description": "Maximum total messages to render in the recall output across all spans (default 80).",
            "default": 80,
        },
    },
    when_to_use=(
        "When conversation history was compacted/summarized and you need the original messages, "
        "or when you need exact details from prior discussions."
    ),
)

REMEMBER_TOOL = ToolDefinition(
    name="remember",
    description=(
        "Remember something by applying durable tags (topic/person/etc) to an archived memory span. "
        "Use this after compaction when you want to reliably find the span later via recall_memory(tags=...)."
    ),
    parameters={
        "span_id": {
            "type": "string",
            "description": (
                "Span identifier (artifact id) or 1-based index into archived spans. "
                "If a summary includes span_id=..., use that exact value."
            ),
        },
        "tags": {
            "type": "object",
            "description": (
                "Tags to set on the span (JSON-safe dict[str,str]), e.g. {\"topic\":\"api\",\"person\":\"alice\"}. "
                "At least one tag is required."
            ),
        },
        "merge": {
            "type": "boolean",
            "description": "If true (default), merges tags into existing tags. If false, replaces existing tags.",
            "default": True,
        },
    },
    when_to_use=(
        "When you want to label a recalled/compacted span with durable metadata so you can find it later by tags."
    ),
)

COMPACT_MEMORY_TOOL = ToolDefinition(
    name="compact_memory",
    description=(
        "Compact older conversation context to reduce active memory usage while preserving provenance. "
        "This archives older messages into a span (ArtifactStore) and inserts a system summary that includes "
        "`span_id=...` so you can later reconstruct details via recall_memory."
    ),
    parameters={
        "preserve_recent": {
            "type": "integer",
            "description": "Number of most recent non-system messages to keep verbatim (default 6).",
            "default": 6,
        },
        "compression_mode": {
            "type": "string",
            "description": "Compression mode: light | standard | heavy (default standard).",
            "default": "standard",
        },
        "focus": {
            "type": "string",
            "description": "Optional focus/topic to prioritize in the summary.",
        },
    },
    when_to_use=(
        "When the active context is getting too large and you need to reduce it while keeping the full sources recoverable."
    ),
)
