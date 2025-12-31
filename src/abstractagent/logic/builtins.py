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
            "description": "Maximum total messages to render in the recall output across all spans (-1 = no truncation).",
            "default": -1,
        },
    },
    when_to_use=(
        "When conversation history was compacted/summarized and you need the original messages, "
        "or when you need exact details from prior discussions."
    ),
)

INSPECT_VARS_TOOL = ToolDefinition(
    name="inspect_vars",
    description=(
        "Inspect durable run state variables (especially scratchpad) by path. "
        "Use this for progressive recall/debugging when you need to see what the workflow/agent stored "
        "outside of the active conversation context."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": (
                "Path to inspect (default 'scratchpad'). Supports dot paths like 'scratchpad.foo[0]' "
                "or JSON pointer paths like '/scratchpad/foo/0'."
            ),
            "default": "scratchpad",
        },
        "keys_only": {
            "type": "boolean",
            "description": "If true, return keys/length instead of the full value (useful to navigate large objects).",
            "default": False,
        },
        "target_run_id": {
            "type": "string",
            "description": "Optional run id to inspect (defaults to the current run).",
        },
    },
    when_to_use=(
        "When you need to inspect scratchpad/runtime vars for debugging or progressive recall. "
        "Prefer keys_only=true first to discover available fields, then retrieve a deeper path."
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

REMEMBER_NOTE_TOOL = ToolDefinition(
    name="remember_note",
    description=(
        "Store a durable memory note (decision/fact) with optional tags and provenance sources. "
        "Use this to remember something memorable without requiring compaction."
    ),
    parameters={
        "note": {
            "type": "string",
            "description": "The note to remember (required). Keep it short and specific.",
        },
        "tags": {
            "type": "object",
            "description": "Optional tags (dict[str,str]) to help recall later, e.g. {\"topic\":\"api\",\"person\":\"alice\"}.",
        },
        "sources": {
            "type": "object",
            "description": (
                "Optional provenance sources for this note. Use span_ids/message_ids when available.\n"
                "Example: {\"span_ids\":[\"span_...\"], \"message_ids\":[\"msg_...\"]}"
            ),
        },
    },
    when_to_use=(
        "When you want to persist a key insight/decision/fact for later recall by time/topic/person, "
        "especially before any compaction span exists."
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

COMPACT_ACTIVE_MEMORY_TOOL = ToolDefinition(
    name="compact_active_memory",
    description=(
        "Compact Structured Active Memory (current tasks/context/insights/history) while preserving provenance. "
        "This archives overflow items per component into a span (ArtifactStore) and appends a Key History event "
        "with `span_id=...` so you can later recall the original items via recall_memory."
    ),
    parameters={
        "components": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of components to compact. Supported: "
                "current_tasks, current_context, critical_insights, key_history."
            ),
        },
        "preserve": {
            "type": "object",
            "description": (
                "Optional per-component keep counts (keep newest N items). "
                "Example: {\"key_history\": 50, \"critical_insights\": 30}."
            ),
        },
    },
    when_to_use=(
        "When Active Memory lists (especially critical_insights/key_history) are growing too large and you want "
        "to archive older items into a span with a durable handle (span_id)."
    ),
)
