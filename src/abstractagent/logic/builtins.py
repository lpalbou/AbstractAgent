"""Built-in tool specs used by agents.

These are tool *definitions* (schemas), not executable tool callables.
"""

from __future__ import annotations

from abstractcore.tools import ToolDefinition

ASK_USER_TOOL = ToolDefinition(
    name="ask_user",
    description="Ask the user a question.",
    parameters={
        "question": {
            "type": "string",
            "description": "The question to ask the user (required)",
        },
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of choices for the user to pick from",
            "default": None,
        },
    },
    when_to_use="Use when the task is ambiguous or you need user input to proceed.",
)

RECALL_MEMORY_TOOL = ToolDefinition(
    name="recall_memory",
    description="Recall archived memory spans with provenance (by span_id/query/tags/time range).",
    parameters={
        "span_id": {
            "type": "string",
            "description": (
                "Optional span identifier (artifact id) or 1-based index into archived spans. "
                "If a summary includes span_id=..., use that exact value."
            ),
            "default": None,
        },
        "query": {
            "type": "string",
            "description": "Optional keyword query (topic/person/etc). Performs metadata-first search with bounded deep scan over archived messages.",
            "default": None,
        },
        "since": {
            "type": "string",
            "description": "Optional ISO8601 start timestamp for time-range filtering.",
            "default": None,
        },
        "until": {
            "type": "string",
            "description": "Optional ISO8601 end timestamp for time-range filtering.",
            "default": None,
        },
        "tags": {
            "type": "object",
            "description": (
                "Optional metadata tag filters.\n"
                "- Values may be a string or a list of strings.\n"
                "- Example: {\"topic\":\"api\",\"person\":[\"alice\",\"bob\"]}\n"
                "Use tags_mode to control AND/OR across tag keys."
            ),
            "default": None,
        },
        "tags_mode": {
            "type": "string",
            "description": (
                "How to combine tag keys: all (AND across keys) | any (OR across keys). "
                "Within a key, list values are treated as OR."
            ),
            "default": "all",
        },
        "usernames": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional author filter (actor ids / usernames). Matches spans created_by case-insensitively. "
                "Semantics: OR (any listed author)."
            ),
            "default": None,
        },
        "locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional location filter. Matches spans by explicit location metadata (or tags.location). "
                "Semantics: OR (any listed location)."
            ),
            "default": None,
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
        "scope": {
            "type": "string",
            "description": "Memory scope to query: run | session | global | all (default run).",
            "default": "run",
        },
    },
    when_to_use="Use after compaction or when you need exact details from earlier context.",
)

INSPECT_VARS_TOOL = ToolDefinition(
    name="inspect_vars",
    description="Inspect durable run-state variables by path (e.g., scratchpad/runtime vars).",
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
            "default": None,
        },
    },
    when_to_use=(
        "Use to debug or inspect scratchpad/runtime vars (prefer keys_only=true first)."
    ),
)

REMEMBER_TOOL = ToolDefinition(
    name="remember",
    description="Tag an archived memory span for later recall.",
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
        "Use when you want to label a recalled/compacted span with durable tags."
    ),
)

REMEMBER_NOTE_TOOL = ToolDefinition(
    name="remember_note",
    description="Store a durable memory note (decision/fact) with optional tags and sources.",
    parameters={
        "note": {
            "type": "string",
            "description": "The note to remember (required). Keep it short and specific.",
        },
        "tags": {
            "type": "object",
            "description": "Optional tags (dict[str,str]) to help recall later, e.g. {\"topic\":\"api\",\"person\":\"alice\"}.",
            "default": None,
        },
        "sources": {
            "type": "object",
            "description": (
                "Optional provenance sources for this note. Use span_ids/message_ids when available.\n"
                "Example: {\"span_ids\":[\"span_...\"], \"message_ids\":[\"msg_...\"]}"
            ),
            "default": None,
        },
        "location": {
            "type": "string",
            "description": "Optional location for this memory note (user perspective).",
            "default": None,
        },
        "scope": {
            "type": "string",
            "description": "Where to store this note: run | session | global (default run).",
            "default": "run",
        },
    },
    when_to_use=(
        "When you want to persist a key insight/decision/fact for later recall by time/topic/person, "
        "especially before any compaction span exists."
    ),
)

COMPACT_MEMORY_TOOL = ToolDefinition(
    name="compact_memory",
    description="Compact older conversation context into an archived span and insert a summary handle.",
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
            "default": None,
        },
    },
    when_to_use="Use when the active context is too large and you need to reduce it while keeping provenance.",
)

COMPACT_ACTIVE_MEMORY_TOOL = ToolDefinition(
    name="compact_active_memory",
    description="Compact Active Memory lists into an archived span (preserving provenance).",
    parameters={
        "components": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of components to compact. Supported: "
                "current_tasks, current_context, critical_insights, key_history."
            ),
            "default": None,
        },
        "preserve": {
            "type": "object",
            "description": (
                "Optional per-component keep counts (keep newest N items). "
                "Example: {\"key_history\": 50, \"critical_insights\": 30}."
            ),
            "default": None,
        },
    },
    when_to_use=(
        "Use when Active Memory lists are growing too large and you need to archive older items."
    ),
)

# ---------------------------------------------------------------------------
# Structured Active Memory editing tools (runtime-owned; schema-only)
# ---------------------------------------------------------------------------
ACTIVE_MEMORY_DELTA_TOOL = ToolDefinition(
    name="active_memory_delta",
    description="Apply a delta patch to Structured Active Memory (tasks/context/insights/history).",
    parameters={
        "current_tasks": {
            "type": "object",
            "description": "Optional patch for Current Tasks: {clear?: bool, remove?: [task_id], upsert?: [task_obj|title_str]}.",
            "default": None,
        },
        "current_context": {
            "type": "object",
            "description": "Optional patch for Current Context: {clear?: bool, remove?: [context_id], upsert?: [context_obj|title_str]}.",
            "default": None,
        },
        "critical_insights": {
            "type": "object",
            "description": "Optional patch for Critical Insights: {clear?: bool, remove?: [insight_id], add?: [insight_obj|text_str]}.",
            "default": None,
        },
        "key_history": {
            "type": "object",
            "description": "Optional patch for Key History: {clear?: bool, remove?: [event_id], add?: [event_obj|summary_str]}.",
            "default": None,
        },
    },
    when_to_use="Use to keep Active Memory up to date as you progress (update tasks/context/insights/history).",
)

# Optional granular aliases (models sometimes try these names directly).
CURRENT_TASKS_TOOL = ToolDefinition(
    name="current_tasks",
    description="Patch the Active Memory Current Tasks module.",
    parameters={
        "clear": {"type": "boolean", "description": "If true, clears all tasks.", "default": False},
        "remove": {"type": "array", "items": {"type": "string"}, "description": "Task IDs to remove.", "default": None},
        "upsert": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Tasks to upsert (task objects). String shorthands are allowed by the runtime.",
            "default": None,
        },
    },
    when_to_use="Use to update the Active Memory Current Tasks list.",
)

CURRENT_CONTEXT_TOOL = ToolDefinition(
    name="current_context",
    description="Patch the Active Memory Current Context module.",
    parameters={
        "clear": {"type": "boolean", "description": "If true, clears all context items.", "default": False},
        "remove": {"type": "array", "items": {"type": "string"}, "description": "Context IDs to remove.", "default": None},
        "upsert": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Context items to upsert (context objects). String shorthands are allowed by the runtime.",
            "default": None,
        },
    },
    when_to_use="Use to update the Active Memory Current Context list.",
)

CRITICAL_INSIGHTS_TOOL = ToolDefinition(
    name="critical_insights",
    description="Patch the Active Memory Critical Insights module.",
    parameters={
        "clear": {"type": "boolean", "description": "If true, clears all critical insights.", "default": False},
        "remove": {"type": "array", "items": {"type": "string"}, "description": "Insight IDs to remove.", "default": None},
        "add": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Insights to add (insight objects). String shorthands are allowed by the runtime.",
            "default": None,
        },
    },
    when_to_use="Use to update the Active Memory Critical Insights list.",
)

KEY_HISTORY_TOOL = ToolDefinition(
    name="key_history",
    description="Patch the Active Memory Key History module.",
    parameters={
        "clear": {"type": "boolean", "description": "If true, clears key history.", "default": False},
        "remove": {"type": "array", "items": {"type": "string"}, "description": "Event IDs to remove.", "default": None},
        "add": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Events to add (event objects). String shorthands are allowed by the runtime.",
            "default": None,
        },
    },
    when_to_use="Use to append/update Key History with durable, natural-language events.",
)
