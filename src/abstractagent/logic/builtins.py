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

OPEN_ATTACHMENT_TOOL = ToolDefinition(
    name="open_attachment",
    description="Read an attachment (artifact) from the current session with bounded output.",
    parameters={
        "artifact_id": {"type": "string", "description": "Attachment artifact id (preferred).", "default": None},
        "handle": {"type": "string", "description": "Attachment handle (usually '@path').", "default": None},
        "expected_sha256": {"type": "string", "description": "Optional sha256 to disambiguate versions.", "default": None},
        "start_line": {"type": "integer", "description": "1-based start line (default 1).", "default": 1},
        "end_line": {"type": "integer", "description": "Optional end line (inclusive).", "default": None},
        "max_chars": {"type": "integer", "description": "Maximum characters to return (default 8000).", "default": 8000},
    },
    when_to_use=(
        "Use to re-open a previously attached file without re-sending its full contents in the chat context. "
        "Prefer artifact_id when available; otherwise use handle."
    ),
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
        "recall_level": {
            "type": "string",
            "enum": ["urgent", "standard", "deep"],
            "description": (
                "Optional recall effort policy.\n"
                "- urgent: fastest, tight budgets\n"
                "- standard: bounded but more thorough\n"
                "- deep: may be slow/expensive\n"
                "No silent downgrade: budgets/permissions are enforced by the runtime."
            ),
            "default": None,
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

DELEGATE_AGENT_TOOL = ToolDefinition(
    name="delegate_agent",
    description="Delegate a subtask to a fresh agent run with smaller context and restricted tools.",
    parameters={
        "task": {
            "type": "string",
            "description": "The delegated task (required). Keep it specific and self-contained.",
        },
        "context": {
            "type": "string",
            "description": "Minimal supporting context for the delegated task (optional).",
            "default": "",
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Tool allowlist for the delegated agent (optional). "
                "When omitted, the delegated agent inherits the current allowlist."
            ),
            "default": None,
        },
    },
    when_to_use=(
        "Use when you can split off a subtask that can be completed with a small context (e.g., "
        "inspect a few files, search for specific symbols, summarize findings)."
    ),
)
