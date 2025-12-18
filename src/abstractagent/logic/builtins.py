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

