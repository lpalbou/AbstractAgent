from __future__ import annotations

from abstractagent.logic.react import ReActLogic
from abstractcore.tools.core import tool


@tool
def read_file(path: str) -> str:
    """Dummy tool schema for tests."""
    return path


def test_build_request_includes_history_and_memory_instruction() -> None:
    logic = ReActLogic(tools=[read_file._tool_definition], max_history_messages=-1, max_tokens=321)
    req = logic.build_request(
        task="Do something",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        iteration=2,
        max_iterations=10,
        vars={"_limits": {"max_history_messages": -1}},
    )
    # User-role content must contain only the user's request (no internal history/memory).
    assert req.prompt == "Do something"
    assert "History:" not in req.prompt
    assert isinstance(req.system_prompt, str) and req.system_prompt.strip()
    assert "Iteration: 2/10" in req.system_prompt
    assert "autonomous ReAct agent" in req.system_prompt
    assert "user: hi" not in req.prompt
    assert "assistant: hello" not in req.prompt


def test_build_request_does_not_slice_history() -> None:
    logic = ReActLogic(tools=[read_file._tool_definition], max_history_messages=-1)
    messages = [
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
    ]
    req = logic.build_request(
        task="t",
        messages=messages,
        iteration=1,
        max_iterations=5,
        vars={"_limits": {"max_history_messages": 1}},
    )
    # Messages are internal state and must not be placed in the user-role prompt.
    assert req.prompt == "t"
    assert "m1" not in req.prompt
    assert "m2" not in req.prompt
    assert "m3" not in req.prompt


def test_build_request_renders_tool_messages_as_observations() -> None:
    logic = ReActLogic(tools=[read_file._tool_definition])
    req = logic.build_request(
        task="t",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "content": "[execute_command]: ok",
                "metadata": {"name": "execute_command", "success": True},
            },
        ],
        iteration=1,
        max_iterations=5,
        vars={"_limits": {"max_history_messages": -1}},
    )
    # Tool observations are internal and must not appear in the user-role prompt.
    assert req.prompt == "t"
    assert "execute_command" not in req.prompt


def test_parse_response_reads_native_tool_calls() -> None:
    logic = ReActLogic(tools=[read_file._tool_definition])
    content, calls = logic.parse_response(
        {
            "content": "ok",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "x"}, "call_id": "1"}],
        }
    )
    assert content == "ok"
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "x"}
