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
    assert "History:" in req.prompt
    assert "Do not claim you have no memory" in req.prompt
    assert "user: hi" in req.prompt
    assert "assistant: hello" in req.prompt


def test_build_request_applies_max_history_limit() -> None:
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
    assert "user: m3" in req.prompt
    assert "assistant: m2" not in req.prompt
    assert "user: m1" not in req.prompt


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

