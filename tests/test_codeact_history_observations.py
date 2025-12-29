from __future__ import annotations

from abstractagent.logic.codeact import CodeActLogic
from abstractcore.tools.core import tool


@tool
def execute_python(code: str, timeout_s: float = 10.0) -> dict:
    """Dummy tool schema for tests."""
    return {"stdout": code, "stderr": "", "exit_code": 0}


def test_build_request_renders_tool_messages_as_observations() -> None:
    logic = CodeActLogic(tools=[execute_python._tool_definition])
    req = logic.build_request(
        task="t",
        messages=[
            {
                "role": "tool",
                "content": "[execute_python]: ok",
                "metadata": {"name": "execute_python", "success": True},
            }
        ],
        iteration=1,
        max_iterations=5,
        vars={"_limits": {"max_history_messages": -1}},
    )

    assert "observation[execute_python] (success): ok" in req.prompt
    assert "tool: [execute_python]" not in req.prompt

