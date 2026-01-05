from __future__ import annotations

from abstractagent.logic.codeact import CodeActLogic
from abstractcore.tools.core import tool


@tool
def execute_python(code: str, timeout_s: float = 10.0) -> dict:
    """Dummy tool schema for tests."""
    return {"stdout": code, "stderr": "", "exit_code": 0}


def test_extract_code_from_python_block() -> None:
    logic = CodeActLogic(tools=[execute_python._tool_definition])
    code = logic.extract_code(
        "Here is code:\n```python\nprint('hello')\n```\nAnd more text"
    )
    assert code == "print('hello')"


def test_build_request_includes_codeact_instructions() -> None:
    logic = CodeActLogic(tools=[execute_python._tool_definition], max_history_messages=-1, max_tokens=123)
    req = logic.build_request(
        task="Compute something",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        iteration=2,
        max_iterations=10,
        vars={"_limits": {"max_history_messages": 1}},
    )
    assert isinstance(req.system_prompt, str) and req.system_prompt.strip()
    assert "You are CodeAct" in req.system_prompt
    assert "execute_python" in req.system_prompt
    assert req.prompt == "Compute something"


def test_format_observation_execute_python_dict() -> None:
    logic = CodeActLogic(tools=[execute_python._tool_definition])
    rendered = logic.format_observation(
        name="execute_python",
        output={"stdout": "ok\n", "stderr": "", "exit_code": 0, "error": None},
        success=True,
    )
    assert "stdout" in rendered
    assert "ok" in rendered
