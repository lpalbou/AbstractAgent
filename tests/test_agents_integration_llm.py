"""Real-LLM integration tests for AbstractAgent (no mocks).

These tests validate the actual ReAct/CodeAct workflows end-to-end:
- LLM call
- Tool call parsing
- TOOL_CALLS execution via ToolExecutor
- Final answer completion

The tests are skipped if no local LLM is reachable (e.g., Ollama/LMStudio not running).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from abstractcore.tools.core import tool


def _llm_config() -> Tuple[str, str, Dict[str, Any]]:
    provider = os.getenv("ABSTRACTAGENT_TEST_PROVIDER", "ollama")
    model = os.getenv("ABSTRACTAGENT_TEST_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    base_url = os.getenv("ABSTRACTAGENT_TEST_BASE_URL")

    llm_kwargs: Dict[str, Any] = {"temperature": 0}
    # Some local providers accept a seed (safe to ignore if unsupported).
    llm_kwargs["seed"] = 42
    if base_url:
        llm_kwargs["base_url"] = base_url
    return provider, model, llm_kwargs


def _skip_if_llm_unavailable(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(
        keyword in msg
        for keyword in (
            "connection",
            "refused",
            "timeout",
            "timed out",
            "not running",
            "operation not permitted",
            "no such host",
            "not found",
            "model not found",
            "pull",
            "failed to connect",
        )
    ):
        pytest.skip(f"Local LLM not available: {exc}")


@pytest.mark.integration
def test_react_agent_reads_file_with_real_llm(tmp_path: Path) -> None:
    from abstractagent.agents.react import create_react_agent

    provider, model, llm_kwargs = _llm_config()

    sentinel = f"sentinel_{uuid.uuid4().hex}"
    target = tmp_path / "sentinel.txt"
    target.write_text(sentinel + "\n", encoding="utf-8")

    @tool(name="read_file", description="Read a UTF-8 text file and return its content.")
    def read_file(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    try:
        agent = create_react_agent(
            provider=provider,
            model=model,
            tools=[read_file],
            max_iterations=10,
            max_tokens=8192,
            llm_kwargs=llm_kwargs,
        )
        agent.start(
            "You do not know the file content.\n"
            "Use the read_file tool to read it and then return the exact content.\n"
            f"path={target}\n"
        )
        state = agent.run_to_completion()
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    if state.status.value == "failed":
        _skip_if_llm_unavailable(RuntimeError(state.error or "unknown error"))
        pytest.fail(f"Run failed unexpectedly: {state.error}")

    assert state.status.value == "completed"
    answer = str((state.output or {}).get("answer") or "")
    assert sentinel in answer

    messages = (state.output or {}).get("messages") or []
    assert isinstance(messages, list)
    tool_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and (m.get("metadata") or {}).get("name") == "read_file"
    ]
    assert tool_msgs, "Expected at least one tool message for read_file."


@pytest.mark.integration
def test_codeact_agent_executes_python_with_real_llm(tmp_path: Path) -> None:
    from abstractagent.agents.codeact import create_codeact_agent

    provider, model, llm_kwargs = _llm_config()

    payload = f"payload_{uuid.uuid4().hex}"
    target = tmp_path / "payload.txt"
    target.write_text(payload + "\n", encoding="utf-8")
    expected = hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()

    try:
        agent = create_codeact_agent(
            provider=provider,
            model=model,
            max_iterations=10,
            max_tokens=8192,
            llm_kwargs=llm_kwargs,
        )
        agent.start(
            "Compute the SHA256 of the exact UTF-8 file content (including the trailing newline).\n"
            "You must use execute_python (or a fenced ```python block if tool calling is unavailable).\n"
            f"file_path={target}\n"
            "Return ONLY the hex sha256 string."
        )
        state = agent.run_to_completion()
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    if state.status.value == "failed":
        _skip_if_llm_unavailable(RuntimeError(state.error or "unknown error"))
        pytest.fail(f"Run failed unexpectedly: {state.error}")

    assert state.status.value == "completed"
    answer = str((state.output or {}).get("answer") or "").strip()
    assert expected in answer

    messages = (state.output or {}).get("messages") or []
    assert isinstance(messages, list)
    tool_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and (m.get("metadata") or {}).get("name") == "execute_python"
    ]
    assert tool_msgs, "Expected at least one tool message for execute_python."
