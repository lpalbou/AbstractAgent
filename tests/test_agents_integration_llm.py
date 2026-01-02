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


if str(os.getenv("ABSTRACTAGENT_RUN_INTEGRATION_TESTS") or "").strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "Skipping real-LLM integration tests (set ABSTRACTAGENT_RUN_INTEGRATION_TESTS=1 to enable).",
        allow_module_level=True,
    )


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


def _skip_if_llm_error_text(text: str) -> None:
    """Skip when the provider returns an error as plain text (no exception raised)."""
    msg = (text or "").lower()
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
            "error:",
        )
    ):
        pytest.skip(f"Local LLM not available (error text): {text}")


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
    _skip_if_llm_error_text(answer)
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
    _skip_if_llm_error_text(answer)
    assert expected in answer

    messages = (state.output or {}).get("messages") or []
    assert isinstance(messages, list)
    tool_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and (m.get("metadata") or {}).get("name") == "execute_python"
    ]
    assert tool_msgs, "Expected at least one tool message for execute_python."


@pytest.mark.integration
def test_react_agent_can_recall_archived_span_with_real_llm() -> None:
    """End-to-end: model calls recall_memory and receives archived messages."""
    from abstractagent.agents.react import create_react_agent

    provider, model, llm_kwargs = _llm_config()

    try:
        agent = create_react_agent(
            provider=provider,
            model=model,
            tools=[],  # keep toolset minimal; recall_memory is schema-only and handled by the adapter/runtime
            max_iterations=8,
            max_tokens=8192,
            llm_kwargs=llm_kwargs,
        )
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    artifact_store = agent.runtime.artifact_store
    if artifact_store is None:
        pytest.skip("ArtifactStore not configured on runtime")

    sentinel = f"sentinel_{uuid.uuid4().hex}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Original memory contains: {sentinel}",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "metadata": {"message_id": "m1"},
            }
        ],
        "span": {"from_timestamp": "2025-01-01T00:00:00+00:00", "to_timestamp": "2025-01-01T00:00:00+00:00", "message_count": 1},
        "created_at": "2025-01-01T00:00:01+00:00",
    }
    meta = artifact_store.store_json(payload, run_id="run", tags={"kind": "conversation_span"})

    agent.start(
        "You do not know the sentinel value.\n"
        "The sentinel exists ONLY inside an archived memory span.\n"
        "You MUST call the recall_memory tool to retrieve it.\n"
        f"Call recall_memory(span_id='{meta.artifact_id}', max_messages=20).\n"
        "Then return ONLY the sentinel string found in the recalled messages (no extra text)."
    )

    # Inject the span index into the run so recall has provenance context.
    run_id = agent.run_id
    assert run_id is not None
    run = agent.runtime.run_store.load(run_id)
    assert run is not None
    runtime_ns = run.vars.setdefault("_runtime", {})
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        run.vars["_runtime"] = runtime_ns
    runtime_ns["memory_spans"] = [
        {
            "kind": "conversation_span",
            "artifact_id": meta.artifact_id,
            "created_at": "2025-01-01T00:00:01+00:00",
            "from_timestamp": "2025-01-01T00:00:00+00:00",
            "to_timestamp": "2025-01-01T00:00:00+00:00",
            "message_count": 1,
            "tags": {"topic": "test"},
        }
    ]
    agent.runtime.run_store.save(run)

    try:
        state = agent.run_to_completion()
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    if state.status.value == "failed":
        _skip_if_llm_unavailable(RuntimeError(state.error or "unknown error"))
        pytest.fail(f"Run failed unexpectedly: {state.error}")

    answer = str((state.output or {}).get("answer") or "").strip()
    _skip_if_llm_error_text(answer)
    assert answer == sentinel


@pytest.mark.integration
def test_react_agent_can_tag_span_with_real_llm() -> None:
    """End-to-end: model calls remember and runtime updates span tags."""
    from abstractagent.agents.react import create_react_agent

    provider, model, llm_kwargs = _llm_config()

    try:
        agent = create_react_agent(
            provider=provider,
            model=model,
            tools=[],  # keep toolset minimal; remember is schema-only and handled by the adapter/runtime
            max_iterations=6,
            max_tokens=8192,
            llm_kwargs=llm_kwargs,
        )
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    artifact_store = agent.runtime.artifact_store
    if artifact_store is None:
        pytest.skip("ArtifactStore not configured on runtime")

    # Create a dummy span artifact and seed the span index into the run.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Original memory about API ownership.",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "metadata": {"message_id": "m1"},
            }
        ],
        "span": {"from_timestamp": "2025-01-01T00:00:00+00:00", "to_timestamp": "2025-01-01T00:00:00+00:00", "message_count": 1},
        "created_at": "2025-01-01T00:00:01+00:00",
    }
    meta = artifact_store.store_json(payload, run_id="run", tags={"kind": "conversation_span"})

    agent.start(
        "You must tag the archived memory span with topic=api.\n"
        "Call the remember tool exactly once:\n"
        f"remember(span_id='{meta.artifact_id}', tags={{\"topic\":\"api\"}})\n"
        "Then respond with exactly: done"
    )

    run_id = agent.run_id
    assert run_id is not None
    run = agent.runtime.run_store.load(run_id)
    assert run is not None
    runtime_ns = run.vars.setdefault("_runtime", {})
    if not isinstance(runtime_ns, dict):
        runtime_ns = {}
        run.vars["_runtime"] = runtime_ns
    runtime_ns["memory_spans"] = [
        {
            "kind": "conversation_span",
            "artifact_id": meta.artifact_id,
            "created_at": "2025-01-01T00:00:01+00:00",
            "from_timestamp": "2025-01-01T00:00:00+00:00",
            "to_timestamp": "2025-01-01T00:00:00+00:00",
            "message_count": 1,
        }
    ]
    agent.runtime.run_store.save(run)

    try:
        state = agent.run_to_completion()
    except Exception as e:
        _skip_if_llm_unavailable(e)
        raise

    if state.status.value == "failed":
        _skip_if_llm_unavailable(RuntimeError(state.error or "unknown error"))
        pytest.fail(f"Run failed unexpectedly: {state.error}")

    answer = str((state.output or {}).get("answer") or "").strip()
    _skip_if_llm_error_text(answer)
    assert answer == "done"

    updated = agent.runtime.run_store.load(run_id)
    assert updated is not None
    spans = (updated.vars.get("_runtime") or {}).get("memory_spans")
    assert isinstance(spans, list) and spans
    tags = spans[0].get("tags")
    assert tags == {"topic": "api"}

    messages = (state.output or {}).get("messages") or []
    assert isinstance(messages, list)
    tool_msgs = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and (m.get("metadata") or {}).get("name") == "recall_memory"
    ]
    assert tool_msgs, "Expected at least one tool message for recall_memory."
    assert sentinel in str(tool_msgs[-1].get("content") or "")
