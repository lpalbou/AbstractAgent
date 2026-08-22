from __future__ import annotations

from abstractagent.adapters.read_orchestration import advance_last_successful_read_batch
from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_READ = ToolDefinition(name="read_file", description="read", parameters={})
_SEARCH = ToolDefinition(name="search_files", description="search", parameters={})


def _react_run(*, start: int, end: int, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="react-read",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "iteration": 1,
                "cycles": [],
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "app.py", "start_line": start, "end_line": end}, "call_id": "c2"}],
                }
            },
            "_limits": {"max_iterations": 5, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        },
    )


def _codeact_run(*, start: int, end: int, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="codeact-read",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "app.py", "start_line": start, "end_line": end}, "call_id": "c2"}],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )


def _previous_read_batch(start: int, end: int) -> dict[str, object]:
    return {
        "count": 1,
        "same_path": True,
        "only_slices": True,
        "items": [{"path": "app.py", "start": start, "end": end, "is_slice": True}],
    }


def _previous_full_read_batch() -> dict[str, object]:
    return {
        "count": 1,
        "same_path": True,
        "only_slices": False,
        "non_read_batches_since": 1,
        "items": [{"path": "app.py", "start": 1, "end": None, "is_slice": False}],
    }


def _react_full_read_run(*, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="react-full-read",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "iteration": 1,
                "cycles": [],
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}, "call_id": "c2"}],
                }
            },
            "_limits": {"max_iterations": 5, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        },
    )


def _codeact_full_read_run(*, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="codeact-full-read",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}, "call_id": "c2"}],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )


def _react_mixed_full_read_run(*, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="react-mixed-full-read",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "iteration": 1,
                "cycles": [],
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file", "search_files"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [
                        {"name": "search_files", "arguments": {"query": "MEADOW_EAST"}, "call_id": "s1"},
                        {"name": "search_files", "arguments": {"query": "FOREST_GHOST"}, "call_id": "s2"},
                        {"name": "search_files", "arguments": {"query": "RIVER_FISHER"}, "call_id": "s3"},
                        {"name": "read_file", "arguments": {"path": "app.py"}, "call_id": "c2"},
                    ],
                }
            },
            "_limits": {"max_iterations": 5, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        },
    )


def _codeact_mixed_full_read_run(*, previous: dict[str, object]) -> RunState:
    return RunState(
        run_id="codeact-mixed-full-read",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {
                "last_successful_read_batch": previous,
            },
            "_runtime": {"inbox": [], "allowed_tools": ["read_file", "search_files"]},
            "_temp": {
                "llm_response": {
                    "content": "",
                    "tool_calls": [
                        {"name": "search_files", "arguments": {"query": "MEADOW_EAST"}, "call_id": "s1"},
                        {"name": "search_files", "arguments": {"query": "FOREST_GHOST"}, "call_id": "s2"},
                        {"name": "search_files", "arguments": {"query": "RIVER_FISHER"}, "call_id": "s3"},
                        {"name": "read_file", "arguments": {"path": "app.py"}, "call_id": "c2"},
                    ],
                }
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )


def test_react_parse_redirects_nearby_same_file_slice_to_reason_with_guidance() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_run(start=101, end=200, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "wider contiguous read_file range" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_react_parse_allows_distant_same_file_slice_to_execute() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_run(start=1000, end=1100, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending


def test_react_parse_redirects_same_file_full_reread_to_reason_with_guidance() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "do not request another full-file read_file call" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_react_parse_redirects_mixed_batch_same_file_full_reread_to_reason_with_guidance() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ, _SEARCH]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["read_file", "search_files"],
    )
    run = _react_mixed_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "do not request another full-file read_file call" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_codeact_parse_redirects_nearby_same_file_slice_to_reason_with_guidance() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_run(start=101, end=200, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "wider contiguous read_file range" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_codeact_parse_allows_distant_same_file_slice_to_execute() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_run(start=1000, end=1100, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending


def test_codeact_parse_redirects_same_file_full_reread_to_reason_with_guidance() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "do not request another full-file read_file call" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_codeact_parse_redirects_mixed_batch_same_file_full_reread_to_reason_with_guidance() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ, _SEARCH]), on_step=None)
    run = _codeact_mixed_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "reason"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "do not request another full-file read_file call" in str(inbox[-1].get("content") or "")
    assert not run.vars["_temp"].get("pending_tool_calls")


def test_advance_last_successful_read_batch_preserves_previous_read_across_non_read_batch() -> None:
    previous = {
        "count": 1,
        "same_path": True,
        "only_slices": False,
        "non_read_batches_since": 0,
        "items": [{"path": "app.py", "start": 1, "end": None, "is_slice": False}],
    }
    current_calls = [{"name": "search_files", "arguments": {"query": "MEADOW_EAST"}, "call_id": "s1"}]
    results = [{"success": True, "name": "search_files", "output": "[]"}]

    advanced = advance_last_successful_read_batch(previous, current_calls, results)
    assert isinstance(advanced, dict)
    assert advanced["non_read_batches_since"] == 1
    assert advanced["items"] == previous["items"]
