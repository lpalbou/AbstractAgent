from __future__ import annotations

from abstractagent.adapters.read_orchestration import (
    advance_last_successful_read_batch,
    detect_nearby_same_file_staircase,
)
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


def test_react_parse_advises_on_nearby_same_file_slice_but_still_reads() -> None:
    """The staircase hint is ADVICE: the guidance lands and the read runs.

    Regression for run 9ea71c55-4405-4b6f-b387-3cc78629880b, where firing this
    hint dropped the batch, returned zero observations and spent a full task
    iteration. `read_file` has no side effects — there is nothing to protect
    by refusing it.
    """
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_run(start=101, end=200, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "wider contiguous read_file range" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending, "the read must still execute"


def test_react_parse_allows_distant_same_file_slice_to_execute() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_run(start=1000, end=1100, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending


def test_react_parse_advises_on_same_file_full_reread_but_still_reads() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    run = _react_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "already read this file successfully" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending, "the read must still execute"


def test_react_parse_advises_on_mixed_batch_full_reread_but_still_reads() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ, _SEARCH]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["read_file", "search_files"],
    )
    run = _react_mixed_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "already read this file successfully" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    # The three search_files calls in this batch were collateral damage of the
    # old refusal: an unrelated, useful batch was thrown away because ONE of
    # its calls looked like a re-read.
    assert isinstance(pending, list) and len(pending) == 4


def test_codeact_parse_advises_on_nearby_same_file_slice_but_still_reads() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_run(start=101, end=200, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "wider contiguous read_file range" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending, "the read must still execute"


def test_codeact_parse_allows_distant_same_file_slice_to_execute() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_run(start=1000, end=1100, previous=_previous_read_batch(1, 100))

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending


def test_codeact_parse_advises_on_same_file_full_reread_but_still_reads() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ]), on_step=None)
    run = _codeact_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "already read this file successfully" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and pending, "the read must still execute"


def test_codeact_parse_advises_on_mixed_batch_full_reread_but_still_reads() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_READ, _SEARCH]), on_step=None)
    run = _codeact_mixed_full_read_run(previous=_previous_full_read_batch())

    plan = wf.get_node("parse")(run, _Ctx())
    assert plan.next_node == "act"
    inbox = run.vars["_runtime"]["inbox"]
    assert isinstance(inbox, list) and inbox
    assert "already read this file successfully" in str(inbox[-1].get("content") or "")
    pending = run.vars["_temp"].get("pending_tool_calls")
    assert isinstance(pending, list) and len(pending) == 4


def test_staircase_hint_fires_once_per_path_not_once_per_range() -> None:
    """The escape hatch must not require a byte-identical retry.

    The signature used to be `path:start:end`, so every adjustment the model
    made — including the widening this hint's own message asks for — looked
    like a new pattern and re-fired the guard. An obedient model could be
    nudged until its iteration budget was gone.
    """
    previous = _previous_read_batch(1505, 1520)
    first = detect_nearby_same_file_staircase(
        previous,
        [{"name": "read_file", "arguments": {"path": "app.py", "start_line": 1495, "end_line": 1515}}],
        previously_warned_signature="",
    )
    assert first is not None
    signature = str(first["signature"])
    assert first["enforcement"] == "advise"

    # The model does exactly what the message asked: ONE wider contiguous range.
    widened = detect_nearby_same_file_staircase(
        previous,
        [{"name": "read_file", "arguments": {"path": "app.py", "start_line": 1480, "end_line": 1560}}],
        previously_warned_signature=signature,
    )
    assert widened is None, "obeying the hint must not re-trigger the hint"

    # A different file is still worth one piece of advice of its own.
    other = detect_nearby_same_file_staircase(
        {
            "count": 1,
            "same_path": True,
            "only_slices": True,
            "items": [{"path": "other.py", "start": 10, "end": 20, "is_slice": True}],
        },
        [{"name": "read_file", "arguments": {"path": "other.py", "start_line": 25, "end_line": 40}}],
        previously_warned_signature=signature,
    )
    assert other is not None and other["signature"] != signature


def test_production_run_9ea71c55_read_sequence_no_longer_stalls() -> None:
    """Replay of the real failure, cycles 14-18 of run 9ea71c55.

    Recorded behaviour before the fix (scratchpad artifact 0029ce699b6a):
        1505-1520 -> executed
        1495-1515 -> observations: []   (iteration spent, no data)
        1480-1515 -> observations: []   (iteration spent, no data)
        1480-1520 -> observations: []   (iteration spent, no data)
        1480-1520 -> executed           (identical repeat finally allowed)

    The run then died: "stopped: iteration budget after 20 iterations".
    """
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_READ]), workflow_id="wf", provider="stub", model="stub", allowed_tools=["read_file"]
    )
    sequence = [(1495, 1515), (1480, 1515), (1480, 1520), (1480, 1520)]
    executed = 0
    hints = 0

    for start, end in sequence:
        run = _react_run(start=start, end=end, previous=_previous_read_batch(1505, 1520))
        # Carry the warned signature forward, as the scratchpad does across cycles.
        run.vars["scratchpad"]["read_orchestration_last_hint"] = (
            "app.py:staircase" if hints else ""
        )
        plan = wf.get_node("parse")(run, _Ctx())
        assert plan.next_node == "act", f"read {start}-{end} must reach the tools"
        pending = run.vars["_temp"].get("pending_tool_calls")
        assert isinstance(pending, list) and pending
        executed += 1
        if run.vars["_runtime"]["inbox"]:
            hints += 1

    assert executed == len(sequence), "every read in the sequence must execute"
    assert hints == 1, f"the model should be advised once, not {hints} times"


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
