"""Delegated sub-agents inherit the run's live token-streaming switch.

`_runtime.stream: true` asks the runtime to stream every LLM call of the run
to the host's live view. A `delegate_agent` child is started with an explicit
`_runtime` rider list; without `stream` in it, the child's calls would silently
stop streaming mid-tree. False (explicit off) is inherited too; an absent value
stays absent.
"""
from types import SimpleNamespace

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractruntime import EffectType, RunState, RunStatus


@pytest.mark.parametrize(
    "factory,logic",
    [
        (create_react_workflow, ReActLogic),
        (create_codeact_workflow, CodeActLogic),
        (create_memact_workflow, MemActLogic),
    ],
)
@pytest.mark.parametrize("stream", [True, False, None])
def test_every_delegate_loop_inherits_stream(factory, logic, stream):
    workflow = factory(logic=logic(tools=[DELEGATE_AGENT_TOOL]), workflow_id="wf")
    runtime_ns = {"inbox": [], "provider": "stub", "model": "stub", "allowed_tools": ["delegate_agent"]}
    if stream is not None:
        runtime_ns["stream"] = stream
    run = RunState(
        run_id="parent",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="act",
        vars={
            "context": {"task": "test", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": 20, "cycles": []},
            "_runtime": runtime_ns,
            "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": {"task": "subtask"}, "call_id": "d1"}]},
            "_limits": {"max_iterations": 20},
        },
    )
    plan = workflow.get_node("act")(run, SimpleNamespace(now_iso=lambda: "2026-09-26T00:00:00Z"))
    assert plan.effect is not None and plan.effect.type == EffectType.START_SUBWORKFLOW
    child = plan.effect.payload["vars"]["_runtime"]
    if stream is None:
        assert "stream" not in child
    else:
        assert child["stream"] is stream
