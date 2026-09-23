"""All Agent loops and delegated children preserve the same Core controls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from abstractagent.adapters.generation_params import runtime_llm_params
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.logic.react import ReActLogic
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractruntime import RunState, RunStatus, EffectType

D2 = {"mode": "native_mtp", "num_draft_tokens": 2}
D4 = {"mode": "native_mtp", "num_draft_tokens": 4}


@pytest.mark.parametrize("parent,override,expected", [(D2, None, D2), (D2, False, False), (False, D4, D4), (False, None, False), (D2, {}, {}), (None, None, None)])
def test_shared_agent_params_preserve_off_and_inheritance(parent, override, expected):
    ns, extra = {"speculation": deepcopy(parent)}, {"speculation": deepcopy(override)}
    result = runtime_llm_params(ns, extra=extra)
    assert result.get("speculation") == expected
    if expected is None:
        assert "speculation" not in result
    if isinstance(result.get("speculation"), dict):
        result["speculation"]["num_draft_tokens"] = 9
    assert ns["speculation"] == parent and extra["speculation"] == override


@pytest.mark.parametrize("factory,logic", [(create_react_workflow, ReActLogic), (create_codeact_workflow, CodeActLogic), (create_memact_workflow, MemActLogic)])
@pytest.mark.parametrize("policy", [False, D2])
@pytest.mark.parametrize("profile_override", [None, False, D4])
def test_every_delegate_loop_inherits_and_accepts_host_profile_override(factory, logic, policy, profile_override):
    workflow = factory(logic=logic(tools=[DELEGATE_AGENT_TOOL]), workflow_id="wf")
    runtime_ns = {"inbox": [], "provider": "stub", "model": "stub", "allowed_tools": ["delegate_agent"], "speculation": deepcopy(policy)}
    args = {"task": "subtask"}
    if profile_override is not None:
        runtime_ns["delegate_substrates"] = {"test": {"provider": "stub", "model": "other", "speculation": deepcopy(profile_override)}}
        args["substrate"] = "test"
    run = RunState(run_id="parent", workflow_id="wf", status=RunStatus.RUNNING, current_node="act", vars={
        "context": {"task": "test", "messages": []},
        "scratchpad": {"iteration": 1, "max_iterations": 20, "cycles": []},
        "_runtime": runtime_ns,
        "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": args, "call_id": "d1"}]},
        "_limits": {"max_iterations": 20},
    })
    plan = workflow.get_node("act")(run, SimpleNamespace(now_iso=lambda: "2026-09-20T00:00:00Z"))
    assert plan.effect is not None and plan.effect.type == EffectType.START_SUBWORKFLOW
    child = plan.effect.payload["vars"]["_runtime"]
    expected = profile_override if profile_override is not None else policy
    assert child["speculation"] == expected
    if isinstance(child["speculation"], dict):
        child["speculation"]["num_draft_tokens"] = 9
    assert runtime_ns["speculation"] == policy
    if profile_override is not None:
        assert runtime_ns["delegate_substrates"]["test"]["speculation"] == profile_override
