from typing import Any

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import EffectType, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


@pytest.mark.parametrize("recover", [False, True])
def test_format_repair_is_bounded_and_never_executes_rejected_calls(recover):
    requests: list[dict[str, Any]] = []
    executed = []

    def llm(run, effect, next_node):
        requests.append(effect.payload)
        if len(requests) > 1:
            assert "nothing was executed" in str(effect.payload["messages"])
            # The repair names no text format of its own (a native tool-calling
            # provider must never be taught a text syntax): it points at the
            # request's own instructions and lists the roster.
            assert "<tool_call>" not in str(effect.payload["messages"])
            assert "format and argument schemas supplied in the instructions" in str(effect.payload["messages"])
            assert "Use only the available tools: web_search." in str(effect.payload["messages"])
        if recover and len(requests) == 2:
            return EffectOutcome.completed({"content": "", "tool_calls": [{"name": "web_search", "arguments": {"query": "test"}, "call_id": "one"}]})
        if recover and len(requests) == 3:
            return EffectOutcome.completed({"content": "Found it."})
        return EffectOutcome.completed({"content": "Broken tool block", "metadata": {
            "tool_call_error": {"code": "invalid_tool_syntax", "available_tools": ["web_search"]}}})

    def tool(run, effect, next_node):
        executed.extend(effect.payload["tool_calls"])
        return EffectOutcome.completed({"mode": "executed", "results": [{"name": "web_search", "call_id": "one", "success": True, "output": "found"}]})

    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
                      effect_handlers={EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: tool})
    workflow = create_react_workflow(logic=ReActLogic(tools=[ToolDefinition(name="web_search", description="Search", parameters={"query": {"type": "string"}})]),
                                    workflow_id="format-repair", provider="stub", model="stub", allowed_tools=["web_search"])
    rid = runtime.start(workflow=workflow, vars={"context": {"task": "Search", "messages": []}, "_runtime": {"inbox": []}})
    def drive():
        for _ in range(80):
            state = runtime.tick(workflow=workflow, run_id=rid, max_steps=1)
            if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                return state
    if recover:
        assert drive().status == RunStatus.COMPLETED
    else:
        # GatewayRunner persists planning exceptions as FAILED (tick_exception).
        with pytest.raises(RuntimeError, match="format repair failed"):
            drive()
    assert len(requests) == 3
    assert len(executed) == (1 if recover else 0)


def test_format_repair_budget_is_per_turn_in_a_composed_conversation():
    """Two turns of one run (visit composition): each turn has two rejected formats and
    still recovers. Without `reset_react_turn` clearing `tool_format_retries`, turn 2's
    first rejection would raise at once (the budget was per conversation)."""
    from abstractagent.adapters.react_runtime import reset_react_turn
    from abstractruntime.core.models import StepPlan
    from abstractruntime.core.spec import WorkflowSpec

    bad = {"content": "Broken tool block", "metadata": {"tool_call_error": {"code": "invalid_tool_syntax", "available_tools": ["web_search"]}}}
    script = [bad, bad, {"content": "Answer one."}, bad, bad, {"content": "Answer two."}]
    calls = {"n": 0}
    events: list[str] = []
    answers: list[str] = []

    def llm(run, effect, next_node):
        i = calls["n"]
        calls["n"] += 1
        return EffectOutcome.completed(dict(script[min(i, len(script) - 1)]))

    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="web_search", description="Search", parameters={"query": {"type": "string"}})]),
        workflow_id="format-repair-turns", provider="stub", model="stub", allowed_tools=["web_search"],
        final_next_node="park", on_step=lambda step, data: events.append(step),
    )

    def park(run, ctx):
        temp = run.vars.get("_temp") or {}
        answers.append(str(temp.get("final_answer") or ""))
        if len(answers) >= 2:
            return StepPlan(node_id="park", complete_output={"turns": len(answers)})
        reset_react_turn(run.vars)
        run.vars["context"]["messages"].append({"role": "user", "content": "Second message"})
        return StepPlan(node_id="park", next_node="reason")

    nodes = dict(react.nodes)
    nodes["park"] = park
    workflow = WorkflowSpec(workflow_id=react.workflow_id, entry_node=react.entry_node, nodes=nodes)
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), effect_handlers={EffectType.LLM_CALL: llm})
    rid = runtime.start(workflow=workflow, vars={"context": {"task": "First message", "messages": []}, "_runtime": {"inbox": []}})
    state = None
    for _ in range(120):
        state = runtime.tick(workflow=workflow, run_id=rid, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert state is not None and state.status == RunStatus.COMPLETED, state
    assert answers == ["Answer one.", "Answer two."]
    assert events.count("parse_retry_tool_format") == 4
    assert calls["n"] == 6
