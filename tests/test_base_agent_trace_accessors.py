from __future__ import annotations

from abstractagent.agents.base import BaseAgent
from abstractruntime import (
    InMemoryLedgerStore,
    InMemoryRunStore,
    RunState,
    RunStatus,
    Runtime,
    WorkflowSpec,
)
from abstractruntime.core.models import Effect, EffectType, StepPlan


class _TestAgent(BaseAgent):
    def _create_workflow(self) -> WorkflowSpec:
        def answer_node(run: RunState, ctx: object) -> StepPlan:
            del run, ctx
            return StepPlan(
                node_id="answer",
                effect=Effect(EffectType.ANSWER_USER, payload={"message": "hi"}),
                next_node=None,
            )

        return WorkflowSpec(
            workflow_id="wf_test_base_agent_trace_accessors",
            entry_node="answer",
            nodes={"answer": answer_node},
        )

    def start(self, task: str) -> str:
        run_id = self.runtime.start(
            workflow=self.workflow,
            vars={"context": {"task": task}, "scratchpad": {"foo": "bar"}},
            actor_id=self._ensure_actor_id(),
            session_id=self._ensure_session_id(),
        )
        self._current_run_id = run_id
        return run_id

    def step(self) -> RunState:
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start() first.")
        return self.runtime.tick(workflow=self.workflow, run_id=self._current_run_id)


def test_base_agent_trace_accessors() -> None:
    runtime = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    agent = _TestAgent(runtime=runtime, tools=[])

    assert agent.get_context() == {}
    assert agent.get_scratchpad() == {}
    assert agent.get_node_traces() == {}
    assert agent.get_node_trace("answer") == {"node_id": "answer", "steps": []}

    agent.start("t")
    state = agent.step()
    assert state.status == RunStatus.COMPLETED

    assert agent.get_context().get("task") == "t"
    assert agent.get_scratchpad().get("foo") == "bar"

    traces = agent.get_node_traces()
    assert "answer" in traces

    trace1 = agent.get_node_trace("answer")
    assert trace1.get("node_id") == "answer"
    assert isinstance(trace1.get("steps"), list)
    assert trace1["steps"]
    assert trace1["steps"][-1]["effect"]["type"] == "answer_user"

    original_len = len(trace1["steps"])
    trace1["steps"].append({"mutated": True})
    trace2 = agent.get_node_trace("answer")
    assert len(trace2.get("steps") or []) == original_len

