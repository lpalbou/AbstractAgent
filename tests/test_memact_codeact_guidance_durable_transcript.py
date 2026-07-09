"""Injected guidance must persist in the MemAct/CodeAct transcripts (maintainer ruling
2026-07-09, extended 2026-07-10 — parity with the ReAct adapter).

Before this ruling both adapters folded drained `_runtime.inbox` guidance into the SYSTEM
PROMPT of a single LLM call: it mutated the cached prefix (cache-busting) AND was
forgotten on the next cycle, so a final answer written later re-anchored on the original
task and dropped the operator's correction.

Per adapter, drives the real `reason` node handler and asserts:
- drained guidance lands in the durable transcript as a user message
  (metadata.kind=operator_guidance) and the inbox is emptied;
- the guidance rides the LLM payload's message lane (not the system prompt) on the drain
  cycle AND on later cycles (persistence — the pre-ruling behavior lost it here);
- the payload never contains consecutive user turns (alternation-strict templates 400
  on user,user; the sanitizer merges at the payload boundary only).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus

pytestmark = pytest.mark.basic

GUIDANCE = "Correction: also report the disk usage; the final answer must mention DISK USAGE."


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2026-07-10T06:00:00+00:00"


def _base_vars() -> Dict[str, Any]:
    return {
        "context": {"task": "Summarize the workspace", "messages": [{"role": "user", "content": "Summarize the workspace"}]},
        "scratchpad": {"iteration": 0, "max_iterations": 5},
        "_runtime": {
            "active_memory": {"version": 1, "persona": "p"},
            "inbox": [{"role": "system", "content": GUIDANCE}],
        },
        "_temp": {},
        "_limits": {"max_history_messages": -1, "max_tokens": 32768, "max_iterations": 5, "current_iteration": 0},
    }


def _payload_messages(plan) -> List[Dict[str, Any]]:
    assert plan.effect is not None
    payload = dict(plan.effect.payload or {})
    msgs = payload.get("messages")
    assert isinstance(msgs, list) and msgs
    return msgs


def _assert_guidance_semantics(workflow, run: RunState) -> None:
    handler = workflow.get_node("reason")

    # Cycle 1: drain. Guidance must move inbox -> durable transcript -> payload message lane.
    plan1 = handler(run, _Ctx())
    payload1 = dict(plan1.effect.payload or {})
    msgs1 = _payload_messages(plan1)

    hits1 = [m for m in msgs1 if GUIDANCE in str(m.get("content") or "")]
    assert hits1, "guidance missing from the drain-cycle LLM payload"
    assert all(m.get("role") == "user" for m in hits1)
    assert any("[Operator guidance" in str(m.get("content") or "") for m in hits1)

    # The old one-shot system-prompt rendering is gone.
    assert "Guidance:" not in str(payload1.get("system_prompt") or "")

    # No consecutive user turns in the payload (strict-template safety).
    roles1 = [m.get("role") for m in msgs1]
    assert all(not (a == b == "user") for a, b in zip(roles1, roles1[1:])), f"user,user adjacency: {roles1}"

    # Durable transcript carries exactly one operator_guidance user message; inbox drained.
    durable = [
        m
        for m in run.vars["context"]["messages"]
        if isinstance(m, dict) and (m.get("metadata") or {}).get("kind") == "operator_guidance"
    ]
    assert len(durable) == 1
    assert durable[0].get("role") == "user"
    assert GUIDANCE in str(durable[0].get("content") or "")
    assert run.vars["_runtime"]["inbox"] == []

    # Cycle 2 (inbox now empty): guidance must STILL ride the payload — the pre-ruling
    # ephemeral rendering lost it exactly here.
    run.vars["_temp"] = {}
    plan2 = handler(run, _Ctx())
    msgs2 = _payload_messages(plan2)
    assert any(GUIDANCE in str(m.get("content") or "") for m in msgs2)
    assert "Guidance:" not in str(dict(plan2.effect.payload or {}).get("system_prompt") or "")

    # Still exactly one durable copy (no re-append without a new injection).
    durable2 = [
        m
        for m in run.vars["context"]["messages"]
        if isinstance(m, dict) and (m.get("metadata") or {}).get("kind") == "operator_guidance"
    ]
    assert len(durable2) == 1


def test_memact_guidance_is_durable_and_persists_across_cycles() -> None:
    logic = MemActLogic(tools=[ToolDefinition(name="list_files", description="list", parameters={})])
    workflow = create_memact_workflow(logic=logic, on_step=None)
    run = RunState(
        run_id="r-memact-guidance",
        workflow_id="memact_agent",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars=_base_vars(),
    )
    _assert_guidance_semantics(workflow, run)


def test_codeact_guidance_is_durable_and_persists_across_cycles() -> None:
    logic = CodeActLogic(tools=[ToolDefinition(name="list_files", description="list", parameters={})])
    workflow = create_codeact_workflow(logic=logic, on_step=None)
    run = RunState(
        run_id="r-codeact-guidance",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="reason",
        vars=_base_vars(),
    )
    _assert_guidance_semantics(workflow, run)
