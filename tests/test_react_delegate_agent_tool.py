from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars(*, task: str) -> Dict[str, Any]:
    return {"context": {"task": task, "messages": []}, "_runtime": {"inbox": []}}


@pytest.mark.basic
def test_react_delegate_agent_runs_subworkflow_and_returns_tool_observation() -> None:
    """delegate_agent should run a fresh sub-agent run and return its answer as a tool observation."""

    call_count: dict[str, int] = {}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        call_count[run.run_id] = int(call_count.get(run.run_id, 0) or 0) + 1

        is_child = bool(getattr(run, "parent_run_id", None))
        if is_child:
            return EffectOutcome.completed({"content": "Child answer", "tool_calls": [], "finish_reason": "stop"})

        idx = call_count[run.run_id]
        if idx == 1:
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [
                        {
                            "name": "delegate_agent",
                            "arguments": {"task": "Find X", "context": "Only look at file A", "tools": ["read_file", "search_files"]},
                            "call_id": "call_1",
                        }
                    ],
                    "finish_reason": "tool_calls",
                }
            )

        return EffectOutcome.completed({"content": "Done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )

    # Minimal tool defs: include delegate_agent + a couple of read-only tools so allowlist normalization works.
    tool_defs = [
        DELEGATE_AGENT_TOOL,
        ToolDefinition(name="read_file", description="Read", parameters={}),
        ToolDefinition(name="search_files", description="Search", parameters={}),
    ]

    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)

    run_id = runtime.start(workflow=workflow, vars=_base_vars(task="Parent task"), actor_id=None, session_id=None)

    for _ in range(200):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED

    output = final.output if isinstance(final.output, dict) else {}
    msgs = output.get("messages") if isinstance(output, dict) else None
    assert isinstance(msgs, list) and msgs
    tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("delegate_agent" in str(m.get("content") or "") and "Child answer" in str(m.get("content") or "") for m in tool_msgs), msgs

