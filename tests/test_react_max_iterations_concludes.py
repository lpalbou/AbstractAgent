from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _vars(*, task: str, max_iterations: int) -> Dict[str, Any]:
    return {
        "context": {"task": task, "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": max_iterations},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }


@pytest.mark.basic
def test_react_max_iterations_triggers_tool_free_conclusion_call() -> None:
    llm_payloads: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(dict(payload))

        # Normal ReAct steps include tool schemas in the payload; the max-iteration conclusion call must not.
        if "tools" in payload:
            idx = len([p for p in llm_payloads if "tools" in p])
            return EffectOutcome.completed(
                {
                    "content": "",
                    "tool_calls": [
                        {"name": "list_files", "arguments": {"directory_path": "."}, "call_id": f"call_{idx}"}
                    ],
                    "finish_reason": "tool_calls",
                }
            )

        sys = str(payload.get("system_prompt") or "")
        assert "Max iterations reached" in sys or "max iterations" in sys.lower()
        assert "Do NOT call tools" in sys or "do not call tools" in sys.lower()
        return EffectOutcome.completed({"content": "FINAL REPORT", "tool_calls": [], "finish_reason": "stop"})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_calls = payload.get("tool_calls")
        assert isinstance(tool_calls, list)
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in tool_calls
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )

    workflow = create_react_workflow(
        logic=ReActLogic(
            tools=[
                ToolDefinition(name="list_files", description="List", parameters={}),
            ]
        ),
        workflow_id="react_agent_max_iterations_conclusion",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )

    run_id = runtime.start(workflow=workflow, vars=_vars(task="t", max_iterations=2), actor_id=None, session_id=None)

    for _ in range(100):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    assert state.current_node == "max_iterations"
    assert isinstance(state.output, dict)
    assert state.output.get("answer") == "FINAL REPORT"

    messages = state.output.get("messages")
    assert isinstance(messages, list)
    assert any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") == "FINAL REPORT" for m in messages)

    assert len(llm_payloads) >= 3
    assert any("tools" not in p for p in llm_payloads), "expected a tool-free max-iterations conclusion call"

