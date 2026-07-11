"""Prefix-reuse measurement (A/B criterion 3 evidence tooling).

Pins `abstractagent.metrics.prefix_reuse` two ways:

1. UNIT: synthetic payload sequences with known reuse shapes (identical heads,
   growing message lanes, a head mutation invalidating everything).
2. INTEGRATION: measured over the REAL adapter loop's captured payloads — the
   same scripted 3-iteration run the prefix-stability suite drives — asserting
   the measured numbers agree with the byte-discipline those tests pin
   (heads identical; the reusable prefix grows monotonically; reuse ratio is
   high and rising by iteration 3).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractagent.metrics import measure_prefix_reuse, prefix_reuse_report
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


def test_unit_reuse_shapes() -> None:
    head = {"system_prompt": "You are X.", "tools": [{"name": "t"}]}
    m1 = {"role": "user", "content": "task"}
    m2 = {"role": "assistant", "content": "thought", "tool_calls": [{"id": "c1"}]}
    m3 = {"role": "tool", "content": "obs", "tool_call_id": "c1"}

    payloads: List[Dict[str, Any]] = [
        {**head, "messages": [m1]},
        {**head, "messages": [m1, m2, m3]},
        {**head, "messages": [m1, m2, m3, {"role": "user", "content": "next"}]},
    ]
    per = measure_prefix_reuse(payloads)
    assert per[0]["reused_bytes"] == 0
    assert per[1]["head_identical"] is True
    assert per[1]["reused_message_count"] == 1  # m1 identical, then growth
    assert per[2]["reused_message_count"] == 3
    assert per[2]["reuse_ratio"] > per[1]["reuse_ratio"] > 0

    report = prefix_reuse_report(payloads)
    assert report["requests"] == 3
    assert report["heads_identical"] is True
    assert 0 < report["overall_reuse_ratio"] < 1

    # A mutated head invalidates the whole prefix (provider caches key on the
    # serialized request prefix; nothing after a changed head can hit).
    mutated = [
        {**head, "messages": [m1, m2, m3]},
        {"system_prompt": "You are X. Iteration 2.", "tools": head["tools"], "messages": [m1, m2, m3]},
    ]
    per_mut = measure_prefix_reuse(mutated)
    assert per_mut[1]["head_identical"] is False
    assert per_mut[1]["reused_bytes"] == 0
    assert per_mut[1]["reuse_ratio"] == 0.0


def test_measured_over_real_adapter_loop() -> None:
    captured: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration <= 2:
            return EffectOutcome.completed(
                {
                    "content": f"cycle {iteration}",
                    "tool_calls": [
                        {"name": "list_files", "arguments": {"directory_path": f"d{iteration}"}, "call_id": f"call_{iteration}"}
                    ],
                }
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    workflow = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="react_prefix_metric",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Measure me", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-metric",
    )
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    assert len(captured) == 3

    report = prefix_reuse_report(captured)
    per = report["per_request"]

    # Agrees with the pinned byte-discipline: heads identical across the run.
    assert report["heads_identical"] is True
    # Known trade, MEASURED (adjacency guard, pinned in the prefix-stability suite):
    # iteration 1's task message merges the volatile tail, so iteration 2 reuses the
    # HEAD but zero messages — the measurement sees exactly what the cache would.
    assert per[1]["head_identical"] is True
    assert per[1]["reused_message_count"] == 0
    assert per[1]["reused_bytes"] == per[1]["head_bytes"] > 0
    # From iteration 3 the transcript prefix is reusable and growing; the ratio is high.
    assert per[2]["reused_message_count"] > 0
    assert per[2]["reused_bytes"] > per[1]["reused_bytes"]
    assert per[2]["reuse_ratio"] > 0.5
    assert 0 < report["overall_reuse_ratio"] < 1
