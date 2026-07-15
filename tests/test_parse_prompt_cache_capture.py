"""0030 residue (gate lifted 2026-07-15): prompt-cache telemetry on `parse`.

Core's local-cache providers record per-call cache telemetry into
`GenerateResponse.metadata["prompt_cache"]` (outcome / cached_tokens /
fed_tokens / #FALLBACK degraded_reason) and the runtime's llm_client folds
`metadata` into every LLM result dict. The loops surface that struct as an
ADDITIVE `prompt_cache` key on the `parse` emit payload — present exactly
when the provider reported one, absent otherwise (remote providers and older
stacks). Consumers keep keying on the 0028 common core; this is observability
riding an existing event, not a new step name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.loop_hooks import HookEvent, LoopHooks
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic

_CACHE_STRUCT = {
    "mode": "key",
    "key": "sess-cache-1",
    "outcome": "hit_restore",
    "cached_tokens": 512,
    "fed_tokens": 9,
}


def _factory(loop: str, hooks: LoopHooks):
    if loop == "react":
        return create_react_workflow(logic=ReActLogic(tools=[]), hooks=hooks, provider="stub", model="stub")
    if loop == "codeact":
        return create_codeact_workflow(logic=CodeActLogic(tools=[]), hooks=hooks)
    return create_memact_workflow(logic=MemActLogic(tools=[]), hooks=hooks)


def _drive(loop: str, response: Dict[str, Any]) -> List[HookEvent]:
    events: List[HookEvent] = []
    hooks = LoopHooks().add(events.append)

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, effect, default_next_node
        return EffectOutcome.completed(dict(response))

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
    )
    workflow = _factory(loop, hooks)
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id=f"sess-cache-{loop}",
    )
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    return events


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_parse_surfaces_provider_prompt_cache_telemetry(loop: str) -> None:
    reply = {
        "content": "Done.",
        "tool_calls": [],
        "finish_reason": "stop",
        "metadata": {"prompt_cache": dict(_CACHE_STRUCT)},
    }
    events = _drive(loop, reply)
    parses = [e for e in events if e.step == "parse"]
    assert parses, f"{loop}: parse must fire"
    assert parses[0].data.get("prompt_cache") == _CACHE_STRUCT
    # Common core untouched (0028): the addition rides beside it.
    assert parses[0].data.get("has_tool_calls") is False
    assert "content_preview" in parses[0].data


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_parse_omits_prompt_cache_when_provider_reported_none(loop: str) -> None:
    reply = {"content": "Done.", "tool_calls": [], "finish_reason": "stop"}
    events = _drive(loop, reply)
    parses = [e for e in events if e.step == "parse"]
    assert parses, f"{loop}: parse must fire"
    assert "prompt_cache" not in parses[0].data, (
        f"{loop}: absent telemetry must stay absent — never an empty/placeholder struct"
    )
