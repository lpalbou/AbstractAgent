"""Reasoning-first-citizen pins for the agent section (plan, commons c5789).

Two deliverables from the agent seat's claimed section:

1. Parse parity — the separated reasoning channel (`response.reasoning` /
   `reasoning_content`) was surfaced in the parse step event by ReAct only.
   All three loops now share one reader (`extract_reasoning_text`) and emit
   the same additive `reasoning` field on the common-core parse payload.

2. Substrate palette gains the reasoning member — a host-granted delegate
   substrate profile may carry `thinking` next to provider/model, because a
   substrate pick is an AI-selection and the selection is a triple. Rules:
   declared-and-valid wins over the inherited parent value; absent means
   inherit (a sub-agent is not a different mind unless the host says so);
   present-but-invalid warns loudly and keeps inheritance (deny-safe, never
   a failed delegation).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.adapters.transcripts import extract_reasoning_text
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


# ---------------------------------------------------------------------------
# 1. Shared reader contract
# ---------------------------------------------------------------------------


def test_extract_reasoning_text_contract() -> None:
    assert extract_reasoning_text({"reasoning": "chain of thought"}) == "chain of thought"
    # OpenAI-compatible servers report `reasoning_content`.
    assert extract_reasoning_text({"reasoning_content": "alt channel"}) == "alt channel"
    # Core-normalized key wins when both are present.
    assert extract_reasoning_text({"reasoning": "primary", "reasoning_content": "secondary"}) == "primary"
    # Absent / null / non-dict are honest empties, never placeholders.
    assert extract_reasoning_text({}) == ""
    assert extract_reasoning_text({"reasoning": None, "reasoning_content": None}) == ""
    assert extract_reasoning_text(None) == ""
    assert extract_reasoning_text("not a dict") == ""


# ---------------------------------------------------------------------------
# 2. Parse parity across the three loops
# ---------------------------------------------------------------------------


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2026-07-27T02:00:00+00:00"


def _mk_run(vars_: Dict[str, Any]) -> RunState:
    return RunState(
        run_id="run-parse-parity",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node="parse",
        vars=vars_,
    )


def _loop_vars(response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
        "scratchpad": {"iteration": 0, "max_iterations": 5, "cycles": []},
        "_runtime": {"inbox": [], "active_memory": {"version": 1}},
        "_temp": {"llm_response": dict(response)},
        "_limits": {"max_iterations": 5, "current_iteration": 0},
    }


def _parse_events(factory, logic, response: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def on_step(step: str, data: Dict[str, Any]) -> None:
        if step == "parse":
            events.append(dict(data))

    workflow = factory(logic=logic, on_step=on_step)
    handler = workflow.get_node("parse")
    handler(_mk_run(_loop_vars(response)), _Ctx())
    assert events, "parse node emitted no parse event"
    return events


@pytest.mark.parametrize(
    "factory,logic_cls",
    [
        (create_react_workflow, ReActLogic),
        (create_codeact_workflow, CodeActLogic),
        (create_memact_workflow, MemActLogic),
    ],
    ids=["react", "codeact", "memact"],
)
def test_parse_event_carries_reasoning_in_all_three_loops(factory, logic_cls) -> None:
    response = {
        "content": "FINAL: done",
        "tool_calls": [],
        "finish_reason": "stop",
        "reasoning": "I checked the file first.",
    }
    events = _parse_events(factory, logic_cls(tools=[]), response)
    assert events[0].get("reasoning") == "I checked the file first."


@pytest.mark.parametrize(
    "factory,logic_cls",
    [
        (create_react_workflow, ReActLogic),
        (create_codeact_workflow, CodeActLogic),
        (create_memact_workflow, MemActLogic),
    ],
    ids=["react", "codeact", "memact"],
)
def test_parse_event_reasoning_is_empty_when_absent(factory, logic_cls) -> None:
    response = {"content": "FINAL: done", "tool_calls": [], "finish_reason": "stop"}
    events = _parse_events(factory, logic_cls(tools=[]), response)
    assert events[0].get("reasoning") == ""


# ---------------------------------------------------------------------------
# 3. Substrate palette: the reasoning member
# ---------------------------------------------------------------------------


def _drive_delegation(
    *,
    parent_runtime_vars: Dict[str, Any],
    delegate_args: Dict[str, Any],
) -> Tuple[Runtime, str, List[Tuple[str, Dict[str, Any]]]]:
    """Parent delegates once; child answers immediately. Returns the runtime,
    the parent run id, and every captured step event (name, payload)."""
    events: List[Tuple[str, Dict[str, Any]]] = []

    def on_step(step: str, data: Dict[str, Any]) -> None:
        events.append((step, dict(data)))

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        if getattr(run, "parent_run_id", None):
            return EffectOutcome.completed({"content": "FINAL: child done", "tool_calls": [], "finish_reason": "stop"})
        if not getattr(llm_handler, "_delegated", False):
            llm_handler._delegated = True  # type: ignore[attr-defined]
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [{"name": "delegate_agent", "arguments": dict(delegate_args), "call_id": "d1"}],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "FINAL: parent done", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )
    workflow = create_react_workflow(logic=ReActLogic(tools=[DELEGATE_AGENT_TOOL]), workflow_id="react_agent", on_step=on_step)
    runtime.workflow_registry.register(workflow)

    vars_: Dict[str, Any] = {
        "context": {"task": "Delegate the task.", "messages": []},
        "_runtime": {"inbox": [], **copy.deepcopy(parent_runtime_vars)},
    }
    run_id = runtime.start(workflow=workflow, vars=vars_, actor_id=None, session_id=None)
    for _ in range(300):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    return runtime, run_id, events


def _child_runtime_ns(runtime: Runtime, parent_id: str) -> Dict[str, Any]:
    children = runtime.run_store.list_children(parent_run_id=parent_id)
    assert children, "no child run found"
    return dict(children[0].vars.get("_runtime") or {})


def _events_named(events: List[Tuple[str, Dict[str, Any]]], name: str) -> List[Dict[str, Any]]:
    return [d for s, d in events if s == name]


PALETTE_BASE = {"provider": "lmstudio", "model": "qwen3.5-4b"}


def test_substrate_profile_thinking_pins_the_child() -> None:
    """A profile carrying a valid thinking level pins the child's reasoning
    effort — it wins over the inherited parent value, and the parent's own
    value is never touched."""
    runtime, parent_id, events = _drive_delegation(
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "thinking": "high",
            "delegate_substrates": {"drafting": {**PALETTE_BASE, "thinking": "low"}},
        },
        delegate_args={"task": "Draft it.", "substrate": "drafting"},
    )
    child_ns = _child_runtime_ns(runtime, parent_id)
    assert child_ns.get("thinking") == "low"
    assert child_ns.get("provider") == "lmstudio"
    # Parent untouched.
    assert runtime.get_state(parent_id).vars["_runtime"].get("thinking") == "high"
    # The substrate event names the applied thinking; no skew warning fired.
    applied = _events_named(events, "delegate_agent_substrate")
    assert applied and applied[0].get("thinking") == "low"
    assert not _events_named(events, "delegate_agent_substrate_skew")


def test_substrate_profile_without_thinking_inherits_parent() -> None:
    """No thinking in the profile: the child keeps the parent's effective
    value — a substrate switch alone must not silently reset reasoning."""
    runtime, parent_id, events = _drive_delegation(
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "thinking": "medium",
            "delegate_substrates": {"drafting": dict(PALETTE_BASE)},
        },
        delegate_args={"task": "Draft it.", "substrate": "drafting"},
    )
    child_ns = _child_runtime_ns(runtime, parent_id)
    assert child_ns.get("thinking") == "medium"
    applied = _events_named(events, "delegate_agent_substrate")
    assert applied and "thinking" not in applied[0]
    assert not _events_named(events, "delegate_agent_substrate_skew")


def test_substrate_profile_invalid_thinking_warns_and_keeps_inheritance() -> None:
    """A host typo in the grant ('verbose' is not a level) must not fail the
    delegation and must not silently apply: loud skew warning, child keeps
    the inherited value."""
    runtime, parent_id, events = _drive_delegation(
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "thinking": "medium",
            "delegate_substrates": {"drafting": {**PALETTE_BASE, "thinking": "verbose"}},
        },
        delegate_args={"task": "Draft it.", "substrate": "drafting"},
    )
    child_ns = _child_runtime_ns(runtime, parent_id)
    assert child_ns.get("thinking") == "medium"
    applied = _events_named(events, "delegate_agent_substrate")
    assert applied and "thinking" not in applied[0]
    skews = _events_named(events, "delegate_agent_substrate_skew")
    assert skews and skews[0].get("ignored_keys") == ["thinking"]
    assert "#FALLBACK" in str(skews[0].get("warning") or "")


def test_substrate_profile_thinking_is_a_known_key_not_skew() -> None:
    """`thinking` is a member of the palette contract now — it must never be
    reported as an unrecognized profile key (version-skew warning)."""
    _runtime, _parent_id, events = _drive_delegation(
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "delegate_substrates": {"drafting": {**PALETTE_BASE, "thinking": "xhigh"}},
        },
        delegate_args={"task": "Draft it.", "substrate": "drafting"},
    )
    for skew in _events_named(events, "delegate_agent_substrate_skew"):
        assert "thinking" not in (skew.get("ignored_keys") or [])


# ---------------------------------------------------------------------------
# 4. CodeAct / MemAct: the same thinking profile must not crash (NameError)
# ---------------------------------------------------------------------------
# Until 2026-09-26 both adapters called `normalize_thinking` in their
# delegate_agent substrate branch without importing it, so a host-granted
# profile carrying `thinking` raised NameError and failed the parent run.


@pytest.mark.parametrize(
    "factory,logic_cls,workflow_id",
    [
        (create_codeact_workflow, CodeActLogic, "codeact_agent"),
        (create_memact_workflow, MemActLogic, "memact_agent"),
    ],
    ids=["codeact", "memact"],
)
def test_sibling_substrate_profile_thinking_pins_the_child_without_crashing(factory, logic_cls, workflow_id) -> None:
    events: List[Tuple[str, Dict[str, Any]]] = []
    delegated = {"done": False}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        if getattr(run, "parent_run_id", None):
            return EffectOutcome.completed({"content": "FINAL: child done", "tool_calls": [], "finish_reason": "stop"})
        if not delegated["done"]:
            delegated["done"] = True
            return EffectOutcome.completed(
                {
                    "content": "Delegating.",
                    "tool_calls": [
                        {
                            "name": "delegate_agent",
                            "arguments": {"task": "Draft it.", "substrate": "drafting"},
                            "call_id": "d1",
                        }
                    ],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "FINAL: parent done", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler},
        workflow_registry=WorkflowRegistry(),
    )
    workflow = factory(
        logic=logic_cls(tools=[DELEGATE_AGENT_TOOL]),
        workflow_id=workflow_id,
        on_step=lambda s, d: events.append((s, dict(d))),
    )
    runtime.workflow_registry.register(workflow)
    run_id = runtime.start(
        workflow=workflow,
        vars={
            "context": {"task": "Delegate the task.", "messages": []},
            "_runtime": {
                "inbox": [],
                "thinking": "high",
                "delegate_substrates": {"drafting": {**PALETTE_BASE, "thinking": "low"}},
            },
        },
        actor_id=None,
        session_id=None,
    )
    for _ in range(300):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    final = runtime.get_state(run_id)
    assert final.status == RunStatus.COMPLETED, final.error
    assert _child_runtime_ns(runtime, run_id).get("thinking") == "low"
    assert runtime.get_state(run_id).vars["_runtime"].get("thinking") == "high"
    applied = _events_named(events, "delegate_agent_substrate")
    assert applied and applied[0].get("thinking") == "low"
