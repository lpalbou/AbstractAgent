"""Operator-directed improvement wave (2026-07-13, two fable5 adversaries).

Pins for the fix set:
- B-F5: sibling system prompts are BYTE-STABLE across iterations (the
  `Iteration: N/M` head line busted the provider prefix cache every cycle);
  loop position rides a trailing volatile message instead (ReAct's 0212
  pattern propagated).
- A-F1: `_runtime.system_prompt_extra` is COMPOSED in CodeAct/MemAct (both
  wrote the sub-agent directive into delegated children but never read it —
  silently dropped on every sibling delegation).
- A-F7/B-F6: delegated children inherit the parent's effective
  provider/model/temperature/seed (runtime.start seeded them from CONFIG, so
  per-run overrides silently reverted mid-tree).
- A-F8: terminal outputs carry a machine-readable `outcome` field.
- c1613 (code seat): observe emits carry `call_id` for fleet correlation.
- B-F1: ReactAgent's review_mode defaults ON (the recorded re-flip condition
  — 0027 containment — was met; CodeAct already defaulted True).
- B-F3: one-shot `context_warning` when usage crosses warn_tokens_pct.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.generation_params import context_usage_warning
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_EXEC = ToolDefinition(name="execute_python", description="run", parameters={})


def test_sibling_system_prompts_are_byte_stable_across_iterations() -> None:
    """B-F5: iteration must NOT appear in the system prompt (cache prefix)."""
    ca = CodeActLogic(tools=[_EXEC])
    ma = MemActLogic(tools=[_EXEC])
    for logic in (ca, ma):
        r1 = logic.build_request(task="t", messages=[], iteration=1, max_iterations=20, vars={})
        r5 = logic.build_request(task="t", messages=[], iteration=5, max_iterations=20, vars={})
        assert r1.system_prompt == r5.system_prompt
        assert "Iteration" not in r1.system_prompt


def _reason_payload(wf, node: str, vars: Dict[str, Any]):
    run = RunState(
        run_id="r", workflow_id="wf", status=RunStatus.RUNNING, current_node=node, vars=vars
    )
    plan = wf.get_node(node)(run, _Ctx())
    assert plan.effect is not None
    return plan.effect.payload or {}


def _codeact_vars(runtime_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "context": {"task": "do the thing", "messages": []},
        "scratchpad": {"iteration": 0},
        "_runtime": dict({"inbox": [], "allowed_tools": ["execute_python"]}, **(runtime_extra or {})),
        "_temp": {},
        "_limits": {"max_iterations": 20, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
    }


def test_codeact_reason_carries_volatile_loop_tail_and_stable_prefix() -> None:
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    p1 = _reason_payload(wf, "reason", _codeact_vars())
    v2 = _codeact_vars()
    v2["scratchpad"]["iteration"] = 7
    v2["_limits"]["current_iteration"] = 7
    p2 = _reason_payload(wf, "reason", v2)
    # System prompt byte-identical across cycles.
    assert p1.get("system_prompt") == p2.get("system_prompt")
    # Loop position rides the messages (merged into the trailing user task or
    # a volatile-flagged tail), never the system prompt.
    def _tail_text(p):
        msgs = p.get("messages") or []
        return str(msgs[-1].get("content") or "") if msgs else ""
    assert "[loop] iteration 1 of 20." in _tail_text(p1)
    assert "[loop] iteration 8 of 20." in _tail_text(p2)


def test_sibling_system_prompt_extra_reaches_the_payload() -> None:
    """A-F1: the delegated-child directive (and any host append: skills block,
    unattended directive) must reach the model in CodeAct AND MemAct."""
    directive = "You are a delegated sub-agent."
    wf_c = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    p = _reason_payload(wf_c, "reason", _codeact_vars({"system_prompt_extra": directive}))
    assert directive in str(p.get("system_prompt") or "")

    wf_m = create_memact_workflow(logic=MemActLogic(tools=[_EXEC]))
    vars_m = {
        "context": {"task": "do the thing", "messages": []},
        "scratchpad": {"iteration": 0},
        "_runtime": {"inbox": [], "allowed_tools": ["execute_python"], "system_prompt_extra": directive},
        "_temp": {},
        "_limits": {"max_iterations": 20, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
    }
    p_m = _reason_payload(wf_m, "reason", vars_m)
    assert directive in str(p_m.get("system_prompt") or "")


def test_delegate_children_inherit_parent_substrate_and_sampling() -> None:
    """A-F7/B-F6: provider/model/temperature/seed flow into sub_vars."""
    parent_runtime = {
        "inbox": [],
        "allowed_tools": ["execute_python", "delegate_agent"],
        "provider": "lmstudio",
        "model": "some-model",
        "temperature": 0.0,
        "seed": 42,
    }
    for factory, logic_cls, wf_id in (
        (create_codeact_workflow, CodeActLogic, "codeact_agent"),
        (create_memact_workflow, MemActLogic, "memact_agent"),
    ):
        wf = factory(logic=logic_cls(tools=[DELEGATE_AGENT_TOOL, _EXEC]))
        run = RunState(
            run_id="r-del", workflow_id=wf_id, status=RunStatus.RUNNING, current_node="act",
            vars={
                "context": {"task": "t", "messages": []},
                "scratchpad": {"iteration": 1, "max_iterations": 20},
                "_runtime": dict(parent_runtime),
                "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": {"task": "sub"}, "call_id": "d1"}]},
                "_limits": {"max_iterations": 20, "max_history_messages": -1, "max_tokens": 32768},
            },
        )
        plan = wf.get_node("act")(run, _Ctx())
        assert plan.effect is not None
        child_rt = plan.effect.payload["vars"]["_runtime"]
        assert child_rt["provider"] == "lmstudio"
        assert child_rt["model"] == "some-model"
        assert child_rt["temperature"] == 0.0
        assert child_rt["seed"] == 42


def test_codeact_honors_per_run_provider_model_override() -> None:
    """B-F10: `_runtime.provider/model` (the gateway per-run routing channel)
    reaches CodeAct's reason payload like it always did for ReAct/MemAct."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    p = _reason_payload(wf, "reason", _codeact_vars({"provider": "lmstudio", "model": "override-model"}))
    assert p.get("provider") == "lmstudio"
    assert p.get("model") == "override-model"


def test_terminal_outputs_carry_machine_readable_outcome() -> None:
    """A-F8 kernel proof on ReAct: done -> outcome final_answer (the
    canonical turn_end vocabulary — one enum, stream and output)."""
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic
    from abstractruntime.core.models import Effect, EffectType
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    def llm(run, effect, dnn):
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []}),
        },
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="ls", parameters={})]),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["list_files"],
    )
    rid = rt.start(workflow=wf, vars={"context": {"task": "t", "messages": []}, "_runtime": {"inbox": []}})
    for _ in range(40):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            break
    final = rt.get_state(rid)
    assert final.status == RunStatus.COMPLETED
    assert final.output.get("outcome") == "final_answer"
    assert final.output.get("review_skipped") is False


def test_observe_emit_carries_call_id() -> None:
    """c1613: fleet controllers correlate tool_call -> approval -> tool_result
    by call_id; the observe emit now carries it (additive, backward-safe)."""
    events: list[tuple[str, dict]] = []
    wf = create_codeact_workflow(
        logic=CodeActLogic(tools=[_EXEC]),
        on_step=lambda step, data: events.append((step, dict(data))),
    )
    vars = _codeact_vars()
    vars["_temp"]["tool_results"] = {
        "results": [{"call_id": "c-42", "name": "execute_python", "success": True, "output": "ok", "error": None}]
    }
    run = RunState(run_id="r-obs", workflow_id="codeact_agent", status=RunStatus.RUNNING, current_node="observe", vars=vars)
    wf.get_node("observe")(run, _Ctx())
    obs = [d for s, d in events if s == "observe"]
    assert obs and obs[0].get("call_id") == "c-42"


def test_react_review_mode_defaults_on() -> None:
    """B-F1: the recorded re-flip condition (0027 containment) was met —
    ReactAgent seeds review_mode=True into _runtime by default."""
    from abstractagent.agents.react import ReactAgent
    from abstractruntime.core.runtime import Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(), effect_handlers={})
    agent = ReactAgent(runtime=rt, tools=[])
    assert agent._review_mode is True
    agent_off = ReactAgent(runtime=rt, tools=[], review_mode=False)
    assert agent_off._review_mode is False


def test_context_usage_warning_latches_once() -> None:
    """B-F3: fires exactly once at the threshold; silent below it."""
    lim = {"estimated_tokens_used": 30000, "max_tokens": 32768, "warn_tokens_pct": 80}
    sp: Dict[str, Any] = {}
    w1 = context_usage_warning(lim, sp)
    assert w1 is not None and "#FALLBACK" in w1["warning"]
    assert context_usage_warning(lim, sp) is None  # latched
    assert context_usage_warning({"estimated_tokens_used": 10, "max_tokens": 32768, "warn_tokens_pct": 80}, {}) is None
