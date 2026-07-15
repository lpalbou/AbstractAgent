"""Pins for the adversary-driven hardening pass (2026-07-13 evening).

Regression adversary (wave C) + design adversary (wave D) findings, fixed
same-evening. Every pin here would fail if its fix were reverted:
- C-P0: shipped abstractcode call shapes construct (deprecation shims, not
  TypeErrors) — pinned in test_sibling_loop_defects_batch (memact) and here
  for the Logic constructors.
- C-P1-1: CodeAct plan/review carry the per-run provider/model override
  (split-brain fix).
- C-P1-2: ReAct's explicit output cap reaches the REVIEW call.
- C-P2-3: sibling volatile tails carry `volatile: True` (cache fingerprint
  exclusion) and MERGE into a trailing user message (alternation safety).
- D-P1: outcome vocabulary is the canonical turn_end one, mirrored in raw
  emits; CodeAct review_skipped is an always-present bool.
- D-P1: palette renders granted names into the system prompt; unknown profile
  keys warn (skew emit); malformed-vs-unknown errors are distinct.
- D-P1: defaults refuse cross-provider guesses (pinned in batch2 test).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.emit_inventory import RENAMED_STEPS
from abstractagent.adapters.generation_params import compose_prompt_slots
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import RunState, RunStatus


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_EXEC = ToolDefinition(name="execute_python", description="run", parameters={})
_LS = ToolDefinition(name="list_files", description="ls", parameters={})


def test_logic_constructors_accept_legacy_kwargs_without_breaking() -> None:
    """C-P0: abstractcode's exact call shapes (workflow_agent.py:520 passes
    max_tokens=None) must construct; meaningful values warn, legacy no-op
    spellings stay silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ReActLogic(tools=[], max_tokens=None)  # the shipped abstractcode shape
        ReActLogic(tools=[], max_history_messages=-1)
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ReActLogic(tools=[], max_tokens=4096)  # meaningful -> loud
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _codeact_vars(runtime_extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "context": {"task": "do the thing", "messages": []},
        "scratchpad": {"iteration": 0},
        "_runtime": dict({"inbox": [], "allowed_tools": ["execute_python"]}, **(runtime_extra or {})),
        "_temp": {},
        "_limits": {"max_iterations": 20, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
    }


def _plan(wf, node: str, vars: Dict[str, Any]):
    run = RunState(run_id="r", workflow_id="wf", status=RunStatus.RUNNING, current_node=node, vars=vars)
    return run, wf.get_node(node)(run, _Ctx())


def test_codeact_plan_and_review_carry_per_run_routing() -> None:
    """C-P1-1: reason carried the override while plan/review ran on the
    runtime default — a split-brain run (worse with review defaulting ON)."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    override = {"provider": "lmstudio", "model": "routed-model", "plan_mode": True}

    vars_plan = _codeact_vars(override)
    _, plan = _plan(wf, "plan", vars_plan)
    assert plan.effect is not None
    assert plan.effect.payload.get("provider") == "lmstudio"
    assert plan.effect.payload.get("model") == "routed-model"

    vars_review = _codeact_vars(dict(override, review_mode=True))
    vars_review["_temp"]["final_answer"] = "candidate answer"
    vars_review["scratchpad"]["review_rounds"] = 0
    _, review = _plan(wf, "review", vars_review)
    assert review.effect is not None
    assert review.effect.payload.get("provider") == "lmstudio"
    assert review.effect.payload.get("model") == "routed-model"


def test_react_review_call_carries_explicit_output_cap() -> None:
    """C-P1-2: with review on the default path, the verifier was the one
    unbounded call type under an explicit cap (init nulls _limits; only the
    _runtime channel survives — review now reads it)."""
    wf = create_react_workflow(
        logic=ReActLogic(tools=[_LS]), workflow_id="wf", provider="stub", model="stub",
        allowed_tools=["list_files"],
    )
    vars: Dict[str, Any] = {
        "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
        "scratchpad": {"iteration": 1, "cycles": [], "review_rounds": 0},
        "_runtime": {"inbox": [], "allowed_tools": ["list_files"], "review_mode": True, "max_output_tokens": 512},
        "_temp": {"final_answer": "candidate"},
        "_limits": {"max_iterations": 20, "max_output_tokens": None},
    }
    _, plan = _plan(wf, "review", vars)
    assert plan.effect is not None
    assert (plan.effect.payload.get("params") or {}).get("max_tokens") == 512


def test_sibling_volatile_tail_flag_and_merge_branch() -> None:
    """C-P2-3: the tail must carry `volatile: True` when appended (cache
    fingerprint exclusion) and MERGE into a trailing user message when the
    transcript ends with one (alternation-strict template safety)."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))

    # Transcript ending in an ASSISTANT message -> appended volatile tail.
    vars_a = _codeact_vars()
    vars_a["context"]["messages"] = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "thinking"},
    ]
    _, plan_a = _plan(wf, "reason", vars_a)
    msgs = plan_a.effect.payload["messages"]
    assert msgs[-1]["role"] == "user"
    assert "[loop] iteration" in msgs[-1]["content"]
    assert msgs[-1].get("volatile") is True

    # Transcript ending in a USER message -> merged, no volatile flag, no
    # user,user adjacency.
    vars_b = _codeact_vars()
    vars_b["context"]["messages"] = [{"role": "user", "content": "task"}]
    _, plan_b = _plan(wf, "reason", vars_b)
    msgs_b = plan_b.effect.payload["messages"]
    assert msgs_b[-1]["role"] == "user"
    assert "task" in msgs_b[-1]["content"] and "[loop] iteration" in msgs_b[-1]["content"]
    assert sum(1 for m in msgs_b if m.get("role") == "user") == 1


def test_outcome_uses_canonical_turn_end_vocabulary_and_mirrors_in_emits() -> None:
    """D-P1: one enum, one vocabulary (final_answer | iteration_budget) —
    output field and raw emits agree with loop_hooks' canonical mapping."""
    events: list[tuple[str, dict]] = []
    wf = create_codeact_workflow(
        logic=CodeActLogic(tools=[_EXEC]),
        on_step=lambda step, data: events.append((step, dict(data))),
    )
    vars = _codeact_vars()
    vars["_temp"]["final_answer"] = "done!"
    run, plan = _plan(wf, "done", vars)
    out = plan.complete_output
    assert out["outcome"] == "final_answer"
    assert out["review_skipped"] is False  # always-present bool (one shape)
    done_emits = [d for s, d in events if s == "done"]
    assert done_emits and done_emits[0].get("outcome") == "final_answer"


def test_palette_renders_into_prompt_and_warns_on_skew() -> None:
    """D-P1: a grant the model cannot SEE manufactures dead guesses — granted
    names render into the stable prefix; unknown profile keys warn loudly."""
    ns = {
        "delegate_substrates": {
            "deep": {"provider": "lmstudio", "model": "big", "description": "large-context reasoning"},
            "fast": {"provider": "ollama", "model": "small"},
        }
    }
    sys = compose_prompt_slots("BASE", ns)
    assert "Available delegate substrates" in sys
    assert "- deep: large-context reasoning" in sys
    assert "- fast" in sys
    assert "lmstudio" not in sys  # names are the vocabulary, never wire ids

    # Skew warning: unknown profile keys emit #FALLBACK (still applied).
    events: list[tuple[str, dict]] = []
    wf = create_react_workflow(
        logic=ReActLogic(tools=[DELEGATE_AGENT_TOOL, _LS]), workflow_id="wf",
        provider="stub", model="stub", allowed_tools=["list_files", "delegate_agent"],
        on_step=lambda step, data: events.append((step, dict(data))),
    )
    run = RunState(
        run_id="r-skew", workflow_id="wf", status=RunStatus.RUNNING, current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": 20, "cycles": []},
            "_runtime": {
                "inbox": [], "allowed_tools": ["list_files", "delegate_agent"],
                "delegate_substrates": {"deep": {"provider": "p", "model": "m", "temperature": 0.1}},
            },
            "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": {"task": "sub", "substrate": "deep"}, "call_id": "d1"}]},
            "_limits": {"max_iterations": 20},
        },
    )
    plan = wf.get_node("act")(run, _Ctx())
    assert plan.effect is not None  # applied despite skew
    skew = [d for s, d in events if s == "delegate_agent_substrate_skew"]
    assert skew and skew[0]["ignored_keys"] == ["temperature"] and "#FALLBACK" in skew[0]["warning"]

    # Malformed ≠ unknown: a granted-but-broken profile names the real problem.
    run2 = RunState(
        run_id="r-mal", workflow_id="wf", status=RunStatus.RUNNING, current_node="act",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": 20, "cycles": []},
            "_runtime": {
                "inbox": [], "allowed_tools": ["list_files", "delegate_agent"],
                "delegate_substrates": {"deep": {"provider": "p"}},
            },
            "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": {"task": "sub", "substrate": "deep"}, "call_id": "d2"}]},
            "_limits": {"max_iterations": 20},
        },
    )
    wf.get_node("act")(run2, _Ctx())
    err = run2.vars["_temp"]["tool_results"]["results"][0]["error"]
    assert "granted but malformed" in err


def test_renamed_steps_records_the_founding_rename() -> None:
    assert RENAMED_STEPS == {"parse_retry_empty_response": "parse_retry_empty"}
