"""Verifier execution preference (R-Type evidence, commons c2725/c2735/c2736).

The defect class this seam answers: LLM-read verification blessed a
crash-on-first-bullet game, an every-frame ReferenceError, and a corner-ninth
draw — only EXECUTION caught all three. The loop does NOT grow an execution
subsystem: executors are environment capabilities that ship tool-side and
declare themselves via ToolDefinition.tags ("executor"); the verifier prompt
gains a preference block teaching that an unexecuted artifact is unverified,
and the existing forced-tool-call seam (next_tool_calls -> act) runs the probe.

Contract pinned here:
- tag-driven detection (executor_tool_names), allowlist-ordered, junk-safe;
- prompt block present exactly when an executor-tagged tool is allowlisted —
  absent it, the verifier prompt is BYTE-IDENTICAL to the pre-seam text
  (deployments without executor tools see zero behavior change);
- both verifier-bearing adapters (ReAct, CodeAct) carry the block.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractagent.adapters.generation_params import (
    executor_tool_names,
    verifier_execution_preference,
)
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


# ---------------------------------------------------------------------------
# Unit half: classification + prompt block
# ---------------------------------------------------------------------------

def test_executor_tool_names_is_tag_driven_and_allowlist_ordered() -> None:
    tags = {
        "browser_probe": ("executor",),
        "run_tests": ("Executor", "read"),  # case-insensitive tag match
        "write_file": ("write",),
        "list_files": (),
    }
    allow = ["run_tests", "list_files", "browser_probe", "write_file"]
    assert executor_tool_names(allow, tool_tags=tags) == ["run_tests", "browser_probe"]


def test_executor_tool_names_survives_junk_shapes() -> None:
    assert executor_tool_names(None, tool_tags=None) == []
    assert executor_tool_names("not-a-list", tool_tags={}) == []
    assert executor_tool_names([None, "", 42], tool_tags={"42": ["executor"]}) == []
    # Unknown names (no tags entry) are simply not executors.
    assert executor_tool_names(["ghost"], tool_tags={}) == []


def test_preference_block_empty_without_executors() -> None:
    assert verifier_execution_preference([]) == ""
    assert verifier_execution_preference(None) == ""


def test_preference_block_names_executors() -> None:
    block = verifier_execution_preference(["browser_probe", "run_tests"])
    assert "Executor tools available: browser_probe, run_tests." in block
    assert "An artifact that was never executed is not verified." in block


# ---------------------------------------------------------------------------
# Integration half: the block reaches the real verifier prompts
# ---------------------------------------------------------------------------

def _drive_react_review(tools: list, allowed: list) -> Dict[str, Any]:
    """Run a ReAct workflow to completion with review on; return the captured
    verifier payload."""
    review_payloads: list[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            review_payloads.append(payload)
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_handler,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []}),
        },
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=tools),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=allowed,
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Build a game", "messages": []},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": list(allowed)},
        },
    )
    for _ in range(40):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    assert review_payloads, "verifier never ran"
    return review_payloads[0]


def test_react_verifier_prompt_carries_execution_preference_with_tagged_tool() -> None:
    probe = ToolDefinition(
        name="browser_probe", description="Execute a web artifact", parameters={}, tags=["executor"]
    )
    plain = ToolDefinition(name="list_files", description="List", parameters={})
    payload = _drive_react_review([probe, plain], ["browser_probe", "list_files"])
    prompt = str(payload.get("prompt") or "")
    assert "Executor tools available: browser_probe." in prompt
    assert "not verified" in prompt


def test_react_verifier_prompt_unchanged_without_executor_tag() -> None:
    plain = ToolDefinition(name="list_files", description="List", parameters={})
    payload = _drive_react_review([plain], ["list_files"])
    prompt = str(payload.get("prompt") or "")
    assert "Executor tools available" not in prompt
    # Pre-seam tail preserved byte-for-byte: the prompt still ENDS at the
    # allowed-tools line (empty preference block appends nothing).
    assert prompt.rstrip().endswith("Allowed tools:\nlist_files")


def test_codeact_verifier_prompt_carries_execution_preference() -> None:
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.logic.codeact import CodeActLogic

    probe = ToolDefinition(
        name="browser_probe", description="Execute a web artifact", parameters={}, tags=["executor"]
    )
    logic = CodeActLogic(tools=[probe])
    wf = create_codeact_workflow(logic=logic, on_step=None)

    captured: Dict[str, Any] = {}

    class _Ctx:
        @staticmethod
        def now_iso() -> str:
            return "2025-01-01T00:00:00+00:00"

    run = RunState(
        run_id="r-exec",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="review",
        vars={
            "context": {"task": "Build a game", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["browser_probe"]},
            "_temp": {"final_answer": "Shipped index.html."},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )
    plan = wf.get_node("review")(run, _Ctx())
    assert plan.effect is not None
    captured = dict(plan.effect.payload or {})
    prompt = str(captured.get("prompt") or "")
    assert "Executor tools available: browser_probe." in prompt


def test_codeact_verifier_prompt_unchanged_without_executor_tag() -> None:
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.logic.codeact import CodeActLogic

    plain = ToolDefinition(name="edit_file", description="edit", parameters={})
    logic = CodeActLogic(tools=[plain])
    wf = create_codeact_workflow(logic=logic, on_step=None)

    class _Ctx:
        @staticmethod
        def now_iso() -> str:
            return "2025-01-01T00:00:00+00:00"

    run = RunState(
        run_id="r-plain",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="review",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["edit_file"]},
            "_temp": {"final_answer": "done"},
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )
    plan = wf.get_node("review")(run, _Ctx())
    prompt = str(dict(plan.effect.payload or {}).get("prompt") or "")
    assert "Executor tools available" not in prompt


def test_verifier_forced_executor_call_routes_to_act() -> None:
    """The execution half of the story is the EXISTING seam: a verifier that
    proposes the executor call gets it executed through act/observe with
    proper transcript threading. Pin it with an executor-tagged tool so the
    composition (preference block -> forced probe -> act) is covered end to
    end."""
    probe = ToolDefinition(
        name="browser_probe", description="Execute a web artifact", parameters={}, tags=["executor"]
    )
    executed: list[str] = []
    review_count = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            review_count["n"] += 1
            if review_count["n"] == 1:
                return EffectOutcome.completed(
                    {
                        "data": {
                            "complete": False,
                            "missing": ["artifact never executed"],
                            "next_prompt": "",
                            "next_tool_calls": [
                                {"name": "browser_probe", "arguments": "{\"path\": \"index.html\"}"}
                            ],
                        }
                    }
                )
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        return EffectOutcome.completed({"content": "Shipped.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        tcs = effect.payload.get("tool_calls") or []
        for tc in tcs:
            executed.append(str(tc.get("name")))
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "probe green", "error": None}
                    for tc in tcs
                ],
            }
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[probe]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["browser_probe"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Build a game", "messages": []},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 2, "allowed_tools": ["browser_probe"]},
        },
    )
    for _ in range(60):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    assert executed == ["browser_probe"]
    assert review_count["n"] >= 2


# ---------------------------------------------------------------------------
# Re-review blowup containment (c2856 A/B live finding): verifier-forced
# batches must not reset the review budget — the reset re-armed the verifier's
# own budget through calls it forced itself, so a verifier that kept forcing a
# green probe re-reviewed an already-good artifact until the wall cap.
# ---------------------------------------------------------------------------

def _probe_tools() -> list:
    return [
        ToolDefinition(name="browser_probe", description="Execute", parameters={}, tags=["executor"]),
    ]


def _green_tool_handler(executed: list):
    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        tcs = effect.payload.get("tool_calls") or []
        for tc in tcs:
            executed.append(str(tc.get("name")))
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "green", "error": None}
                    for tc in tcs
                ],
            }
        )

    return tool_handler


def test_forced_green_probes_cannot_rearm_the_review_budget() -> None:
    """A verifier that NEVER says complete and keeps forcing a green probe is
    bounded by review_max_rounds — the run completes after exactly that many
    review rounds instead of looping to the iteration/wall cap (the c2856
    run2 shape: 40 minutes re-reviewing a good artifact)."""
    executed: list[str] = []
    reviews = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            reviews["n"] += 1
            return EffectOutcome.completed(
                {
                    "data": {
                        "complete": False,
                        "missing": ["still not convinced"],
                        "next_prompt": "",
                        "next_tool_calls": [{"name": "browser_probe", "arguments": "{}"}],
                    }
                }
            )
        return EffectOutcome.completed({"content": "Shipped index.html.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: _green_tool_handler(executed)},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=_probe_tools()),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["browser_probe"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Build a game", "messages": []},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 2, "allowed_tools": ["browser_probe"]},
        },
    )
    for _ in range(120):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    # The budget genuinely bounds consecutive verifier rounds on one answer.
    assert reviews["n"] == 2
    assert executed == ["browser_probe", "browser_probe"]


def test_model_issued_activity_still_rearms_the_review_budget() -> None:
    """The deliberate asymmetry: the model's OWN tool activity is a new claim
    and re-arms review — only verifier-forced batches are budget-neutral."""
    executed: list[str] = []
    reviews = {"n": 0}
    model_turn = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            reviews["n"] += 1
            if reviews["n"] == 1:
                return EffectOutcome.completed(
                    {
                        "data": {
                            "complete": False,
                            "missing": ["execute it"],
                            "next_prompt": "",
                            "next_tool_calls": [{"name": "browser_probe", "arguments": "{}"}],
                        }
                    }
                )
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        model_turn["n"] += 1
        if model_turn["n"] == 2:
            # After the forced probe's observation, the model does its OWN work.
            return EffectOutcome.completed(
                {"content": "Fixing.", "tool_calls": [{"name": "browser_probe", "arguments": {}, "call_id": "own_1"}]}
            )
        return EffectOutcome.completed({"content": "Shipped index.html.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: _green_tool_handler(executed)},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=_probe_tools()),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["browser_probe"],
    )
    rid = rt.start(
        workflow=wf,
        vars={
            "context": {"task": "Build a game", "messages": []},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["browser_probe"]},
        },
    )
    for _ in range(120):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    # Round 1 consumed by the forced probe; the model's own batch re-armed the
    # budget, so a SECOND review ran and accepted. With max_rounds=1 and no
    # re-arm, the second review could never have fired.
    assert reviews["n"] == 2


def test_codeact_forced_batch_does_not_reset_review_count() -> None:
    """CodeAct symmetry (its observe reset was the same defect): a forced
    batch leaves review_count intact; a model-issued batch resets it."""
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.logic.codeact import CodeActLogic

    class _Ctx:
        @staticmethod
        def now_iso() -> str:
            return "2025-01-01T00:00:00+00:00"

    logic = CodeActLogic(tools=_probe_tools())
    wf = create_codeact_workflow(logic=logic, on_step=None)

    def _observe_run(*, forced: bool) -> RunState:
        run = RunState(
            run_id="r-obs",
            workflow_id="codeact_agent",
            status=RunStatus.RUNNING,
            current_node="observe",
            vars={
                "context": {"task": "t", "messages": []},
                "scratchpad": {"review_count": 1},
                "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1, "allowed_tools": ["browser_probe"]},
                "_temp": {
                    "tool_results": {"results": [{"call_id": "c1", "name": "browser_probe", "success": True, "output": "green", "error": None}]},
                    **({"review_forced_batch": True} if forced else {}),
                },
                "_limits": {"max_history_messages": -1, "max_tokens": 32768},
            },
        )
        wf.get_node("observe")(run, _Ctx())
        return run

    forced_run = _observe_run(forced=True)
    assert forced_run.vars["scratchpad"]["review_count"] == 1, "forced batch must not reset"
    assert "review_forced_batch" not in forced_run.vars["_temp"], "marker ends with its batch"

    model_run = _observe_run(forced=False)
    assert model_run.vars["scratchpad"]["review_count"] == 0, "model-issued batch resets"
