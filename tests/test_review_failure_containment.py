"""Review/verifier failure degrades to accept-with-#FALLBACK (backlog 0027).

Live incident class (commons c1128, 2026-07-12): with review_mode enabled, the
verifier's structured LLM call failed validation and the run DIED while
`_temp.final_answer` already held a valid answer — a verification aid killing a
run that had succeeded. Contract pinned here:

- the review LLM_CALL opts into the runtime's failure absorption
  (`payload._absorb_failure`, the fdf01e0 rule class): a terminally failed
  verifier call lands as {"ok": False, "absorbed_failure": <error>} at
  result_key instead of failing the run;
- review_parse contains the absorbed record: the held answer is ACCEPTED, the
  run COMPLETES, and the skip is loud — #FALLBACK marker in the scratchpad,
  a `review: #FALLBACK skipped (...)` line in the report, and a dedicated
  `review_skipped` emit;
- the review SUCCESS path is byte-unchanged (no marker, no skip event).

Runtimes without the absorption mechanism ignore the payload key — there the
failure still fails the run exactly as before (no silent behavior change); the
kernel tests below run against the sibling runtime tree, which carries it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars(*, task: str, runtime_ns: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "context": {"task": task, "messages": []},
        "_runtime": dict({"inbox": []}, **(runtime_ns or {})),
    }


def _react_workflow(on_step=None):
    return create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        on_step=on_step,
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )


def test_react_review_failure_accepts_held_answer_with_fallback() -> None:
    """Kernel proof: scripted review-effect FAILURE with a held final answer ->
    the run COMPLETES with that answer + loud #FALLBACK review-skipped markers."""
    review_payloads: list[dict[str, Any]] = []
    events: list[tuple[str, dict[str, Any]]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            review_payloads.append(payload)
            # Terminal failure (validation-class): retryable=False reaches the
            # absorption layer without burning retry attempts.
            return EffectOutcome.failed(
                "1 validation error for ReActVerifier_next_tool_callsItem", retryable=False
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
    wf = _react_workflow(on_step=lambda step, data: events.append((step, dict(data))))
    rid = rt.start(workflow=wf, vars=_base_vars(task="Say hi", runtime_ns={"review_mode": True}))
    for _ in range(40):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            break

    final = rt.get_state(rid)
    # The run completed with the HELD answer — the failed verifier never killed it.
    assert final.status == RunStatus.COMPLETED
    assert isinstance(final.output, dict)
    assert final.output.get("answer") == "All done."

    # The verifier genuinely ran (and failed) with the absorption flag on the wire.
    assert len(review_payloads) == 1
    assert review_payloads[0].get("_absorb_failure") is True

    # Loud, not silent: scratchpad marker + report line + dedicated emit.
    skipped = final.vars["scratchpad"].get("review_skipped")
    assert isinstance(skipped, list) and len(skipped) == 1
    assert skipped[0]["warning"] == "#FALLBACK"
    assert "validation error" in skipped[0]["reason"]
    report = str(final.output.get("report") or "")
    assert "review: #FALLBACK skipped" in report
    skip_events = [d for s, d in events if s == "review_skipped"]
    assert len(skip_events) == 1
    assert skip_events[0].get("accepted_held_answer") is True
    assert skip_events[0].get("warning") == "#FALLBACK"

    # The stale review response never leaks into the next turn's temp.
    assert "review_llm_response" not in (final.vars.get("_temp") or {})


def test_react_review_success_path_unchanged() -> None:
    """Guard: a healthy verifier round leaves no skip marker and no skip event."""
    events: list[str] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        if str(payload.get("response_schema_name") or "") == "ReActVerifier":
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_handler,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []}),
        },
    )
    wf = _react_workflow(on_step=lambda step, data: events.append(step))
    rid = rt.start(workflow=wf, vars=_base_vars(task="Say hi", runtime_ns={"review_mode": True}))
    for _ in range(40):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            break

    final = rt.get_state(rid)
    assert final.status == RunStatus.COMPLETED
    assert "review_skipped" not in final.vars["scratchpad"]
    assert "review_skipped" not in events
    assert "review: #FALLBACK skipped" not in str((final.output or {}).get("report") or "")


def test_codeact_review_parse_contains_absorbed_failure() -> None:
    """CodeAct carries the same verifier fork (pre-C5/0021): the absorbed record
    must route to done — NOT fall through into the unactionable-retry path,
    which would re-issue the failing call and then re-enter reason."""
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.logic.codeact import CodeActLogic

    class _Ctx:
        @staticmethod
        def now_iso() -> str:
            return "2025-01-01T00:00:00+00:00"

    events: list[tuple[str, dict[str, Any]]] = []
    wf = create_codeact_workflow(
        logic=CodeActLogic(tools=[ToolDefinition(name="edit_file", description="edit", parameters={})]),
        on_step=lambda step, data: events.append((step, dict(data))),
    )

    run = RunState(
        run_id="r-absorb",
        workflow_id="codeact_agent",
        status=RunStatus.RUNNING,
        current_node="review_parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"review_count": 1},
            "_runtime": {"inbox": [], "review_mode": True, "review_max_rounds": 1},
            "_temp": {
                "final_answer": "Already fixed.",
                "review_llm_response": {"ok": False, "absorbed_failure": "verifier call failed"},
            },
            "_limits": {"max_history_messages": -1, "max_tokens": 32768},
        },
    )

    plan = wf.get_node("review_parse")(run, _Ctx())
    assert plan.next_node == "done"
    skipped = run.vars["scratchpad"].get("review_skipped")
    assert isinstance(skipped, list) and skipped[0]["warning"] == "#FALLBACK"
    assert skipped[0]["reason"] == "verifier call failed"
    assert "review_llm_response" not in run.vars["_temp"]
    assert any(s == "review_skipped" for s, _ in events)
    # The unactionable-retry counter was never touched (that path did not run).
    assert not run.vars["_runtime"].get("review_retry_count")
