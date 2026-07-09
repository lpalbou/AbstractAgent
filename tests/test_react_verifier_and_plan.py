"""ReAct verifier + update_plan wiring (backlog 0217b)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus  # noqa: F401
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


def _base_vars(*, task: str, runtime_ns: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "context": {"task": task, "messages": []},
        "_runtime": dict({"inbox": []}, **(runtime_ns or {})),
    }


def test_verifier_forces_another_act_round_when_incomplete() -> None:
    """With review_mode on, an initial 'final answer' is re-checked; if the verifier says
    incomplete with next_tool_calls, the loop runs another act round before finishing."""
    llm_calls: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        is_review = str(payload.get("response_schema_name") or "") == "ReActVerifier"
        llm_calls.append({"review": is_review, "payload": payload})
        if is_review:
            # First verification: not complete, propose a concrete tool call.
            n_reviews = sum(1 for c in llm_calls if c["review"])
            if n_reviews == 1:
                return EffectOutcome.completed(
                    {
                        "data": {
                            "complete": False,
                            "missing": ["must list files"],
                            "next_prompt": "",
                            "next_tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}}],
                        }
                    }
                )
            # Second verification: complete.
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        # Non-review LLM turn: the model declares it is done with no tool calls.
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        tcs = effect.payload.get("tool_calls") or []
        return EffectOutcome.completed(
            {"mode": "executed", "results": [{"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None} for tc in tcs]}
        )

    rt = Runtime(
        run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["list_files"],
    )
    rid = rt.start(workflow=wf, vars=_base_vars(task="Explore", runtime_ns={"review_mode": True, "review_max_rounds": 2}))
    for _ in range(60):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    final = rt.get_state(rid)
    assert final.status == RunStatus.COMPLETED
    # The verifier ran (at least twice) and forced a tool execution the base loop hadn't done.
    assert sum(1 for c in llm_calls if c["review"]) >= 2

    # BUG-FIX regression (adversarial review): verifier-forced tool calls must NOT orphan the
    # transcript. Every role="tool" message must have a preceding assistant message carrying a
    # tool_call with the SAME id — otherwise strict OpenAI-compatible providers 400 the next call.
    msgs = final.vars["context"]["messages"]
    announced: set[str] = set()
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m["tool_calls"]:
                cid = str((tc.get("id") if isinstance(tc, dict) else "") or "")
                if cid:
                    announced.add(cid)
        elif m.get("role") == "tool":
            cid = str((m.get("metadata") or {}).get("call_id") or "")
            assert cid in announced, f"orphan tool message: call_id={cid!r} not announced by any assistant tool_calls"


def test_update_plan_is_advertised_to_the_model() -> None:
    """BUG-FIX regression: update_plan must be in the tool set the model actually sees, not just
    reachable in scripted tests (it was dead in production before)."""
    from abstractagent.logic.builtins import UPDATE_PLAN_TOOL

    assert UPDATE_PLAN_TOOL.name == "update_plan"

    # ReactAgent prepends the builtin schemas; update_plan must be among them.
    from abstractagent.agents.react import ReactAgent

    rt = Runtime(run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore())
    agent = ReactAgent(runtime=rt, tools=[])
    advertised = {getattr(t, "name", "") for t in agent.logic.tools}
    assert "update_plan" in advertised


def test_review_mode_off_finishes_without_verifier() -> None:
    llm_calls: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        llm_calls.append(payload)
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []})},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["list_files"],
    )
    rid = rt.start(workflow=wf, vars=_base_vars(task="Say hi"))  # review_mode not set -> off
    for _ in range(20):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    assert not any(str(p.get("response_schema_name") or "") == "ReActVerifier" for p in llm_calls)


def test_update_plan_persists_and_renders_at_tail() -> None:
    captured: list[dict[str, Any]] = []
    step = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        captured.append(dict(effect.payload or {}))
        step["n"] += 1
        if step["n"] == 1:
            return EffectOutcome.completed(
                {"content": "Planning.", "tool_calls": [{"name": "update_plan", "arguments": {"plan": ["step one", "step two"]}, "call_id": "p1"}]}
            )
        return EffectOutcome.completed({"content": "Done.", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(), ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []})},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["list_files"],
    )
    rid = rt.start(workflow=wf, vars=_base_vars(task="Plan then finish"))
    for _ in range(30):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert rt.get_state(rid).status == RunStatus.COMPLETED
    # The plan persisted to the scratchpad.
    assert rt.get_state(rid).vars["scratchpad"].get("plan") == "- step one\n- step two"
    # The second LLM call carries the plan in the trailing message (cache-safe placement).
    second = captured[1]
    msgs = second.get("messages") or []
    assert msgs and "[plan]" in str(msgs[-1].get("content") or "")
    assert "step one" in str(msgs[-1].get("content") or "")
    # It is NOT in the (cache-stable) system prompt.
    assert "step one" not in str(second.get("system_prompt") or "")
