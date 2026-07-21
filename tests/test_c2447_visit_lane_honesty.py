"""c2447 incident fixes (2026-07-15): loop-tail suppression + the tools_ran gauge.

The maintainer's incident (commons c2447): the entity Ephemeral read
"[loop] iteration 3 of 20" as part of its conversation, and its "0 tools"
label was unfalsifiable. Adapter-owner fixes pinned here:

- `_runtime.suppress_loop_tail` keeps the ENTIRE loop-position tail (the
  iteration line AND the [plan] render) out of every LLM payload, on all
  three loops — including the MERGE branch that lands the tail inside the
  final user message (the P0 leak path in composed entity visits). Default
  absent/falsy = task-agent behavior byte-unchanged.
- `turn_captures.tools_ran` is now written at ReAct's observe boundary:
  every result that reached execution counts (success or failure); BLOCKED
  calls (structural `blocked` marker, never error prose) never ran and are
  excluded. This feeds the visit workflow's HARVEST fold, which previously
  read a key nobody wrote — the drawer's "0 tools" was structurally zero.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.memact_runtime import create_memact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.codeact import CodeActLogic
from abstractagent.logic.memact import MemActLogic
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic

_PROBE_TOOL = ToolDefinition(name="probe", description="Probe", parameters={})
_TOOL_REPLY = {
    "content": "Looking.",
    "tool_calls": [{"name": "probe", "arguments": {"q": "x"}, "call_id": "call_1"}],
}
_FINAL_REPLY = {"content": "Done.", "tool_calls": []}
_MEMACT_FINALIZE_REPLY = {"content": json.dumps({"content": "Done."}), "tool_calls": []}


def _factory(loop: str, vars_runtime: Dict[str, Any]):
    del vars_runtime
    if loop == "react":
        return create_react_workflow(logic=ReActLogic(tools=[_PROBE_TOOL]), provider="stub", model="stub")
    if loop == "codeact":
        return create_codeact_workflow(logic=CodeActLogic(tools=[_PROBE_TOOL]))
    return create_memact_workflow(logic=MemActLogic(tools=[_PROBE_TOOL]))


def _drive(
    loop: str,
    llm_responses: List[Dict[str, Any]],
    *,
    runtime_vars: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], RunState]:
    llm_payloads: List[Dict[str, Any]] = []
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(llm_responses) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(llm_responses[idx]))

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
    rt_ns: Dict[str, Any] = {"inbox": []}
    if runtime_vars:
        rt_ns.update(runtime_vars)
    workflow = _factory(loop, rt_ns)
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Visit-lane honesty task", "messages": []}, "_runtime": rt_ns},
        actor_id=None,
        session_id=f"sess-c2447-{loop}",
    )
    for _ in range(120):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED, f"{loop}: {state.error}"
    return llm_payloads, state


def _all_payload_text(payloads: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for p in payloads:
        parts.append(str(p.get("prompt") or ""))
        parts.append(str(p.get("system_prompt") or ""))
        for m in p.get("messages") or []:
            if isinstance(m, dict):
                parts.append(str(m.get("content") or ""))
    return "\n".join(parts)


def _scripts_for(loop: str, *, with_tool: bool) -> List[Dict[str, Any]]:
    script: List[Dict[str, Any]] = [_TOOL_REPLY] if with_tool else []
    script.append(_FINAL_REPLY)
    if loop == "memact":
        script.append(_MEMACT_FINALIZE_REPLY)
    return script


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_default_payloads_carry_the_loop_tail(loop: str) -> None:
    """Task-agent behavior unchanged: without the flag, the tail rides."""
    payloads, _ = _drive(loop, _scripts_for(loop, with_tool=True))
    assert "[loop] iteration" in _all_payload_text(payloads)


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_suppress_loop_tail_keeps_every_payload_clean(loop: str) -> None:
    """The c2447 P0 fix: with `_runtime.suppress_loop_tail`, NO payload —
    merge branch (payload ends with a user message: first cycle, the entity
    visit shape) or volatile-append branch (post-tool shape) — carries the
    loop-position line, on any loop."""
    payloads, _ = _drive(
        loop,
        _scripts_for(loop, with_tool=True),
        runtime_vars={"suppress_loop_tail": True},
    )
    text = _all_payload_text(payloads)
    assert "[loop] iteration" not in text
    assert "[plan]" not in text
    # The visitor-shaped first payload still carries the actual task words —
    # suppression removes the tail, never the user's content.
    first_msgs = payloads[0].get("messages") or []
    assert any(
        "Visit-lane honesty task" in str(m.get("content") or "") for m in first_msgs if isinstance(m, dict)
    )


def test_suppress_loop_tail_string_spelling_works() -> None:
    """Run-vars arrive as strings from some hosts (tool-arg coercion lesson):
    'true' must suppress exactly like True."""
    payloads, _ = _drive(
        "react",
        [_TOOL_REPLY, _FINAL_REPLY],
        runtime_vars={"suppress_loop_tail": "true"},
    )
    assert "[loop] iteration" not in _all_payload_text(payloads)


def test_tools_ran_captured_at_observe_for_the_harvest_fold() -> None:
    """c2447 F3: executed tool names land in `_temp.turn_captures.tools_ran`
    (the key the visit workflow's HARVEST folds — previously never written)."""
    _, state = _drive("react", [_TOOL_REPLY, _FINAL_REPLY])
    captures = ((state.vars or {}).get("_temp") or {}).get("turn_captures") or {}
    assert captures.get("tools_ran") == ["probe"]


_WRITE_TOOL = ToolDefinition(name="write_file", description="Write", parameters={})
_WRITE_BATCH = {
    "content": "Writing.",
    "tool_calls": [{"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}, "call_id": "w1"}],
}


def _prior_turn_cycle() -> Dict[str, Any]:
    """A turn-1 cycle exactly as parse_node records it: same batch, all-OK
    observations — the shape the repeat guard judges against."""
    return {
        "i": 1,
        "thought": "Writing.",
        "tool_calls": [{"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}, "call_id": "w1"}],
        "observations": [
            {"call_id": "w1", "name": "write_file", "success": True, "output": "ok", "error": None, "rendered": "[write_file]: ok"}
        ],
    }


def _drive_with_seed(seed_scratchpad: Dict[str, Any]) -> Tuple[List[str], RunState]:
    """Drive one ReAct segment over a pre-seeded scratchpad (simulating the
    state a composed visit carries into a NEW turn), returning emitted steps."""
    steps: List[str] = []
    calls = {"n": 0}
    script = [_WRITE_BATCH, _FINAL_REPLY]

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, effect, default_next_node
        idx = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(script[idx]))

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
        logic=ReActLogic(tools=[_WRITE_TOOL]),
        provider="stub",
        model="stub",
        on_step=lambda s, d: steps.append(s),
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={
            "context": {"task": "t", "messages": []},
            "_runtime": {"inbox": []},
            "scratchpad": seed_scratchpad,
        },
        actor_id=None,
        session_id="sess-c2447-fence",
    )
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return steps, state


def test_repeat_guard_judges_within_the_run_without_a_fence() -> None:
    """Control (existing behavior pinned): with NO turn fence, an identical
    side-effect batch later in the run is repeat-skipped."""
    steps, _ = _drive_with_seed({"cycles": [_prior_turn_cycle()], "iteration": 1})
    assert "parse_repeat_tool_calls" in steps


def test_turn_fence_stops_the_repeat_guard_at_the_turn_boundary() -> None:
    """c2447 F5: with the fence recorded at the turn boundary (what
    reset_react_turn now writes), the guard never judges against a PRIOR
    turn's cycle — the identical batch EXECUTES."""
    steps, _ = _drive_with_seed(
        {"cycles": [_prior_turn_cycle()], "iteration": 1, "turn_first_cycle": 1}
    )
    assert "parse_repeat_tool_calls" not in steps
    assert "observe" in steps, "the batch must actually execute in the new turn"


def test_reset_react_turn_records_the_fence() -> None:
    from abstractagent.adapters.react_runtime import reset_react_turn

    vars: Dict[str, Any] = {"scratchpad": {"cycles": [_prior_turn_cycle()], "iteration": 3}}
    reset_react_turn(vars)
    assert vars["scratchpad"]["turn_first_cycle"] == 1
    assert vars["scratchpad"]["iteration"] == 0


def test_suppressed_conclusion_carries_no_loop_chrome() -> None:
    """code's C3 finding (c2500): the max-iterations CONCLUSION chrome merges
    into the last user message exactly like the reason tail — under
    suppression the whole budget-exhausted path must stay chrome-free: the
    conclusion prompt (no "Max iterations"/"ReAct"/scratchpad render), the
    retry line, and the empty-answer fallback the visitor reads."""
    steps: List[Tuple[str, Dict[str, Any]]] = []
    llm_payloads: List[Dict[str, Any]] = []
    calls = {"n": 0}
    # Budget 1: first call burns it with a tool batch; conclusion call then
    # returns EMPTY so the durable fallback fires too.
    script: List[Dict[str, Any]] = [_WRITE_BATCH, {"content": "", "tool_calls": []}]

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(script[idx]))

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
        logic=ReActLogic(tools=[_WRITE_TOOL]),
        provider="stub",
        model="stub",
        on_step=lambda s, d: steps.append((s, d)),
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={
            "context": {"task": "t", "messages": []},
            "_runtime": {"inbox": [], "suppress_loop_tail": True},
            "_limits": {"max_iterations": 1},
        },
        actor_id=None,
        session_id="sess-c2447-conclusion",
    )
    for _ in range(80):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    assert (state.output or {}).get("outcome") == "iteration_budget"

    # No loop chrome anywhere the model (or the visitor) reads.
    wire_text = _all_payload_text(llm_payloads)
    for marker in ("[loop] iteration", "Max iterations", "ReAct iterations", "Scratchpad"):
        assert marker not in wire_text, f"conclusion chrome leaked on the wire: {marker!r}"
    # The functional wrap-up instruction still reached the model.
    assert "bring your reply to a close" in wire_text
    # The durable fallback answer the visitor reads carries no machine chrome.
    answer = str((state.output or {}).get("answer") or "")
    for marker in ("Max iterations", "max_iterations", "scratchpad", "/conclude"):
        assert marker not in answer, f"loop chrome in the visitor-facing answer: {marker!r}"
    assert answer, "the fallback must still answer honestly"


def test_blocked_calls_never_count_as_ran() -> None:
    """A blocked call never executed: the structural `blocked` marker (never
    error prose) keeps it out of tools_ran — true zero stays zero.

    Scope honesty: the adapter's own allowlist gate covers BUILTIN-shaped
    calls (recall_memory etc., react_runtime act_node); external tools are
    forwarded with `allowed_tools` and refused by the RUNTIME executor —
    those refusals arrive as executed-and-failed results and count as
    attempted activity (the entity DID reach for a tool), which is the
    honest reading for the drawer gauge."""
    blocked_reply = {
        "content": "Trying.",
        "tool_calls": [{"name": "recall_memory", "arguments": {"query": "x"}, "call_id": "call_b"}],
    }
    _, state = _drive(
        "react",
        [blocked_reply, _FINAL_REPLY],
        runtime_vars={"allowed_tools": ["probe"]},
    )
    captures = ((state.vars or {}).get("_temp") or {}).get("turn_captures") or {}
    assert not captures.get("tools_ran"), "blocked builtin calls must not count as ran"


# ---------------------------------------------------------------------------
# Drained-guidance wrapper (c2447 residue closed: c2792 proposal,
# runtime voice-owner sign-off c2798, semantics adoption + 2 pins c2796)
# ---------------------------------------------------------------------------

_GUIDANCE_ITEM = {"content": "Please weave the earlier thread back in."}
_TASK_WRAPPER = "[Operator guidance — this amends the task; the final answer must satisfy it]"
_VISIT_WRAPPER = "[A note arrived during this conversation — not from your visitor]"


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_task_lane_guidance_wrapper_byte_unchanged(loop: str) -> None:
    """Without the visit knob, the historical wrapper stays byte-identical —
    task-agent transcripts and their consumers see zero change."""
    payloads, _ = _drive(
        loop,
        _scripts_for(loop, with_tool=False),
        runtime_vars={"inbox": [dict(_GUIDANCE_ITEM)]},
    )
    text = _all_payload_text(payloads)
    assert _TASK_WRAPPER in text
    assert _VISIT_WRAPPER not in text


@pytest.mark.parametrize("loop", ["react", "codeact", "memact"])
def test_visit_lane_guidance_wrapper_is_host_voiced(loop: str) -> None:
    """Under `suppress_loop_tail`, drained guidance — operator injections and
    the loops' own retry nudges alike — wears the host-voiced wrapper: no
    fabricated operator attribution, no task/final-answer vocabulary (the
    c2447 chrome class). The guidance WORDS still reach the model."""
    payloads, _ = _drive(
        loop,
        _scripts_for(loop, with_tool=False),
        runtime_vars={"suppress_loop_tail": True, "inbox": [dict(_GUIDANCE_ITEM)]},
    )
    text = _all_payload_text(payloads)
    assert _VISIT_WRAPPER in text
    assert "[Operator guidance" not in text
    assert "Please weave the earlier thread back in." in text


# ---------------------------------------------------------------------------
# Retry-nudge lane honesty (iteration-3 adversary P0, 2026-07-19): the parse
# boundary's own retry nudges ride the inbox and re-enter the transcript under
# the visit wrapper — in visit lanes their wording must be host-voiced (no
# task/tool-call vocabulary), and the followthrough heuristic must not correct
# musing ("I will read that entry again" is a legitimate visit thought).
# ---------------------------------------------------------------------------

def test_visit_lane_empty_retry_is_host_voiced() -> None:
    payloads, _ = _drive(
        "react",
        [{"content": "", "tool_calls": []}, _FINAL_REPLY],
        runtime_vars={"suppress_loop_tail": True},
    )
    text = _all_payload_text(payloads)
    assert "Continue the task" not in text
    assert "nothing reached the conversation" in text


def test_task_lane_empty_retry_byte_unchanged() -> None:
    payloads, _ = _drive("react", [{"content": "", "tool_calls": []}, _FINAL_REPLY])
    assert "Your previous response was empty. Continue the task." in _all_payload_text(payloads)


# The musing MUST genuinely match the deferred-action heuristic (intent word
# "I will" + action verb "read"), or the defaults-off pin passes vacuously —
# the explicit-check_plan control test proves the phrase trips the heuristic.
_MUSING_REPLY = {"content": "I will read that diary entry again, when the moment is right.", "tool_calls": []}


def test_visit_lane_followthrough_nudge_defaults_off() -> None:
    """Deferred-action prose ("I will read...") is musing in a visit — the
    turn completes with the prose as the answer, no corrective nudge."""
    payloads, state = _drive(
        "react",
        [dict(_MUSING_REPLY)],
        runtime_vars={"suppress_loop_tail": True},
    )
    assert "you did not call any tools" not in _all_payload_text(payloads)
    assert (state.output or {}).get("answer", "").startswith("I will read that diary entry")


def test_visit_lane_explicit_check_plan_still_wins() -> None:
    """An EXPLICIT _runtime.check_plan=true is honored even in a visit lane —
    the lane flips only the DEFAULT (and this control proves _MUSING_REPLY
    trips the heuristic, so the defaults-off pin above is not vacuous)."""
    payloads, _ = _drive(
        "react",
        [dict(_MUSING_REPLY), _FINAL_REPLY],
        runtime_vars={"suppress_loop_tail": True, "check_plan": True},
    )
    assert "you did not call any tools" in _all_payload_text(payloads)


def test_visit_lane_truncation_retry_is_host_voiced() -> None:
    truncated = {"content": "I was starting to say", "tool_calls": [], "finish_reason": "length"}
    payloads, _ = _drive(
        "react",
        [truncated, _FINAL_REPLY],
        runtime_vars={"suppress_loop_tail": True},
    )
    text = _all_payload_text(payloads)
    # The task-lane wording (tool-call coaching) must not reach a visit; the
    # host-voiced line must. Assert on the nudge strings, not the whole
    # payload (the persona/system prompt legitimately mentions tools).
    assert "output token limit" not in text
    assert "Pick up where it stopped" in text


def test_guidance_machine_anchor_is_metadata_never_prose() -> None:
    """Semantics pin 2 (c2796): the wrapper is for the entity's reading only
    — a visitor can type the same bytes, so machine detection of drained
    guidance keys on the durable message metadata (kind="operator_guidance",
    which deliberately does NOT rename with the visible string), never on
    the bracket prose."""
    _, state = _drive(
        "react",
        [_FINAL_REPLY],
        runtime_vars={"suppress_loop_tail": True, "inbox": [dict(_GUIDANCE_ITEM)]},
    )
    messages = ((state.vars or {}).get("context") or {}).get("messages") or []
    drained = [
        m for m in messages
        if isinstance(m, dict) and (m.get("metadata") or {}).get("kind") == "operator_guidance"
    ]
    assert len(drained) == 1, "the structured anchor must identify exactly the drained message"
    assert _VISIT_WRAPPER in str(drained[0].get("content") or "")
