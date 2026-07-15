"""Pins for the gated-remainder wave (2026-07-14, operator green light).

- A1: `codeact_fenced_fallback` default is capability-conditional (NOT
  supports_native_tools; explicit flag wins both ways; absent bit -> True),
  and the PROMPT teaches the fence only when the parser will extract it (one
  resolver — teaching a disabled channel manufactures dead replies).
- read_skill: schema-only builtin (host-owned execution; the open_attachment
  precedent).
- Side-effect classifier: shared, deny-safe, origin-READY (curated names +
  mcp:: prefix + ToolDefinition.tags).
- reset_react_task: pure vars transform — per-task state resets, door-owned
  state persists, input dict not mutated.
- 0028: one turn_end per budget-exhausted turn (announce step separate);
  review_skipped resets at the ask_user turn boundary; parse payload common
  core (has_tool_calls + tool_calls + content_preview) across all three loops.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.generation_params import is_side_effect_tool, tool_tags_map
from abstractagent.agents.task_reset import reset_react_task
from abstractagent.logic.builtins import READ_SKILL_TOOL
from abstractagent.logic.codeact import CodeActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


class _Ctx:
    @staticmethod
    def now_iso() -> str:
        return "2025-01-01T00:00:00+00:00"


_EXEC = ToolDefinition(name="execute_python", description="run", parameters={})


def test_fenced_fallback_default_is_capability_conditional() -> None:
    f = CodeActLogic.fenced_fallback_enabled
    assert f({}) is True  # absent bit -> fail-safe ON
    assert f({"supports_native_tools": True}) is False
    assert f({"supports_native_tools": False}) is True
    # Explicit flag wins both ways, incl. string spellings (_flag parity).
    assert f({"supports_native_tools": True, "codeact_fenced_fallback": True}) is True
    assert f({"supports_native_tools": False, "codeact_fenced_fallback": False}) is False
    assert f({"supports_native_tools": True, "codeact_fenced_fallback": "on"}) is True
    assert f({"codeact_fenced_fallback": "false"}) is False
    # Wave-F P3: the descriptive string key (same seeder) is an equivalent
    # capability signal — a host seeding only tool_support must not get the
    # competing-channel fence on a native model.
    assert f({"tool_support": "native"}) is False
    assert f({"tool_support": "prompted"}) is True


def test_fenced_fallback_distrusts_stale_capability_bit_on_routed_runs() -> None:
    """Wave-F P1: the seeded bit describes the model it was DERIVED from
    (`tool_support_model` stamp); a per-run `_runtime.model` override to a
    different model makes it stale. Fail toward fence ON — a prompted model
    without the fence completes SILENTLY with code-as-prose."""
    f = CodeActLogic.fenced_fallback_enabled
    # Routed to a different model than the bit describes -> distrust -> ON.
    assert f({"supports_native_tools": True, "tool_support_model": "big-native", "model": "small-prompted"}) is True
    # Same model (case-insensitive) -> trust the bit -> OFF.
    assert f({"supports_native_tools": True, "tool_support_model": "Big-Native", "model": "big-native"}) is False
    # No stamp (older seeder) -> current behavior: trust the bit.
    assert f({"supports_native_tools": True, "model": "small-prompted"}) is False
    # Explicit flag still beats the distrust rule.
    assert (
        f({"supports_native_tools": True, "tool_support_model": "a", "model": "b", "codeact_fenced_fallback": False})
        is False
    )


def test_prompt_fence_line_matches_the_parser_gate() -> None:
    """One resolver, two surfaces: when extraction is off, the prompt must not
    teach the fence; when on, it must."""
    logic = CodeActLogic(tools=[_EXEC])
    native_vars = {"_runtime": {"supports_native_tools": True}, "_limits": {}}
    prompted_vars = {"_runtime": {"supports_native_tools": False}, "_limits": {}}
    req_native = logic.build_request(task="t", messages=[], vars=native_vars)
    req_prompted = logic.build_request(task="t", messages=[], vars=prompted_vars)
    assert "fenced" not in req_native.system_prompt
    assert "fenced ```python" in req_prompted.system_prompt


def test_codeact_parse_extraction_follows_capability_bit() -> None:
    steps: List[tuple] = []
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]), on_step=lambda s, d: steps.append((s, d)))
    reply = "Here is the fix:\n```python\nprint('hi')\n```"

    def _vars(native: bool) -> Dict[str, Any]:
        return {
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1},
            "_runtime": {"inbox": [], "allowed_tools": ["execute_python"], "supports_native_tools": native},
            "_temp": {"llm_response": {"content": reply, "tool_calls": []}},
            "_limits": {"max_iterations": 20, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        }

    # Native model: fence NOT extracted — the REAL contract is routing (wave-E
    # adversary P1-2, proven by mutation: CodeAct's parse returns no effect on
    # ANY branch, extraction routes via next_node="execute_code", so the old
    # effect-shaped assertion passed against the pre-A1 gate too).
    run_n = RunState(run_id="rn", workflow_id="codeact_agent", status=RunStatus.RUNNING, current_node="parse", vars=_vars(True))
    plan_n = wf.get_node("parse")(run_n, _Ctx())
    assert plan_n.next_node != "execute_code"
    assert "pending_code" not in run_n.vars["_temp"]
    # Wave-F P3: the parse emit carries the code signal — a common-core
    # consumer must see the action coming (has_tool_calls=False alone hid it).
    parse_native = [d for s, d in steps if s == "parse"][-1]
    assert parse_native["has_code"] is False

    # Prompted model: fence extracted (the taught convention) -> execute path.
    run_p = RunState(run_id="rp", workflow_id="codeact_agent", status=RunStatus.RUNNING, current_node="parse", vars=_vars(False))
    plan_p = wf.get_node("parse")(run_p, _Ctx())
    assert plan_p.next_node == "execute_code"
    assert run_p.vars["_temp"].get("pending_code")
    parse_prompted = [d for s, d in steps if s == "parse"][-1]
    assert parse_prompted["has_code"] is True
    assert parse_prompted["has_tool_calls"] is False


def test_read_skill_is_schema_only_and_not_in_defaults() -> None:
    assert READ_SKILL_TOOL.name == "read_skill"
    assert "name" in READ_SKILL_TOOL.parameters
    from abstractagent.tools import ALL_TOOLS

    names = set()
    for t in ALL_TOOLS:
        n = getattr(t, "_tool_definition", None)
        names.add(n.name if n is not None else getattr(t, "__name__", ""))
    assert "read_skill" not in names  # opt-in only (dead calls without a host executor)

    # The REAL attack surface (wave-E P3): the facades' default builtin
    # declarations — a facade adding READ_SKILL_TOOL unconditionally would
    # teach models a tool nothing executes. Pin each facade's defaults.
    from abstractagent.agents.react import ReactAgent
    from abstractagent.agents.codeact import CodeActAgent
    from abstractagent.agents.memact import MemActAgent

    import inspect

    for agent_cls in (ReactAgent, CodeActAgent, MemActAgent):
        src = inspect.getsource(agent_cls)
        assert "READ_SKILL_TOOL" not in src, f"{agent_cls.__name__} must not declare read_skill by default"


def test_side_effect_classifier_signals() -> None:
    assert is_side_effect_tool("write_file") is True
    assert is_side_effect_tool("read_file") is False
    assert is_side_effect_tool("mcp::github::create_issue") is True  # prefix
    # Wave-F P2: shipped-registry mutators the curated set missed — false
    # negatives re-execute the duplicate-send class the guard exists for.
    for name in (
        "fetch_url",  # never read-only-safe (model-controlled method can POST)
        "shell_exec",
        "shell_write_stdin",
        "shell_close",
        "agora_post_message",
        "agora_send_dm",
        "agora_ack_inbox",
    ):
        assert is_side_effect_tool(name) is True, name
    # Origin-ready: tags light up without code changes.
    tags = {"innocuous_name": ("mcp",), "tagged_writer": ("write",), "plain": ()}
    assert is_side_effect_tool("innocuous_name", tool_tags=tags) is True
    assert is_side_effect_tool("tagged_writer", tool_tags=tags) is True
    assert is_side_effect_tool("plain", tool_tags=tags) is False
    # tool_tags_map builds from ToolDefinitions (missing tags -> ()).
    m = tool_tags_map([_EXEC])
    assert m == {"execute_python": ()}


def test_reset_react_task_resets_per_task_and_keeps_door_state() -> None:
    old = {
        "context": {"task": "old task", "messages": [{"role": "user", "content": "old task"}, {"role": "assistant", "content": "done"}]},
        "scratchpad": {"iteration": 7, "cycles": [{"i": 1}], "review_skipped": [{"reason": "x"}], "plan": "old plan"},
        "_runtime": {"allowed_tools": ["read_file"], "skills_block": "- s1", "provider": "lmstudio", "model": "m"},
        "_temp": {"final_answer": "done", "max_iterations_announced": True},
        "_limits": {"max_iterations": 20, "current_iteration": 7, "max_tokens": 32768},
    }
    snapshot = {k: dict(v) for k, v in old.items()}

    new = reset_react_task(old, "new task", handoff_note="prior task done: done")
    # Input not mutated (the completed run's vars stay an honest record).
    assert old["scratchpad"] == snapshot["scratchpad"]
    assert old["context"]["task"] == "old task"

    assert new["context"]["task"] == "new task"
    assert new["scratchpad"] == {"iteration": 0}
    assert new["_temp"] == {}
    assert new["_limits"]["current_iteration"] == 0
    assert new["_limits"]["max_iterations"] == 20  # budget persists
    assert new["_limits"]["estimated_tokens_used"] == 0  # per-task accounting
    # Door-owned state persists in CONTENT but not by ALIAS (wave-E P2-1 —
    # the old tautological pin `is old or == old` could never fail): mutating
    # the new run's _runtime/messages must not rewrite the old record.
    # EXCEPT the steering inbox (wave-F P2): undelivered guidance addressed
    # to the OLD task must not drain into the new one as an amendment of it.
    assert new["_runtime"]["inbox"] == []
    assert {k: v for k, v in new["_runtime"].items() if k != "inbox"} == {
        k: v for k, v in old["_runtime"].items() if k != "inbox"
    }
    new["_runtime"]["allowed_tools"] = ["mutated"]
    assert old["_runtime"]["allowed_tools"] == ["read_file"]
    new["context"]["messages"][0]["content"] = "mutated"
    assert old["context"]["messages"][0]["content"] == "old task"
    # Messages carried + handoff note appended durably, timestamped.
    assert new["context"]["messages"][-1]["metadata"]["kind"] == "task_handoff"
    assert "prior task done" in new["context"]["messages"][-1]["content"]
    assert new["context"]["messages"][-1].get("timestamp")
    # Isolation mode.
    iso = reset_react_task(old, "next", carry_messages=False)
    assert iso["context"]["messages"] == []
    # Legacy budget carried when _limits lacks it (never widen silently).
    legacy = {"context": {"task": "t", "messages": []}, "scratchpad": {"iteration": 3, "max_iterations": 5}, "_limits": {}}
    out = reset_react_task(legacy, "next")
    assert out["scratchpad"]["max_iterations"] == 5


def test_one_turn_end_per_budget_exhausted_turn() -> None:
    """0028: the conclusion node re-enters (LLM dispatch -> parse); turn_end
    (raw `max_iterations`) must fire exactly once, at the completion branch;
    `max_iterations_reached` announces once at first entry."""
    from abstractagent.adapters.react_runtime import create_react_workflow
    from abstractagent.logic.react import ReActLogic

    events: List[str] = []

    def llm(run, effect, dnn):
        payload = effect.payload or {}
        # Loop replies until budget exhaustion, then the conclusion answers.
        msgs = payload.get("messages") or []
        tail = str(msgs[-1].get("content") if msgs else payload.get("prompt") or "")
        if "Max iterations reached" in tail:
            return EffectOutcome.completed({"content": "FINAL REPORT", "tool_calls": [], "finish_reason": "stop"})
        return EffectOutcome.completed(
            {"content": "", "tool_calls": [{"name": "list_files", "arguments": {}, "call_id": "c1"}], "finish_reason": "tool_calls"}
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed(
                {"mode": "executed", "results": [{"call_id": "c1", "name": "list_files", "success": True, "output": "ok", "error": None}]}
            ),
        },
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="ls", parameters={})]),
        workflow_id="wf", provider="stub", model="stub", allowed_tools=["list_files"],
        on_step=lambda step, data: events.append(step),
    )
    rid = rt.start(workflow=wf, vars={
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2},
        "_runtime": {"inbox": [], "review_mode": False},
        "_limits": {"max_iterations": 2},
    })
    for _ in range(60):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert st.status == RunStatus.COMPLETED
    assert events.count("max_iterations_reached") == 1
    assert events.count("max_iterations") == 1


def test_visit_turn_boundary_resets_announce_latch_and_review_state() -> None:
    """Wave-E adversary P1-1 (live-reproduced): the VISIT composition boundary
    (`reset_react_turn`) is where turns actually cycle — without the resets,
    turn 2 announced zero times and carried turn 1's stale review_skipped."""
    from abstractagent.adapters.react_runtime import reset_react_turn

    run_vars: Dict[str, Any] = {
        "context": {"task": "t", "messages": []},
        "scratchpad": {
            "iteration": 4,
            "review_count": 1,
            "review_skipped": [{"reason": "x", "warning": "#FALLBACK"}],
            "used_tools": True,
        },
        "_runtime": {"inbox": []},
        "_temp": {"max_iterations_announced": True, "final_answer": "a"},
        "_limits": {"max_iterations": 4, "current_iteration": 4},
    }
    reset_react_turn(run_vars)
    assert "review_skipped" not in run_vars["scratchpad"]
    assert "max_iterations_announced" not in run_vars["_temp"]
    assert run_vars["scratchpad"]["iteration"] == 0
    assert run_vars["_limits"]["current_iteration"] == 0
    # used_tools is the same per-turn latch class (wave-F P4): a toolless
    # turn 2 must not report turn 1's tool use.
    assert run_vars["scratchpad"]["used_tools"] is False


def test_sibling_budget_terminals_carry_skip_flags() -> None:
    """Wave-F P3: 'review_skipped is an always-present bool' was false at the
    CodeAct/MemAct budget terminals — a work door reading the documented shape
    got None exactly on budget-exhausted runs (type instability)."""
    import inspect

    from abstractagent.adapters import codeact_runtime, memact_runtime

    codeact_src = inspect.getsource(codeact_runtime)
    memact_src = inspect.getsource(memact_runtime)
    # Both terminals in each sibling carry the flag as a dict KEY
    # (done + max_iterations; the done terminals bind different expressions).
    assert codeact_src.count('"review_skipped":') >= 2
    assert memact_src.count('"finalize_skipped":') >= 2


def test_parse_payload_common_core_across_loops() -> None:
    """Every adapter's parse payload guarantees the common core keys."""
    from abstractagent.adapters.memact_runtime import create_memact_workflow
    from abstractagent.logic.memact import MemActLogic

    captured: Dict[str, Dict[str, Any]] = {}

    def make_sink(name: str):
        def sink(step: str, data: Dict[str, Any]) -> None:
            if step == "parse":
                captured[name] = dict(data)

        return sink

    resp = {"content": "thinking about it", "tool_calls": [{"name": "execute_python", "arguments": {"code": "1"}, "call_id": "x1"}]}

    wf_c = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]), on_step=make_sink("codeact"))
    run_c = RunState(
        run_id="rc", workflow_id="codeact_agent", status=RunStatus.RUNNING, current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1},
            "_runtime": {"inbox": [], "allowed_tools": ["execute_python"]},
            "_temp": {"llm_response": resp},
            "_limits": {"max_iterations": 20, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        },
    )
    wf_c.get_node("parse")(run_c, _Ctx())

    wf_m = create_memact_workflow(logic=MemActLogic(tools=[_EXEC]), on_step=make_sink("memact"))
    run_m = RunState(
        run_id="rm", workflow_id="memact_agent", status=RunStatus.RUNNING, current_node="parse",
        vars={
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1},
            "_runtime": {"inbox": [], "allowed_tools": ["execute_python"]},
            "_temp": {"llm_response": resp},
            "_limits": {"max_iterations": 20, "current_iteration": 1, "max_history_messages": -1, "max_tokens": 32768},
        },
    )
    wf_m.get_node("parse")(run_m, _Ctx())

    for name in ("codeact", "memact"):
        payload = captured[name]
        assert payload["has_tool_calls"] is True, name
        assert payload["tool_calls"][0]["call_id"] == "x1", name
        assert isinstance(payload["content_preview"], str) and payload["content_preview"], name
