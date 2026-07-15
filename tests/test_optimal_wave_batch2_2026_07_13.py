"""Batch 2 of the operator-directed improvement wave (2026-07-13).

Pins for the promoted 0030 residue:
- skills_block: a NAMED slot beside system_prompt_extra (delegate children
  overwrite the extra — a skills attachment must survive delegation framing).
- Delegate substrate palette: host-gated named profiles; unknown names fail
  as TOOL errors; raw provider/model passthrough is structurally impossible.
- Default provider/model resolution: explicit wins; config pair when both
  unset; packaged fallback pair is LOUD (#FALLBACK UserWarning).
- Output-cap honesty: ReactAgent's explicit cap survives init_node's
  full-context nulling via the _runtime channel; sibling facades wire
  max_output_tokens to _limits.
- Conclusion cache honesty (B-F7): the max-iterations directive rides the
  trailing message, never the system prompt (pinned in
  test_react_max_iterations_concludes.py's double).
- Unattended recipe: packaged two-move helper.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractagent.adapters.codeact_runtime import create_codeact_workflow
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.agents.defaults import (
    FALLBACK_MODEL,
    FALLBACK_PROVIDER,
    resolve_provider_model,
)
from abstractagent.agents.unattended import (
    UNATTENDED_DIRECTIVE,
    unattended_allowlist,
    unattended_runtime_overrides,
)
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


def _codeact_vars(runtime_extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "context": {"task": "do the thing", "messages": []},
        "scratchpad": {"iteration": 0},
        "_runtime": dict({"inbox": [], "allowed_tools": ["execute_python"]}, **(runtime_extra or {})),
        "_temp": {},
        "_limits": {"max_iterations": 20, "current_iteration": 0, "max_history_messages": -1, "max_tokens": 32768},
    }


def _payload(wf, node: str, vars: Dict[str, Any]):
    run = RunState(run_id="r", workflow_id="wf", status=RunStatus.RUNNING, current_node=node, vars=vars)
    plan = wf.get_node(node)(run, _Ctx())
    return plan, (plan.effect.payload if plan.effect is not None else {})


def test_skills_block_is_a_separate_slot_and_survives_delegation_framing() -> None:
    """skills_block and system_prompt_extra compose independently — the
    delegation directive (which OVERWRITES system_prompt_extra in children)
    can no longer erase a host-attached skills block."""
    wf = create_codeact_workflow(logic=CodeActLogic(tools=[_EXEC]))
    _, p = _payload(
        wf, "reason",
        _codeact_vars({"skills_block": "- pdf-tools: extract text", "system_prompt_extra": "You are a delegated sub-agent."}),
    )
    sys = str(p.get("system_prompt") or "")
    skills_at = sys.find("Available skills:")
    extra_at = sys.find("Additional system instructions:")
    assert skills_at != -1 and extra_at != -1
    assert skills_at < extra_at  # fixed slot order
    assert "pdf-tools" in sys


def test_delegate_substrate_palette_resolves_and_refuses() -> None:
    wf = create_react_workflow(
        logic=ReActLogic(tools=[DELEGATE_AGENT_TOOL, ToolDefinition(name="list_files", description="ls", parameters={})]),
        workflow_id="react_agent", provider="stub", model="stub",
        allowed_tools=["list_files", "delegate_agent"],
    )

    def _act_vars(substrate: str, palette: Any) -> Dict[str, Any]:
        rt: Dict[str, Any] = {"inbox": [], "allowed_tools": ["list_files", "delegate_agent"]}
        if palette is not None:
            rt["delegate_substrates"] = palette
        return {
            "context": {"task": "t", "messages": []},
            "scratchpad": {"iteration": 1, "max_iterations": 20, "cycles": []},
            "_runtime": rt,
            "_temp": {"pending_tool_calls": [{"name": "delegate_agent", "arguments": {"task": "sub", "substrate": substrate}, "call_id": "d1"}]},
            "_limits": {"max_iterations": 20},
        }

    # Known profile -> child rides the palette's provider/model.
    plan, p = _payload(wf, "act", _act_vars("strong", {"strong": {"provider": "lmstudio", "model": "big-model"}}))
    assert plan.effect is not None
    child_rt = p["vars"]["_runtime"]
    assert child_rt["provider"] == "lmstudio" and child_rt["model"] == "big-model"
    # The palette itself must NOT propagate (each level needs its own grant).
    assert "delegate_substrates" not in child_rt

    # Unknown profile -> loud TOOL error, no subworkflow effect.
    run = RunState(
        run_id="r2", workflow_id="react_agent", status=RunStatus.RUNNING, current_node="act",
        vars=_act_vars("nope", {"strong": {"provider": "lmstudio", "model": "big-model"}}),
    )
    plan2 = wf.get_node("act")(run, _Ctx())
    assert plan2.effect is None
    results = run.vars["_temp"]["tool_results"]["results"]
    assert results[0]["success"] is False
    assert "Unknown delegate substrate 'nope'" in results[0]["error"]
    assert "strong" in results[0]["error"]  # names what IS available

    # No palette granted at all -> refusal says so.
    run3 = RunState(
        run_id="r3", workflow_id="react_agent", status=RunStatus.RUNNING, current_node="act",
        vars=_act_vars("strong", None),
    )
    wf.get_node("act")(run3, _Ctx())
    err = run3.vars["_temp"]["tool_results"]["results"][0]["error"]
    assert "No substrate palette was granted" in err


def test_resolve_provider_model_explicit_wins_and_fallback_is_loud() -> None:
    assert resolve_provider_model("lmstudio", "m1") == ("lmstudio", "m1")

    import abstractagent.agents.defaults as d

    # Config pair configured -> used silently when both args missing.
    orig = d._configured_global_defaults
    d._configured_global_defaults = lambda: ("lmstudio", "cfg-model")  # type: ignore[assignment]
    try:
        assert resolve_provider_model(None, None) == ("lmstudio", "cfg-model")
        # provider matching config -> config model fills the gap silently.
        assert resolve_provider_model("lmstudio", None) == ("lmstudio", "cfg-model")
    finally:
        d._configured_global_defaults = orig  # type: ignore[assignment]

    # Nothing configured -> packaged pair with a #FALLBACK warning.
    d._configured_global_defaults = lambda: (None, None)  # type: ignore[assignment]
    try:
        with pytest.warns(UserWarning, match="#FALLBACK"):
            assert resolve_provider_model(None, None) == (FALLBACK_PROVIDER, FALLBACK_MODEL)
        # Packaged provider without a model -> packaged model, loud.
        with pytest.warns(UserWarning, match="#FALLBACK"):
            assert resolve_provider_model("ollama", None) == (FALLBACK_PROVIDER, FALLBACK_MODEL)
        # Foreign provider without a model -> REFUSE, never a wrong-shaped
        # guess (design adversary P1 2026-07-13: an Ollama tag shipped to
        # lmstudio is a guaranteed 404 with a worse provider-side error).
        with pytest.raises(ValueError, match="refusing to guess"):
            resolve_provider_model("lmstudio", None)
        # Model without provider (no matching config) -> REFUSE.
        with pytest.raises(ValueError, match="does not identify its server"):
            resolve_provider_model(None, "some-model")
    finally:
        d._configured_global_defaults = orig  # type: ignore[assignment]


def test_react_explicit_output_cap_survives_init_full_context_nulling() -> None:
    """B-F9/0029 #9: ReactAgent(max_tokens=4096) was silently dropped by
    init_node's caps nulling. The explicit cap now rides _runtime and reaches
    the LLM params."""
    from abstractruntime.core.models import Effect, EffectType
    from abstractruntime.core.runtime import EffectOutcome, Runtime
    from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

    from abstractagent.agents.react import ReactAgent

    seen_params: list[Dict[str, Any]] = []

    def llm(run, effect, dnn):
        seen_params.append(dict((effect.payload or {}).get("params") or {}))
        return EffectOutcome.completed({"content": "done", "tool_calls": []})

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm,
            EffectType.TOOL_CALLS: lambda r, e, d: EffectOutcome.completed({"mode": "executed", "results": []}),
        },
    )
    agent = ReactAgent(runtime=rt, tools=[], max_tokens=4096, review_mode=False)
    agent.start("say done")
    for _ in range(20):
        state = agent.step()
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert seen_params and seen_params[0].get("max_tokens") == 4096


def test_unattended_recipe_two_moves() -> None:
    assert unattended_allowlist(["read_file", "ask_user", "write_file"]) == ["read_file", "write_file"]
    assert unattended_allowlist(None) is None  # never invents a registry
    ov = unattended_runtime_overrides(["read_file", "ask_user"])
    assert ov["allowed_tools"] == ["read_file"]
    assert ov["system_prompt_extra"] == UNATTENDED_DIRECTIVE
    assert "Do not ask the user questions" in UNATTENDED_DIRECTIVE
    ov2 = unattended_runtime_overrides(None)
    assert "allowed_tools" not in ov2
