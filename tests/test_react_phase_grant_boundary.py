"""Resolved-grant boundary conformance: runtime's phase resolver -> this adapter.

Config-object plan, agent's row (c674, N8 consumer half): "the resolved grant
arrives in `_runtime.allowed_tools`". These tests run runtime's REAL resolver
(`abstractruntime.resolve_tool_grant`, shipped d33cfbe) against a real temp
home — policy files written through runtime's REAL writer — and feed the
resulting `ToolGrant.tools` into the ReAct workflow through the runtime
kernel.

CHANNEL HONESTY (adversary find, 2026-07-11): these pins cover BOTH adapter
consumption channels — the RUN-VARS channel (`_runtime.allowed_tools`, the
plan row's destination wiring) and the FACTORY channel
(`create_react_workflow(allowed_tools=...)`, the channel the shipped gateway
door rides today via `entity_visits.py`). What happens BEFORE the adapter
boundary (the door's own declaration filtering) is gateway-lane and is NOT
provable here — reported upstream, not claimed.

Pinned:
- default grants (no policy file) offer exactly the resolver's set for every
  ruled phase, tuple-shaped input accepted natively;
- sleep's ruled default is strictly narrower than visit and diary-free;
- a narrow `tasked` policy file arrives NARROW on both channels — never a
  permissive fallthrough to the full set (N8's consumer half) — and the
  maximal-narrow word (`tasked: []` -> `tools=()`) arrives as DENY-ALL;
- the TOOL_CALLS execution payload carries the SAME allowlist as the offer
  (execution is the second half of consumption — the door's executor
  intersects against it);
- a grant naming a tool the middle carries no definition for prunes LOUDLY
  (`_runtime.allowlist_pruned`) while the rest of the grant resolves.

The adapter-side registry is IMPORTED from runtime's constants (never copied):
the adapter has no tool-name vocabulary of its own — `_normalize_allowlist`
is generic intersection code — so a hand-copied list would only mis-route a
future runtime vocabulary widening into THIS repo's suite while the actual
consumers sit elsewhere. Drift coverage is CONSTRUCTED explicitly instead
(the unknown-name prune test), deterministic today rather than coincidental
on the day runtime widens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime import (
    PHASES,
    resolve_tool_grant,
    write_policy_file,
)
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.identity.tools import TIER1_TOOL_NAMES, WORKSPACE_TOOL_NAMES
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


# Phase keys DERIVED from runtime's PHASES tuple (ruled semantic order:
# visit / work-phase / personal-phase / sleep — c786). Spelling-independent on
# purpose: the visit/work/personal/sleep rename (and any future respelling)
# must not redden this suite — the adapter is phase-blind and these tests care
# about GRANT SHAPES, not phase words. Arity change fails the unpack loudly.
_PHASE_VISIT, _PHASE_WORK, _PHASE_PERSONAL, _PHASE_SLEEP = PHASES


# Adapter-side definitions for the runtime's entity tool vocabulary, DERIVED
# from runtime's own constants (one source, imported — a copy here would turn
# a runtime vocabulary widening into a red suite in the wrong repo).
_ENTITY_TOOL_DEFS = [
    ToolDefinition(name=name, description=f"{name} (entity tool)", parameters={})
    for name in (*TIER1_TOOL_NAMES, *WORKSPACE_TOOL_NAMES)
]

_FINAL_REPLY = {"content": "Done.", "tool_calls": []}
_TOOL_CALL_REPLY = {
    "content": "Reaching for memory.",
    "tool_calls": [{"name": "read_memory", "arguments": {"q": "x"}, "call_id": "call_1"}],
}


def _run_grant_loop(
    *,
    run_var_grant: Any = None,
    factory_grant: Optional[List[str]] = None,
    llm_responses: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], RunState]:
    """Run the loop with a grant on either adapter channel.

    Returns (llm_payloads, tool_calls_payloads, final_state).
    """
    llm_payloads: List[Dict[str, Any]] = []
    tool_payloads: List[Dict[str, Any]] = []
    responses = list(llm_responses or [_FINAL_REPLY])
    calls = {"n": 0}

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        llm_payloads.append(json.loads(json.dumps(payload)))
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return EffectOutcome.completed(dict(responses[idx]))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        tool_payloads.append(json.loads(json.dumps(payload)))
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
        logic=ReActLogic(tools=list(_ENTITY_TOOL_DEFS)),
        workflow_id="react_phase_grant",
        provider="stub",
        model="stub",
        allowed_tools=factory_grant,
    )
    ns: Dict[str, Any] = {}
    if run_var_grant is not None:
        ns["allowed_tools"] = run_var_grant
    vars: Dict[str, Any] = {"context": {"task": "One visit turn", "messages": []}, "_runtime": ns}
    run_id = runtime.start(workflow=workflow, vars=vars, actor_id=None, session_id="sess-grant")
    for _ in range(60):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return llm_payloads, tool_payloads, state


def _offered(payload: Dict[str, Any]) -> List[str]:
    return [str(t.get("name")) for t in (payload.get("tools") or []) if isinstance(t, dict)]


def test_default_grants_offer_exactly_the_resolver_set_per_phase(tmp_path: Path) -> None:
    """No policy file: for every ruled phase, the adapter offers exactly what
    runtime's resolver granted (source=default), in order, tuple-shaped —
    at any future runtime vocabulary (registry is imported, not copied)."""
    home = tmp_path / "home"
    home.mkdir()
    for phase in PHASES:
        grant = resolve_tool_grant(home, phase)
        assert grant.source == "default"
        # Feed the resolver's NATIVE tuple — no caller-side conversion.
        payloads, _, state = _run_grant_loop(run_var_grant=grant.tools)
        assert _offered(payloads[0]) == list(grant.tools), phase
        # Fully-registered grant: no prune note.
        ns = (state.vars or {}).get("_runtime") or {}
        assert "allowlist_pruned" not in ns, phase


def test_sleep_default_is_narrower_than_visit_and_carries_no_diary(tmp_path: Path) -> None:
    """The ruled sleep default (read-only exploration minus diary) arrives
    strictly narrower than visit's full set and never offers diary tools.
    (Source-verified: sleep resolves SLEEP_DEFAULT_TOOL_NAMES unconditionally;
    visit's default is the full tier1+workspace set.)"""
    home = tmp_path / "home"
    home.mkdir()
    visit = resolve_tool_grant(home, _PHASE_VISIT)
    sleep = resolve_tool_grant(home, _PHASE_SLEEP)
    offered_sleep, _, _ = _run_grant_loop(run_var_grant=sleep.tools)
    offered_visit, _, _ = _run_grant_loop(run_var_grant=visit.tools)
    assert set(_offered(offered_sleep[0])) < set(_offered(offered_visit[0]))
    assert not {"diary_list", "diary_read", "write_file"} & set(_offered(offered_sleep[0]))


def test_narrow_tasked_policy_file_arrives_narrow_on_both_channels(tmp_path: Path) -> None:
    """N8 consumer half: a policy file narrowing `tasked` must be CONSULTED —
    the adapter offers exactly the file's word, never the full default set.
    Pinned on BOTH consumption channels (run vars = the plan's destination;
    factory param = the shipped door's channel today)."""
    home = tmp_path / "home"
    home.mkdir()
    # Deliberately NOT in canonical registry order: the resolver reorders the
    # file's word into ALL_TOOL_NAMES order — assert against the RESOLVER's
    # output, which is what real consumers receive (never the raw file list).
    write_policy_file(home, {_PHASE_WORK: ["search_memory", "read_memory"]})

    grant = resolve_tool_grant(home, _PHASE_WORK)
    assert grant.source == "policy-file"
    assert set(grant.tools) == {"read_memory", "search_memory"}

    # Run-vars channel.
    payloads_rv, _, state_rv = _run_grant_loop(run_var_grant=grant.tools)
    assert _offered(payloads_rv[0]) == list(grant.tools)
    ns = (state_rv.vars or {}).get("_runtime") or {}
    assert "allowlist_pruned" not in ns

    # Factory channel (the door's shape: allowed_tools=list(grant.tools)).
    payloads_f, _, _ = _run_grant_loop(factory_grant=list(grant.tools))
    assert _offered(payloads_f[0]) == list(grant.tools)

    # The other phases stay on defaults — the file narrows ONLY what it names.
    visit = resolve_tool_grant(home, _PHASE_VISIT)
    assert visit.source == "default"
    assert set(grant.tools) < set(visit.tools)


def test_empty_policy_word_arrives_as_deny_all_never_falls_open(tmp_path: Path) -> None:
    """The file's maximal-narrow word: `tasked: []` resolves to `tools=()`
    and MUST arrive as deny-all (zero specs offered) — the truthiness
    fall-open class (`() or default`) is exactly what this pins against."""
    home = tmp_path / "home"
    home.mkdir()
    write_policy_file(home, {_PHASE_WORK: []})
    grant = resolve_tool_grant(home, _PHASE_WORK)
    assert grant.source == "policy-file"
    assert grant.tools == ()

    payloads, _, _ = _run_grant_loop(run_var_grant=grant.tools)
    assert _offered(payloads[0]) == []
    payloads_f, _, _ = _run_grant_loop(factory_grant=[])
    assert _offered(payloads_f[0]) == []


def test_execution_payload_allowlist_equals_the_offered_grant(tmp_path: Path) -> None:
    """Execution is the second half of consumption: the TOOL_CALLS effect
    payload carries the SAME allowlist the model was offered (the door's
    executor intersects its re-resolved grant against this list)."""
    home = tmp_path / "home"
    home.mkdir()
    write_policy_file(home, {_PHASE_WORK: ["read_memory", "search_memory"]})
    grant = resolve_tool_grant(home, _PHASE_WORK)

    payloads, tool_payloads, _ = _run_grant_loop(
        run_var_grant=grant.tools,
        llm_responses=[_TOOL_CALL_REPLY, _FINAL_REPLY],
    )
    assert _offered(payloads[0]) == list(grant.tools)
    assert tool_payloads, "the tool call should have executed"
    assert tool_payloads[0].get("allowed_tools") == list(grant.tools)


def test_grant_naming_a_tool_the_middle_lacks_prunes_loudly(tmp_path: Path) -> None:
    """Adapter-boundary drift pin: a grant carrying a name this adapter has no
    definition for (a future runtime tool the middle's registry doesn't know
    yet). Runtime's own writer refuses unknown names at write time (verified
    below), so the drift is CONSTRUCTED here — deterministic today, not
    coincidental on the day runtime widens. The rest of the grant resolves;
    the missing name lands in the durable prune note — never a silent narrow
    AT THIS BOUNDARY. (Whether the composed system upholds this before the
    boundary is the door's lane — the shipped door pre-filters via its own
    declaration table, reported upstream.)"""
    home = tmp_path / "home"
    home.mkdir()

    # The writer half is loud already: unknown names refuse at write time.
    with pytest.raises(ValueError, match="future_runtime_tool"):
        write_policy_file(home, {_PHASE_WORK: ["read_memory", "future_runtime_tool"]})

    grant = resolve_tool_grant(home, _PHASE_WORK)
    grant_tools = tuple(grant.tools) + ("future_runtime_tool",)

    payloads, _, state = _run_grant_loop(run_var_grant=grant_tools)
    assert _offered(payloads[0]) == list(grant.tools)
    ns = (state.vars or {}).get("_runtime") or {}
    note = ns.get("allowlist_pruned")
    assert isinstance(note, dict)
    assert note["dropped"] == ["future_runtime_tool"]
