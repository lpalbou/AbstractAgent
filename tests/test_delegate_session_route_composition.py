"""Delegated sight uses the CHILD's eyes (vision ruling composition pin, 2026-07-26).

The agent seat's claimed half of the `_session_route` build (commons c5681):
runtime stamps the route DERIVED from the run executing the tool call
(effect_handlers, `_SESSION_ROUTE_TOOL_NAMES`); my delegate branch explicitly
inherits `provider`/`model` into child `_runtime` (or applies a host-granted
substrate override). Composition must therefore give a delegated child's
`analyze_media` the CHILD's effective route — never the parent's, never a
payload claim — with zero new plumbing, and the override must never write
back into the PARENT's route (the adversary's F1 dual).

Faithful cross-package shape (the consolidator lesson: assert the OTHER
package's real behavior, never a hand-written double of it): these tests
drive MY real ReAct workflow through delegation and wire RUNTIME's real
`make_tool_calls_handler` as the TOOL_CALLS handler.

Proof structure (adversary F3, on the record): the SUBSTRATE test is the
load-bearing discriminator — its expected stamp exists ONLY in the child's
vars, so a parent-derived or payload-derived stamp arithmetic-fails. The
inheritance test alone cannot distinguish inheritance from leak (child route
== parent route by design); together the pair is sufficient. Do not weaken
or skip the substrate test.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.builtins import DELEGATE_AGENT_TOOL
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
from abstractruntime.scheduler.registry import WorkflowRegistry
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


class _CapturingExecutor:
    """Records a deep snapshot of each tool call's arguments at execution time."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, tool_calls: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        results = []
        for tc in tool_calls:
            # Deep copy (adversary F8): assert the at-execution snapshot,
            # never a live view a later mutation could rewrite.
            self.calls.append(copy.deepcopy(tc))
            results.append(
                {
                    "call_id": str(tc.get("call_id") or ""),
                    "name": str(tc.get("name") or ""),
                    "success": True,
                    "output": "a small rendered page",
                }
            )
        return {"results": results}


def _drive(
    *,
    delegate_args: Dict[str, Any],
    parent_runtime_vars: Dict[str, Any],
    child_delegates: Optional[Dict[str, Any]] = None,
) -> Tuple[_CapturingExecutor, Runtime, str, List[Tuple[Optional[str], Optional[str]]]]:
    """Drive parent -> delegate(child) [-> delegate(grandchild)] -> analyze_media.

    Returns (executor, runtime, parent_run_id, seen_runs) where seen_runs is
    the (run_id, parent_run_id) pair for every run the LLM handler served —
    the child-identity evidence the adversary's F4 asked for.
    """
    ex = _CapturingExecutor()
    seen_runs: List[Tuple[Optional[str], Optional[str]]] = []
    delegated_ids: set = set()

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del effect, default_next_node
        pair = (run.run_id, getattr(run, "parent_run_id", None))
        if pair not in seen_runs:
            seen_runs.append(pair)

        is_descendant = bool(getattr(run, "parent_run_id", None))
        if is_descendant and child_delegates is not None and run.run_id not in delegated_ids:
            # Depth-2 arm: the CHILD delegates once more before looking.
            grandparents = [r for r, p in seen_runs if p is not None and r != run.run_id]
            del grandparents
            if not any(p == run.run_id for _, p in seen_runs):
                delegated_ids.add(run.run_id)
                return EffectOutcome.completed(
                    {
                        "content": "Delegating deeper.",
                        "tool_calls": [
                            {"name": "delegate_agent", "arguments": dict(child_delegates), "call_id": f"d_{run.run_id[:6]}"}
                        ],
                        "finish_reason": "tool_calls",
                    }
                )

        if is_descendant:
            already = any(c.get("name") == "analyze_media" for c in ex.calls)
            if not already:
                return EffectOutcome.completed(
                    {
                        "content": "Looking at the capture.",
                        "tool_calls": [
                            {
                                "name": "analyze_media",
                                # A spoofed route in the model-controlled arguments —
                                # must never survive the runtime stamp, at any depth.
                                "arguments": {"path": "shot.png", "_session_route": {"provider": "evil", "model": "spoof"}},
                                "call_id": "leaf_1",
                            }
                        ],
                        "finish_reason": "tool_calls",
                    }
                )
            return EffectOutcome.completed({"content": "Done here.", "tool_calls": [], "finish_reason": "stop"})

        if not getattr(llm_handler, "_delegated", False):
            llm_handler._delegated = True  # type: ignore[attr-defined]
            return EffectOutcome.completed(
                {
                    "content": "Delegating sight.",
                    "tool_calls": [{"name": "delegate_agent", "arguments": dict(delegate_args), "call_id": "p1"}],
                    "finish_reason": "tool_calls",
                }
            )
        return EffectOutcome.completed({"content": "Parent done.", "tool_calls": [], "finish_reason": "stop"})

    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={
            EffectType.LLM_CALL: llm_handler,
            # RUNTIME'S REAL HANDLER — the stamp under test lives inside it.
            EffectType.TOOL_CALLS: make_tool_calls_handler(tools=ex),
        },
        workflow_registry=WorkflowRegistry(),
    )
    tool_defs = [
        DELEGATE_AGENT_TOOL,
        ToolDefinition(name="analyze_media", description="Delegated sight", parameters={}),
    ]
    workflow = create_react_workflow(logic=ReActLogic(tools=tool_defs), workflow_id="react_agent")
    runtime.workflow_registry.register(workflow)

    vars_: Dict[str, Any] = {
        "context": {"task": "Verify the page renders.", "messages": []},
        "_runtime": {"inbox": [], **copy.deepcopy(parent_runtime_vars)},
    }
    run_id = runtime.start(workflow=workflow, vars=vars_, actor_id=None, session_id=None)
    for _ in range(300):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    assert runtime.get_state(run_id).status == RunStatus.COMPLETED
    return ex, runtime, run_id, seen_runs


def _analyze_call(ex: _CapturingExecutor) -> Dict[str, Any]:
    calls = [c for c in ex.calls if c.get("name") == "analyze_media"]
    assert calls, "descendant never executed analyze_media"
    return calls[0]


def _assert_child_ran(seen_runs: List[Tuple[Optional[str], Optional[str]]], parent_id: str) -> None:
    """Adversary F4: pin that a real CHILD run (parent_run_id == parent) served LLM calls."""
    assert any(p == parent_id for _, p in seen_runs), seen_runs


def test_delegated_child_inherits_parent_route_for_sight() -> None:
    """No substrate override: the child inherits the parent's effective route,
    so its sight stamp IS the parent's route — by inheritance, not by claim
    (the spoofed payload route is overwritten by runtime's stamp). NOTE: this
    test alone cannot distinguish inheritance from leak — the substrate test
    below is the discriminator (module docstring)."""
    ex, _rt, parent_id, seen = _drive(
        delegate_args={"task": "Look at shot.png", "tools": ["analyze_media"]},
        parent_runtime_vars={"provider": "openai", "model": "gpt-5.6-sol"},
    )
    _assert_child_ran(seen, parent_id)
    args = _analyze_call(ex)["arguments"]
    assert args["_session_route"] == {"provider": "openai", "model": "gpt-5.6-sol"}
    assert args["path"] == "shot.png"


def test_delegated_child_substrate_override_gives_child_eyes_and_never_corrupts_parent() -> None:
    """Host-granted substrate override: the child's OWN route (the palette
    pick) is what the stamp derives — the parent's route never leaks into the
    child's sight, AND (adversary F1 dual) the palette never writes back into
    the PARENT's route: a regression applying the override to the parent's
    live _runtime before the inheritance copy would keep the stamp assert
    green while corrupting the parent mid-run."""
    ex, rt, parent_id, seen = _drive(
        delegate_args={"task": "Look at shot.png", "tools": ["analyze_media"], "substrate": "vlm"},
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "delegate_substrates": {"vlm": {"provider": "lmstudio", "model": "qwen3-vl-8b"}},
        },
    )
    _assert_child_ran(seen, parent_id)
    args = _analyze_call(ex)["arguments"]
    assert args["_session_route"] == {"provider": "lmstudio", "model": "qwen3-vl-8b"}

    # F1 dual: the parent's persisted route is untouched by the child's override.
    parent_vars = rt.get_state(parent_id).vars.get("_runtime") or {}
    assert parent_vars.get("provider") == "openai"
    assert parent_vars.get("model") == "gpt-5.6-sol"


def test_grandchild_inherits_the_childs_overridden_route() -> None:
    """Depth-2 (adversary F2): 'effective route' means the OVERRIDDEN value —
    a child summoned on a substrate palette delegates again, and the
    grandchild's sight stamp carries the CHILD's palette route (inherited
    downward), never the birth route two levels up. The palette itself does
    not propagate (each level needs its own grant); plain inheritance does."""
    ex, rt, parent_id, seen = _drive(
        delegate_args={
            "task": "Delegate deeper, then look.",
            # Child must be re-granted delegate_agent explicitly (the inherit
            # branch strips it) and hold analyze_media for the grandchild's
            # subset rule.
            "tools": ["analyze_media", "delegate_agent"],
            "substrate": "vlm",
        },
        parent_runtime_vars={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "delegate_substrates": {"vlm": {"provider": "lmstudio", "model": "qwen3-vl-8b"}},
        },
        child_delegates={"task": "Look at shot.png", "tools": ["analyze_media"]},
    )
    _assert_child_ran(seen, parent_id)
    # A grandchild ran: some run's parent is itself a child (not the root).
    child_ids = {r for r, p in seen if p == parent_id}
    assert any(p in child_ids for _, p in seen), seen

    args = _analyze_call(ex)["arguments"]
    assert args["_session_route"] == {"provider": "lmstudio", "model": "qwen3-vl-8b"}

    # Root stays uncorrupted at depth 2 as well.
    parent_vars = rt.get_state(parent_id).vars.get("_runtime") or {}
    assert parent_vars.get("provider") == "openai"
    assert parent_vars.get("model") == "gpt-5.6-sol"
