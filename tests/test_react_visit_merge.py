"""THE MERGE PROOF: this adapter's ReAct cycle inside runtime's real visit workflow.

Consumes runtime's SHIPPED merge parameter (`build_visit_workflow(react_middle=...)`,
entity-work a78185c — the a2a 0014 ownership ruling: one owner for the visit graph in
runtime's package, the middle arriving as DATA to keep the dependency arrow one-way).
This test is the CALLER the `ReactMiddle` contract names: it builds the middle from
this package's public API and hands it over —

    react = create_react_workflow(..., final_next_node=HARVEST_NODE)
    middle = ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn)
    build_visit_workflow(home, ..., react_middle=middle)

and pins the composed graph over a REAL home + per-entity runtime:

- the G1 write direction runs LIVE through the middle: the scripted model elects a
  diary entry MID-LOOP (iteration 1, alongside a tool call) — the entity runtime's
  act-only wrapper captures it at the result boundary, words fly to the book, my
  parse accumulates the word-free metadata across iterations
  (`_temp.turn_captures`), runtime's HARVEST folds it, and the elected words rest
  NOWHERE outside the book (run vars + ledger grepped);
- runtime's ELECT/COMMIT/FORM/ANSWER/REFLECT run byte-unchanged downstream (episode
  carries `visit_id` + stamped participants; D2 access counts stay 0);
- the prelude head stays byte-identical across the middle's iterations;
- restart-mid-visit holds under the MERGED workflow (criterion 5): the hosting
  process dies between turns; a fresh runtime over the same home resumes turn 2
  through the multi-iteration cycle.

The seam is thereby tested from BOTH directions: runtime's suite pins it with a
contract-faithful stub middle (no adapter import); this suite pins it with the REAL
cycle. Skips honestly when the entity stack (abstractmemory / yaml) is absent.
"""

from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("abstractmemory")

from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn  # noqa: E402
from abstractagent.logic.react import ReActLogic  # noqa: E402
from abstractcore.tools import ToolDefinition  # noqa: E402
from abstractruntime.core.models import Effect, EffectType, RunStatus  # noqa: E402
from abstractruntime.core.runtime import EffectOutcome  # noqa: E402
from abstractruntime.core.spec import WorkflowSpec  # noqa: E402
from abstractruntime.identity.entity_runtime import open_entity_runtime  # noqa: E402
from abstractruntime.identity.visit_workflow import (  # noqa: E402
    HARVEST_NODE,
    VISITOR_WAIT_KEY,
    ReactMiddle,
    build_visit_workflow,
)

pytestmark = pytest.mark.basic


_ELECTED_WORDS = "The first merged visit happened - adapter loop and durable run together"


def _make_home(tmp_path: Path, slug: str = "mergeling") -> Path:
    from abstractmemory import (
        DEFAULT_SPARK_TEMPLATE,
        MemorySystem,
        SQLiteJournal,
        SQLiteTripleStore,
        engram,
        lint_spark,
    )

    home_dir = tmp_path / "entities" / slug
    home_dir.mkdir(parents=True)
    entity_id = f"entity:{slug}@home-test"
    spark = _copy.deepcopy(dict(DEFAULT_SPARK_TEMPLATE))
    spark["name"] = slug.capitalize()
    assert lint_spark(spark) == []
    (home_dir / "spark.yaml").write_text(yaml.safe_dump(spark, sort_keys=False), encoding="utf-8")
    (home_dir / "manifest.json").write_text(json.dumps({"entity_id": entity_id}), encoding="utf-8")
    db = home_dir / "memory.sqlite3"
    store = SQLiteTripleStore(db)
    journal = SQLiteJournal(db)
    ms = MemorySystem(store=store, journal=journal)
    assert engram(ms, spark, owner_id=entity_id).created is True
    store.close()
    journal.close()
    return home_dir


class _ScriptedLLM:
    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
        self.calls.append(json.loads(json.dumps(effect.payload or {}, default=str)))
        reply = self.replies.pop(0) if self.replies else {"content": "…"}
        return EffectOutcome.completed(dict(reply))


def _scripted_tools(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
    payload = effect.payload if isinstance(effect.payload, dict) else {}
    results = [
        {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None}
        for tc in (payload.get("tool_calls") or [])
    ]
    return EffectOutcome.completed({"mode": "executed", "results": results})


def _merged_workflow(home: Any) -> WorkflowSpec:
    """Runtime's visit workflow with my cycle as the SHIPPED react_middle
    parameter (a78185c) — this is the exact call shape production callers use."""
    react = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="entity-visit-merged",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
        final_next_node=HARVEST_NODE,
    )
    middle = ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn)
    return build_visit_workflow(
        home,
        participants=["person:albou"],
        idle_seconds=3600,
        model_info={"provider": "test", "model": "scripted"},
        visit_id="visit-merge01",
        react_middle=middle,
    )


def _open(home_dir: Path, llm: _ScriptedLLM) -> Tuple[Any, WorkflowSpec]:
    ert = open_entity_runtime(
        home_dir,
        extra_handlers={EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: _scripted_tools},
    )
    return ert, _merged_workflow(ert.home)


_TURN1_ELECT_AND_TOOL = {
    "content": (
        "Let me hold this moment and look around.\n"
        "```diary kind=note\n"
        f"gist: merged visit note\n{_ELECTED_WORDS}\n"
        "```"
    ),
    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c1"}],
}
_TURN1_ANSWER = {"content": "It is good to be here - the loop and the run are one now.", "tool_calls": []}
_REFLECTION = {"content": "Looking back: the merged turn held. Goodbye.", "tool_calls": []}


def test_merged_visit_full_lifecycle_with_midloop_election(tmp_path: Path) -> None:
    from abstractmemory import TripleQuery

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM([_TURN1_ELECT_AND_TOOL, _TURN1_ANSWER, _REFLECTION])
    ert, wf = _open(home_dir, llm)
    try:
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-merge")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING

        # ONE visitor turn that exercises the whole merged cycle: mid-loop diary
        # election + tool call (iteration 1), then the answer (iteration 2).
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Hello - keep a note of this merged moment.", "speaker": "person:albou"},
            max_steps=200,
        )
        assert state.status == RunStatus.WAITING  # parked again after answering

        # Close -> reflection -> done (runtime's nodes, untouched).
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=300,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 1
        assert state.output["close_reason"] == "closed"

        # My cycle really ran multi-iteration: 2 turn calls + 1 reflection call.
        assert len(llm.calls) == 3
        # The turn's LLM payloads carried the seam facts my adapter passes through.
        assert llm.calls[0].get("turn_id") == "t-0001"
        assert "anchor_record_ids" in llm.calls[0]
        # Head discipline under the merge: byte-identical prelude head, both iterations.
        assert llm.calls[0].get("system_prompt") == llm.calls[1].get("system_prompt")
        assert "MY PERSONA" not in str(llm.calls[0].get("system_prompt"))

        # G1 write direction, LIVE through my loop: the elected words rest ONLY in
        # the book — not in run vars (transcript, scratchpad, node_traces), not in
        # the ledger.
        entries = ert.home.diary.list_entries()
        assert any(_ELECTED_WORDS in (e.get("text") or "") for e in entries)
        final_state = ert.runtime.get_state(run_id)
        assert _ELECTED_WORDS not in json.dumps(final_state.vars, default=str)
        ledger = ert.runtime.get_ledger(run_id)
        assert _ELECTED_WORDS not in json.dumps(ledger, default=str)

        # The episode formed with the door facts (participants + item-14 key).
        rows = ert.home.ms.query(TripleQuery(scope="life", owner_id=ert.entity_id, limit=0))
        episode_attrs = [
            a.attributes for a in rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "episode"
        ]
        assert len(episode_attrs) == 1
        assert episode_attrs[0].get("visit_id") == "visit-merge01"
        assert episode_attrs[0].get("participants") == ["person:albou", ert.entity_id]

        # The visitor got the ANSWER (my cycle's final answer through runtime's node).
        answers = [
            ((r.get("effect") or {}).get("payload") or {}).get("message")
            for r in ledger
            if isinstance(r, dict) and ((r.get("effect") or {}).get("type")) == "answer_user"
        ]
        assert any("the loop and the run are one now" in (m or "") for m in answers)

        # D2 through the merged workflow: presence is not use.
        self_rows = ert.home.ms.query(TripleQuery(scope="self", owner_id=ert.entity_id, limit=0))
        value_ids = [
            str(a.subject) for a in self_rows
            if isinstance(a.attributes, dict) and a.attributes.get("record_kind") == "value"
        ]
        assert value_ids
        counts = ert.home.ms.access_counts(record_ids=value_ids)
        recs = counts.get("records", counts)
        assert all(int(v) == 0 for v in recs.values()), f"identity strengthened: {recs}"
    finally:
        ert.close()


def test_merged_visit_survives_restart_between_turns(tmp_path: Path) -> None:
    """Criterion 5 under the MERGE: process dies between turns; a fresh runtime
    over the same home resumes the visit through my multi-iteration cycle."""
    home_dir = _make_home(tmp_path, slug="mergerevive")
    llm1 = _ScriptedLLM([
        {"content": "First merged reply before the crash.", "tool_calls": []},
    ])
    ert1, wf1 = _open(home_dir, llm1)
    run_id = ert1.runtime.start(workflow=wf1, vars={}, session_id="visit-merge")
    ert1.runtime.tick(workflow=wf1, run_id=run_id, max_steps=50)
    state = ert1.runtime.resume(
        workflow=wf1, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
        payload={"text": "Hello before the crash.", "speaker": "person:albou"}, max_steps=200,
    )
    assert state.status == RunStatus.WAITING
    ert1.close()  # the restart: everything in-process is gone

    llm2 = _ScriptedLLM([
        {
            "content": "I remember the start - checking the room again.",
            "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "c9"}],
        },
        {"content": "Still here after the restart.", "tool_calls": []},
        {"content": "Reflection after the merged restart.", "tool_calls": []},
    ])
    ert2, wf2 = _open(home_dir, llm2)
    try:
        state = ert2.runtime.resume(
            workflow=wf2, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Do you remember the start?", "speaker": "person:albou"}, max_steps=200,
        )
        assert state.status == RunStatus.WAITING
        # Turn 2 ran MULTI-ITERATION through my cycle in the fresh process, and its
        # payload carried turn 1's transcript from run vars (durable continuity).
        assert len(llm2.calls) == 2
        turn2_msgs = json.dumps(llm2.calls[0].get("messages") or [])
        assert "First merged reply before the crash." in turn2_msgs
        assert "Hello before the crash." in turn2_msgs

        state = ert2.runtime.resume(
            workflow=wf2, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"kind": "close"}, max_steps=300,
        )
        assert state.status == RunStatus.COMPLETED
        assert state.output["turns"] == 2
    finally:
        ert2.close()
