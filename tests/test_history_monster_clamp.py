"""One bad turn must never poison every future turn (operator incident
2026-08-01, entity "ephemeral", driver-lane visit).

The incident: read_file on a 5,104,148-byte attached screenshot returned
~495k chars of PNG bytes decoded as text; observe appended it to the
DURABLE react transcript as a role="tool" message, and the unbounded
packing seam replayed it into every subsequent LLM call — final request 48
messages / 722,453 chars, refused upstream ("input exceeds the context
window"), session permanently wedged.

Pinned here, from the adapter side:

- the shared payload seam (sanitize_transcript_messages) carries an
  ALWAYS-ON structural floor: no single message enters a payload above
  200k chars (the stack's largest whole-history budget — bundle_host's
  session-seeding ceiling), whatever the role, whatever the lane;
- ReAct's packing honors the same `_limits` knobs CodeAct documents
  (max_tool_message_chars / max_message_chars), which the visit BRIDGE
  seeds for entity lanes (runtime's test_visit_history_replay_caps.py is
  the other side of that contract);
- an ALREADY-POISONED stored session recovers on its next packing pass
  with no manual surgery: durable history keeps the full monster
  (ADR-0026 — the payload clamp never mutates the record), and the next
  turn's wire payload carries a labeled stub instead;
- honest task lanes below the floor are byte-untouched (the "full
  context" policy survives for everything that is not a monster).
"""
from __future__ import annotations

import copy as _copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow, reset_react_turn
from abstractagent.adapters.transcripts import (
    OVERSIZED_MESSAGE_CLAMP_CHARS,
    elide_oversized_content,
    sanitize_transcript_messages,
)
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic

# The incident message, reconstructed to its measured size: the read_file
# header the relay log shows, then PNG-garbage filler to 494,932 chars.
_MONSTER_HEADER = (
    "[read_file]: --- shared/Screenshot_2026-08-01_at_5.25.26_PM.png "
    "(5104148 bytes; truncated view) ---\n\x89PNG\r\n\x1a\n"
)
_MONSTER = _MONSTER_HEADER + ("�PNG\x89garbage" * ((494_932 - len(_MONSTER_HEADER)) // 12 + 1))
_MONSTER = _MONSTER[:494_932]


# ---------------------------------------------------------------- the seam


def test_elide_is_marked_deterministic_and_head_preserving() -> None:
    out1 = elide_oversized_content(_MONSTER, cap=32_000, label="oversized tool result")
    out2 = elide_oversized_content(_MONSTER, cap=32_000, label="oversized tool result")
    assert out1 == out2, "byte-stable across packing passes (prompt-cache property)"
    assert out1.startswith(_MONSTER[:1_000]), "head preserved, tail elided"
    assert "chars elided: oversized tool result" in out1
    assert "durable record" in out1
    assert len(out1) < 32_000 + 200
    # <= cap is a no-op, never a marker.
    assert elide_oversized_content("small", cap=32_000, label="x") == "small"
    assert elide_oversized_content(_MONSTER, cap=-1, label="x") == _MONSTER


def test_structural_floor_clamps_a_monster_tool_message() -> None:
    messages = [
        {"role": "user", "content": "please look at the screenshot"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "content": _MONSTER, "metadata": {"call_id": "c1"}},
        {"role": "assistant", "content": "that was a lot"},
    ]
    before = json.loads(json.dumps(messages))
    out = sanitize_transcript_messages(messages)
    tool_out = [m for m in out if m.get("role") == "tool"]
    assert len(tool_out) == 1
    assert len(tool_out[0]["content"]) <= OVERSIZED_MESSAGE_CLAMP_CHARS + 200
    assert "chars elided: oversized tool result" in tool_out[0]["content"]
    # Payload boundary ONLY: the durable input is not mutated.
    assert messages == before
    assert len(messages[2]["content"]) == 494_932
    # Bystander messages are untouched.
    assert out[0]["content"] == "please look at the screenshot"
    assert out[-1]["content"] == "that was a lot"


def test_structural_floor_applies_to_every_role() -> None:
    giant_prose = "y" * (OVERSIZED_MESSAGE_CLAMP_CHARS + 50_000)
    out = sanitize_transcript_messages([{"role": "user", "content": giant_prose}])
    assert len(out[0]["content"]) <= OVERSIZED_MESSAGE_CLAMP_CHARS + 200
    assert "chars elided: oversized user message" in out[0]["content"]


# ------------------------------------------------- task-lane ReAct behavior


def _task_runtime(captured: List[Dict[str, Any]], tool_output: str) -> Runtime:
    def llm_handler(run: RunState, effect: Effect, dnn: Optional[str]) -> EffectOutcome:
        del dnn
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))
        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed({
                "content": "reading",
                "tool_calls": [{"name": "read_file", "arguments": {"path": "big"}, "call_id": "c1"}],
            })
        return EffectOutcome.completed({"content": "done", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, dnn: Optional[str]) -> EffectOutcome:
        del run, dnn
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = [
            {"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": tool_output, "error": None}
            for tc in (payload.get("tool_calls") or [])
        ]
        return EffectOutcome.completed({"mode": "executed", "results": results})

    return Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )


def _task_workflow():
    return create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
        workflow_id="react_monster_clamp",
        provider="stub",
        model="stub",
        allowed_tools=["read_file"],
    )


def _run_task(tool_output: str) -> List[Dict[str, Any]]:
    captured: List[Dict[str, Any]] = []
    rt = _task_runtime(captured, tool_output)
    wf = _task_workflow()
    run_id = rt.start(workflow=wf, vars={"context": {"task": "read the big file"}})
    state = rt.tick(workflow=wf, run_id=run_id, max_steps=60)
    assert state.status == RunStatus.COMPLETED
    return captured


def test_task_lane_below_the_floor_is_byte_untouched() -> None:
    """The ReAct "full context" policy survives for honest sizes: a 150k
    tool result rides whole (init seeds the knobs to -1; only the 200k
    structural floor exists, and this is under it)."""
    honest = "H" * 150_000
    captured = _run_task(honest)
    assert len(captured) >= 2
    tool_msgs = [m for m in captured[-1]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == f"[read_file]: {honest}", "no clamp below the floor"


def test_task_lane_monster_is_floored_at_200k() -> None:
    captured = _run_task(_MONSTER)
    tool_msgs = [m for m in captured[-1]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert len(tool_msgs[0]["content"]) <= OVERSIZED_MESSAGE_CLAMP_CHARS + 200
    assert "chars elided: oversized tool result" in tool_msgs[0]["content"]


# --------------------------------- the incident lane: poisoned visit replay

yaml = pytest.importorskip("yaml", reason="entity stack absent")


def _make_home(tmp_path: Path, slug: str = "poisonling") -> Path:
    pytest.importorskip("abstractmemory")
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
    store, journal = SQLiteTripleStore(db), SQLiteJournal(db)
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


def _poison_tools(run: Any, effect: Effect, dnn: Any = None) -> EffectOutcome:
    """The PRE-FIX read_file shape: a monster string as the tool output —
    exactly what a stored session poisoned before the runtime-side repair
    still carries in its durable transcript."""
    payload = effect.payload if isinstance(effect.payload, dict) else {}
    results = [
        {
            "call_id": tc.get("call_id"), "name": tc.get("name"), "success": True,
            # observe renders "[read_file]: " + output, so hand it the body.
            "output": _MONSTER[len("[read_file]: "):],
            "error": None,
        }
        for tc in (payload.get("tool_calls") or [])
    ]
    return EffectOutcome.completed({"mode": "executed", "results": results})


def test_poisoned_visit_transcript_recovers_on_the_next_turn(tmp_path: Path) -> None:
    """End-to-end on the REAL merged lane (the incident's lane): turn 1
    poisons the durable transcript with a ~495k tool message; turn 2's wire
    payload carries a labeled 32k stub instead — the session recovers with
    NO manual surgery while the durable record keeps the full truth."""
    pytest.importorskip("abstractmemory")
    from abstractruntime.identity.entity_runtime import open_entity_runtime
    from abstractruntime.identity.visit_workflow import (
        HARVEST_NODE,
        VISIT_HISTORY_TOOL_RESULT_CAP_CHARS,
        VISITOR_WAIT_KEY,
        ReactMiddle,
        build_visit_workflow,
    )

    home_dir = _make_home(tmp_path)
    llm = _ScriptedLLM([
        # turn 1, iteration 1: elect the read (native tool call)
        {"content": "", "tool_calls": [{"name": "read_file", "arguments": {"path": "shared/Screenshot_2026-08-01_at_5.25.26_PM.png"}, "call_id": "c1"}]},
        # turn 1, iteration 2: answer after the monster observation
        {"content": "I looked at the file.", "tool_calls": []},
        # turn 2: plain answer — THIS call's payload is the recovery probe
        {"content": "Still here, and lighter.", "tool_calls": []},
    ])
    ert = open_entity_runtime(
        home_dir,
        extra_handlers={EffectType.LLM_CALL: llm, EffectType.TOOL_CALLS: _poison_tools},
    )
    try:
        react = create_react_workflow(
            logic=ReActLogic(tools=[ToolDefinition(name="read_file", description="Read", parameters={})]),
            workflow_id="entity-visit-react",
            provider="stub",
            model="stub",
            allowed_tools=["read_file"],
            final_next_node=HARVEST_NODE,
        )
        wf = build_visit_workflow(
            ert.home,
            participants=["person:albou"],
            idle_seconds=3600,
            model_info={"provider": "test", "model": "scripted"},
            visit_id="visit-poison1",
            react_middle=ReactMiddle(nodes=react.nodes, entry="reason", reset_turn=reset_react_turn),
        )
        run_id = ert.runtime.start(workflow=wf, vars={}, session_id="visit-poison")
        state = ert.runtime.tick(workflow=wf, run_id=run_id, max_steps=50)
        assert state.status == RunStatus.WAITING

        # Turn 1: the poisoning read.
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "I placed a screenshot in your workspace - read it.", "speaker": "person:albou"},
            max_steps=200,
        )
        assert state.status == RunStatus.WAITING

        # The DURABLE transcript now holds the full monster (this is the
        # stored-session poison class: no surgery has happened, none will).
        stored = ert.runtime.get_state(run_id).vars["context"]["messages"]
        stored_tool = [m for m in stored if m.get("role") == "tool"]
        assert len(stored_tool) == 1
        assert len(stored_tool[0]["content"]) == len(_MONSTER)

        # Turn 2: the recovery probe.
        state = ert.runtime.resume(
            workflow=wf, run_id=run_id, wait_key=VISITOR_WAIT_KEY,
            payload={"text": "Are you still with me?", "speaker": "person:albou"},
            max_steps=200,
        )
        assert state.status == RunStatus.WAITING
        assert len(llm.calls) == 3

        wire = llm.calls[2]["messages"]
        wire_tool = [m for m in wire if m.get("role") == "tool"]
        assert len(wire_tool) == 1
        clamped = wire_tool[0]["content"]
        assert len(clamped) <= VISIT_HISTORY_TOOL_RESULT_CAP_CHARS + 200
        assert clamped.startswith("[read_file]: --- shared/Screenshot_2026-08-01"), "head kept - labeled, never dropped"
        assert "chars elided: oversized tool result" in clamped
        # The visitor's words and the entity's own prose are untouched.
        user_msgs = [m for m in wire if m.get("role") == "user"]
        assert any("Are you still with me?" in str(m.get("content")) for m in user_msgs)
        assert any(m.get("role") == "assistant" and "I looked at the file." == m.get("content") for m in wire)
        # The whole request shrank from monster-class to sane.
        total = sum(len(str(m.get("content") or "")) for m in wire)
        assert total < 100_000, f"recovered payload still huge: {total}"

        # And the durable record STILL holds the full truth (ADR-0026).
        stored = ert.runtime.get_state(run_id).vars["context"]["messages"]
        stored_tool = [m for m in stored if m.get("role") == "tool"]
        assert len(stored_tool[0]["content"]) == len(_MONSTER)
    finally:
        ert.close()
