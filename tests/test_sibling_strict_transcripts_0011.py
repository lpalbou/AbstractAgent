"""Pins for backlog 0011: CodeAct/MemAct transcripts on strict providers.

The bug: sibling parse nodes appended assistant messages CONTENT-ONLY while
their sanitizers emitted `role:"tool"` + `tool_call_id` — an orphan on any
strict provider (native OpenAI: "assistant tool_calls must be followed by
matching tool messages" -> 400 at iteration 2). ReAct documented and repaired
the exact class; the fix EXTRACTS its pipeline (adapters/transcripts.py) and
wires all three loops onto it.

Validation shape per the backlog: a two-iteration tool run per sibling whose
sanitized payload satisfies the native tool-calling contract — every tool
message's `tool_call_id` answered by a preceding assistant `tool_calls` entry.
"""
from __future__ import annotations

from typing import Any, Dict, List

from abstractagent.adapters.transcripts import sanitize_transcript_messages
from abstractcore.tools import ToolDefinition


def _contract_violations(payload: List[Dict[str, Any]]) -> List[str]:
    """Native tool-calling contract check over a sanitized payload."""
    problems: List[str] = []
    announced: set = set()
    pending: set = set()
    for i, m in enumerate(payload):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if pending:
                problems.append(f"msg {i}: new tool_calls while {sorted(pending)} unanswered")
            pending = {str(tc.get("id")) for tc in m["tool_calls"]}
            announced |= pending
        elif role == "tool":
            tid = str(m.get("tool_call_id") or "")
            if not tid:
                problems.append(f"msg {i}: tool message without tool_call_id")
            elif tid not in announced:
                problems.append(f"msg {i}: tool_call_id {tid} never announced")
            pending.discard(tid)
        else:
            if pending:
                problems.append(f"msg {i}: {sorted(pending)} unanswered before role={role}")
                pending = set()
    if pending:
        problems.append(f"trailing unanswered ids: {sorted(pending)}")
    return problems


def _drive_two_tool_iterations(create_workflow, logic, wf_id: str, sanitizer_probe) -> List[Dict[str, Any]]:
    """Run parse->act->observe twice with scripted responses, then sanitize."""
    from abstractruntime.core.models import RunState, RunStatus

    class _Ctx:
        @staticmethod
        def now_iso() -> str:
            return "2025-01-01T00:00:00+00:00"

    wf = create_workflow(logic=logic)
    vars: Dict[str, Any] = {
        "context": {"task": "t", "messages": [{"role": "user", "content": "t"}]},
        "scratchpad": {"iteration": 0, "max_iterations": 10},
        "_runtime": {"inbox": [], "allowed_tools": ["read_file"]},
        "_temp": {},
        "_limits": {"max_iterations": 10, "current_iteration": 0},
    }
    run = RunState(run_id="r", workflow_id=wf_id, status=RunStatus.RUNNING, current_node="parse", vars=vars)

    for turn in (1, 2):
        run.vars["_temp"]["llm_response"] = {
            "content": f"looking (turn {turn})",
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": f"f{turn}.txt"}, "call_id": f"call_t{turn}"}
            ],
        }
        plan = wf.get_node("parse")(run, _Ctx())
        assert plan.next_node == "act", f"turn {turn} did not route to act"
        # Simulate the executed observation exactly as observe_node appends it.
        run.vars["context"]["messages"].append(
            {
                "role": "tool",
                "content": f"[read_file]: contents {turn}",
                "metadata": {"name": "read_file", "call_id": f"call_t{turn}", "success": True},
            }
        )
        run.vars["_temp"].pop("pending_tool_calls", None)

    return sanitizer_probe(run.vars["context"]["messages"])


_READ = ToolDefinition(name="read_file", description="r", parameters={})


def test_codeact_two_iteration_transcript_satisfies_strict_contract() -> None:
    from abstractagent.adapters.codeact_runtime import create_codeact_workflow
    from abstractagent.logic.codeact import CodeActLogic

    payload = _drive_two_tool_iterations(
        create_codeact_workflow,
        CodeActLogic(tools=[_READ]),
        "codeact_agent",
        lambda msgs: sanitize_transcript_messages(msgs),
    )
    assert _contract_violations(payload) == []
    # The assistant turns actually announce their batches (not repaired-away).
    announcing = [m for m in payload if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(announcing) == 2


def test_memact_two_iteration_transcript_satisfies_strict_contract() -> None:
    from abstractagent.adapters.memact_runtime import create_memact_workflow
    from abstractagent.logic.memact import MemActLogic

    payload = _drive_two_tool_iterations(
        create_memact_workflow,
        MemActLogic(tools=[_READ]),
        "memact_agent",
        lambda msgs: sanitize_transcript_messages(msgs),
    )
    assert _contract_violations(payload) == []
    announcing = [m for m in payload if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(announcing) == 2


def test_orphan_repair_both_directions() -> None:
    """Unanswered tool_calls ids get deterministic synthetic results; unpaired
    tool messages fold into inert user notes (never a 400 either way)."""
    msgs = [
        {"role": "user", "content": "t"},
        # Assistant announces two calls; only one is answered (e.g. crash window).
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "a", "function": {"name": "read_file", "arguments": "{}"}},
                {"type": "function", "id": "b", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "ok", "metadata": {"call_id": "a"}},
        # Unpaired tool message (compaction cut its assistant turn).
        {"role": "tool", "content": "stray result", "metadata": {"call_id": "zzz"}},
    ]
    payload = sanitize_transcript_messages(msgs)
    assert _contract_violations(payload) == []
    synthetic = [m for m in payload if m.get("role") == "tool" and m.get("tool_call_id") == "b"]
    assert synthetic and "missing" in synthetic[0]["content"]
    folded = [m for m in payload if m.get("role") == "user" and "unpaired tool result" in str(m.get("content"))]
    assert folded and "stray result" in folded[0]["content"]


def test_interactive_builtin_unanswered_is_labeled_honestly() -> None:
    msgs = [
        {"role": "user", "content": "t"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "q1", "function": {"name": "ask_user", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "the human's answer"},
    ]
    payload = sanitize_transcript_messages(msgs)
    assert _contract_violations(payload) == []
    synth = [m for m in payload if m.get("role") == "tool" and m.get("tool_call_id") == "q1"]
    assert synth and "handled interactively" in synth[0]["content"]
