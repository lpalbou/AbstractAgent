"""ReAct prompt-prefix cache stability (backlog 0212) + context fidelity (backlog 0213).

Drives a scripted 3-iteration ReAct tool loop through the REAL runtime (so the runtime
ledger grounding pass applies, exactly as in production) and asserts the provider
prompt-cache contract:

- the request prefix (system prompt + tools + all-but-last message) is byte-identical
  between consecutive iterations;
- volatile per-call state (grounding envelope, loop position) rides ONLY the trailing
  message;
- the assistant tool-call transcript message carries the model's reasoning content;
- the system prompt carries no per-cycle scratchpad block and no iteration counter;
- every lossy slice left in the adapter carries the ADR-0026 `#[WARNING:TRUNCATION]` tag.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore

pytestmark = pytest.mark.basic


_THOUGHTS = {
    1: "Checking the workspace first.",
    2: "Now creating the project folder.",
}


def _run_scripted_loop() -> tuple[List[Dict[str, Any]], RunState]:
    """3 LLM iterations: two tool cycles, then a final answer. Returns captured payloads."""
    captured: List[Dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        captured.append(json.loads(json.dumps(payload)))

        iteration = int(run.vars.get("_limits", {}).get("current_iteration", 0) or 0)
        if iteration == 1:
            return EffectOutcome.completed(
                {
                    "content": _THOUGHTS[1],
                    "tool_calls": [{"name": "list_files", "arguments": {"directory_path": "."}, "call_id": "call_1"}],
                }
            )
        if iteration == 2:
            return EffectOutcome.completed(
                {
                    "content": _THOUGHTS[2],
                    "tool_calls": [{"name": "execute_command", "arguments": {"command": "mkdir -p project"}, "call_id": "call_2"}],
                }
            )
        return EffectOutcome.completed({"content": "Created the project folder.", "tool_calls": []})

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
        logic=ReActLogic(
            tools=[
                ToolDefinition(name="list_files", description="List", parameters={}),
                ToolDefinition(name="execute_command", description="Cmd", parameters={}),
            ]
        ),
        workflow_id="react_prefix_stability",
        provider="stub",
        model="stub",
        allowed_tools=["list_files", "execute_command"],
    )
    run_id = runtime.start(
        workflow=workflow,
        vars={"context": {"task": "Create a project folder", "messages": []}, "_runtime": {"inbox": []}},
        actor_id=None,
        session_id="sess-prefix",
    )
    for _ in range(50):
        state = runtime.tick(workflow=workflow, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    state = runtime.get_state(run_id)
    assert state.status == RunStatus.COMPLETED
    return captured, state


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def test_react_request_prefix_is_byte_identical_across_iterations() -> None:
    payloads, _ = _run_scripted_loop()
    assert len(payloads) == 3

    # (1) The static request modules are byte-identical across ALL iterations.
    sys_prompts = [str(p.get("system_prompt") or "") for p in payloads]
    assert sys_prompts[0] == sys_prompts[1] == sys_prompts[2]
    tools = [_dumps(p.get("tools")) for p in payloads]
    assert tools[0] == tools[1] == tools[2]

    # (2) No per-cycle entropy in the system prompt.
    assert "Iteration:" not in sys_prompts[0]
    assert "Scratchpad" not in sys_prompts[0]
    assert "[cycle" not in sys_prompts[0]

    # (3) Message-lane prefix stability: everything before the trailing volatile message
    # of iteration N reappears BYTE-IDENTICAL at the head of iteration N+1.
    for a, b in ((payloads[0], payloads[1]), (payloads[1], payloads[2])):
        msgs_a = a.get("messages")
        msgs_b = b.get("messages")
        assert isinstance(msgs_a, list) and isinstance(msgs_b, list)
        stable_prefix = msgs_a[:-1]
        assert _dumps(msgs_b[: len(stable_prefix)]) == _dumps(stable_prefix)
        # The transcript grew (assistant tool-call + tool observation + trailing turn).
        assert len(msgs_b) > len(msgs_a)

    # Mission A (2026-09-22) strengthens the first pair: iteration 1 no longer
    # decorates the task message with loop chrome, so ALL of iteration 1's
    # messages — not just all-but-last — reappear byte-identical in iteration 2.
    assert _dumps((payloads[1].get("messages") or [])[: len(payloads[0].get("messages") or [])]) == _dumps(
        payloads[0].get("messages") or []
    )

    # (4) Structural metric: the stable prefix grows monotonically (append-only lane).
    prefix_chars = [len(_dumps((p.get("messages") or [])[:-1])) for p in payloads]
    assert prefix_chars[0] < prefix_chars[1] < prefix_chars[2]


def test_react_volatile_state_rides_only_the_trailing_message() -> None:
    payloads, _ = _run_scripted_loop()

    for i, p in enumerate(payloads, start=1):
        msgs = p.get("messages")
        assert isinstance(msgs, list) and msgs

        # The loop counter lands ONLY in the trailing message. The grounding
        # envelope no longer moves at all: since mission A it is STAMPED once into
        # the durable task message (that is what makes the message replayable
        # byte-for-byte), and the per-call clock rides the trailing volatile
        # message with the counter.
        # Mission A3: earlier iterations' tails are now DURABLE (sent bytes ==
        # stored bytes) and reappear where they were sent; every one of them is a
        # marked, adapter-authored message — never folded into another message.
        for m in msgs[:-1]:
            if "[loop]" in str(m.get("content") or ""):
                assert m.get("_af_synthetic") == "loop_tail"
                assert str(m.get("content") or "").startswith("[loop] iteration ")

        task_msg = str(msgs[0].get("content") or "")
        assert task_msg.startswith("<runtime_metadata>")
        assert task_msg.endswith("Create a project folder")

        tail = str(msgs[-1].get("content") or "")
        assert msgs[-1].get("role") == "user"
        if i == 1:
            # Iteration 1 is chat-shaped: the durable task IS the trailing message,
            # so it carries no chrome at all — nothing volatile is sent.
            assert tail is task_msg or tail == task_msg
            assert "[loop]" not in tail
        else:
            assert f"[loop] iteration {i}" in tail

    # The adjacency-guard TRADE is gone (mission A, 2026-09-22). On iteration 1 the
    # task message is also the trailing user message, but the loop-position line is
    # chrome and is dropped there instead of merged in — so the task message is
    # byte-identical from iteration 1 onward and the whole prefix is reusable for
    # the entire run, not from iteration 2.
    firsts = [_dumps((p.get("messages") or [])[0]) for p in payloads]
    assert firsts[0] == firsts[1] == firsts[2]
    assert "Create a project folder" in firsts[0]
    assert "[loop]" not in firsts[0]

    # Structural volatile marker (B1 pair, code c971 → runtime c986): when the
    # tail rides as a SEPARATE trailing message (iterations 2+, the tool-loop
    # shape), it carries top-level `volatile: true` — runtime's llm_client
    # excludes it from the prompt-cache fingerprint sequence and strips the
    # key before any provider sees it. The merged first-turn message must NOT
    # carry the flag (it holds the real task).
    # Mission A3 supersedes the `volatile` flag here: the tail is durable and
    # marked `_af_synthetic: loop_tail` (runtime skips it when locating the turn
    # and strips the key before the provider), so iteration N's payload is an
    # EXACT prefix of N+1's instead of all-but-its-last-message.
    for p in payloads[1:]:
        tail_msg = (p.get("messages") or [])[-1]
        assert tail_msg.get("_af_synthetic") == "loop_tail"
        assert "volatile" not in tail_msg
    for a, b in zip(payloads, payloads[1:]):
        ma, mb = a.get("messages") or [], b.get("messages") or []
        assert _dumps(mb[: len(ma)]) == _dumps(ma)
    first_msg = (payloads[0].get("messages") or [])[0]
    assert "volatile" not in first_msg
    # Iteration 1 now carries NO volatile message at all: the only message is the
    # durable task, stamped once.
    assert len(payloads[0].get("messages") or []) == 1


def test_react_assistant_tool_call_messages_retain_reasoning_content() -> None:
    payloads, state = _run_scripted_loop()

    # In the LLM-visible payload of iteration 3, both prior assistant tool-call turns
    # carry their reasoning content (0213: no more content="").
    msgs = payloads[2].get("messages")
    assert isinstance(msgs, list)
    assistant_tool_msgs = [m for m in msgs if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)]
    assert len(assistant_tool_msgs) == 2
    assert assistant_tool_msgs[0].get("content") == _THOUGHTS[1]
    assert assistant_tool_msgs[1].get("content") == _THOUGHTS[2]

    # And the durable transcript stores the same non-empty reasoning.
    context = state.vars.get("context") if isinstance(state.vars, dict) else None
    durable = [
        m
        for m in (context.get("messages") if isinstance(context, dict) else [])
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list)
    ]
    assert [m.get("content") for m in durable] == [_THOUGHTS[1], _THOUGHTS[2]]

    # The durable scratchpad record is kept for host observability (0213 non-goal: no data loss).
    scratchpad = state.vars.get("scratchpad") if isinstance(state.vars, dict) else None
    cycles = scratchpad.get("cycles") if isinstance(scratchpad, dict) else None
    assert isinstance(cycles, list) and len(cycles) >= 2
    assert cycles[0].get("thought") == _THOUGHTS[1]


def test_react_adapter_lossy_slices_carry_adr_0026_truncation_tag() -> None:
    """ADR-0026 §4: every lossy content truncation site must carry the literal
    `#[WARNING:TRUNCATION]` tag. A lossy site is a function whose body both slices text
    and emits the `…` truncation marker. Hash/identifier prefixes (sha256 digests,
    fingerprints) carry no `…` marker and are explicitly out of ADR scope."""
    adapter_path = Path(__file__).resolve().parents[1] / "src" / "abstractagent" / "adapters" / "react_runtime.py"
    source = adapter_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if not segment:
            continue
        # Ignore the docstring when looking for the marker character in code.
        docstring = ast.get_docstring(node) or ""
        code_only = segment.replace(docstring, "")
        has_marker = "…" in code_only
        has_slice = "[:" in code_only or "[: " in code_only
        if has_marker and has_slice and "#[WARNING:TRUNCATION]" not in segment:
            offenders.append(node.name)

    assert not offenders, f"Untagged lossy truncation sites in react_runtime.py: {offenders}"
    # Guard against the tag disappearing entirely (both known sites must stay tagged).
    assert source.count("#[WARNING:TRUNCATION]") >= 2


def test_react_truncation_previews_carry_explicit_marker() -> None:
    from abstractagent.adapters.react_runtime import _tool_call_signature, _truncate_preview

    long_text = "x" * 2000
    preview = _truncate_preview(long_text, max_chars=360)
    assert preview.endswith("… (truncated, 2,000 chars total)")
    assert len(preview) <= 360

    sig = _tool_call_signature("fetch_url", {"url": "https://example.com/" + "a" * 500, "include_full_content": False})
    assert "… (truncated," in sig

    # Short content passes through unmodified (no gratuitous markers).
    assert _truncate_preview("short", max_chars=360) == "short"
    assert "truncated" not in _tool_call_signature("web_search", {"query": "hi", "num_results": 3})
