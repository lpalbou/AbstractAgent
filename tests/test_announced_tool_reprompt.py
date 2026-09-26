"""A reply that announces tool use, or carries tool calls that cannot run, is not an answer.

Mission AGX (2026-09-26). Evidence: untracked/missions-2026-09-25/XP/REPORT.md in
the framework root, and root backlog 0918.

- With Qwen3.x on MLX the model often ends a step with ONE sentence announcing
  a tool use and no call ("I have strong material. Let me verify ... before
  writing the digest."). The ReAct loop took it as the final answer: 5/5
  baseline runs "completed" with no digest.
- Re-prompting ONCE with the failed reply kept VERBATIM made the model re-issue
  the same calls visibly 5/5; re-prompting with the reply as the runtime
  records it (content "") made it believe the tools had already run, 5/5.
- abstractcore >= 2.16.0 recovers complete calls from the thinking block
  (`metadata.tool_calls_from_reasoning`) and reports the ones it cannot run
  (`metadata.unparsed_tool_call`, unknown-tool warnings).

These tests drive the real workflows with a fake provider.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractagent.adapters.announced_calls import (
    REASON_ANNOUNCED,
    REASON_UNRUNNABLE,
    classify_no_call_reply,
    looks_like_tool_announcement,
    verbatim_reply,
)
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


# XP report, Appendix A: every intent-only final answer observed (verbatim).
XP_ANNOUNCEMENTS = [
    "I have strong material. Let me verify a couple of key specifics (oil price level, gold, and the Iran war status) before writing the digest.",
    "I have strong material. Let me get a bit more on the US-Iran war status, the ECB/monetary picture, and emerging tech/AI investment opportunities to round out the digest.",
    "Let me gather a few more specific threads: the Iran war timeline/ceasefire, the US-Iran talks, and broader market/economic signals.",
    "I have strong material. Let me fill two gaps: the US–Iran situation (a major driver of oil and geopolitics) and the Trump–Xi summit, then I'll compile the digest.",
    "Let me get a few more specifics: oil price reaction, the Germany political crisis, and the Trump-Xi summit outcome.",
    "I have strong material. Let me confirm the Iran/Hormuz situation and grab the trade truce detail, then I'll compose the digest.",
    "I have strong material. Let me pull a couple of primary sources to firm up the Iran war and the Trump–Xi summit specifics, then synthesize.",
    "I have enough to synthesize a high-quality, actionable digest. Let me pull one more concrete thread — the Strait of Hormuz / oil supply situation, since it's the linchpin connecting war risk, commodity pricing, and the Fed.",
    "I have strong material across all five areas. Let me do two final targeted checks — the commodity outlook and any notable investment/commercial opportunities — then compile the digest.",
    "I have strong material. Let me confirm a couple of specifics — the current oil price level (sources conflict) and the rare earths angle — then I'll write the digest.",
    "I have strong material across all five areas. Let me do one final pass to confirm the macro/inflation picture and any notable investment/commercial angles, then I'll compile the digest.",
    "I have strong material. Let me confirm a couple of specifics — the commodity outlook and the rare-earth/energy angle — then compose the digest.",
    "I have strong material. Let me verify a couple of key data points (oil price level, gold) and check the broader macro/inflation picture before writing the digest.",
    "I have strong material across all threads. Let me confirm a few specifics: the Iran war ceasefire status and gold levels, plus the rare-earth angle.",
    "I have strong material. Let me confirm a couple more specifics: the current oil price picture (sources conflict on ~$98 vs ~$130), the gold level, and the investment/commercial angle on the memory shortage.",
    "I have strong material across all requested areas. Let me do one more pass to confirm a couple of forward-looking items—rare earths, defense investment, and the inflation trajectory—then compile the digest.",
    "I have strong material. Let me confirm a couple of key figures (oil price level, gold, and the rare-earth angle) before writing the digest.",
    "I have strong material across all five areas. Let me verify a couple of the highest-impact claims (oil price level, the Hormuz/Iran war status, and gold) before writing the digest.",
]

# Realistic final answers that must NOT be re-prompted.
GENUINE_FINALS = [
    "Done.",
    "Done. The folder `project/` now contains main.py and a README.",
    "The capital of France is Paris.",
    "I will not run that command: it would delete your home directory.",
    "Here is the summary: the build fails because PYTHONPATH is unset in CI.",
    "Should I also update the README?",
    "Let me know if you want me to run the full test suite as well.",
    "I checked the logs; the failure comes from a missing environment variable (API_KEY).",
    "The file is already correct, so no changes were needed.",
    "All set — the three files were renamed and the imports updated.",
    (
        "## Digest\n\n"
        "- Oil: Brent traded near $98 after the Hormuz incident; I'll keep this short.\n"
        "- Gold: record high on safe-haven demand.\n"
        "- Rates: the ECB held, citing energy-driven inflation.\n"
    ),
    # Long prose answer that happens to carry intent words: over the short-reply cap.
    (
        "The Fed held rates at 4.25-4.50% and signalled two cuts later this year; markets priced the first for "
        "September. Oil rose 6% on the Hormuz disruption and gold printed a record. In Europe, the ECB paused while "
        "Germany's coalition talks stalled. For investors, energy and defense outperformed while rate-sensitive tech "
        "lagged. Let me add that the rare-earth export curbs remain the main supply-chain risk for Q4."
    ),
]


def test_every_xp_announcement_is_detected() -> None:
    for text in XP_ANNOUNCEMENTS:
        assert looks_like_tool_announcement(text), text


@pytest.mark.parametrize("text", GENUINE_FINALS)
def test_genuine_finals_are_not_announcements(text: str) -> None:
    assert looks_like_tool_announcement(text) is False
    assert classify_no_call_reply({"content": text}, text, tools_offered=True) == ("", "")


def test_ill_is_not_intent() -> None:
    # The old optional-apostrophe spelling matched the word "ill".
    assert looks_like_tool_announcement("The ill-formed file was fixed; run it again to check.") is False


def test_reasoning_markup_behind_a_visible_answer_is_an_answer() -> None:
    resp = {
        "content": "Paris.",
        "reasoning": "Maybe <tool_call>{\"name\": \"web_search\"}</tool_call> — no, I know this.",
    }
    assert classify_no_call_reply(resp, "Paris.", tools_offered=True) == ("", "")


def test_verbatim_reply_keeps_the_thinking_block() -> None:
    resp = {"content": "", "reasoning": "<tool_call>\n<function=fetch_page>\n</function>\n</tool_call>"}
    raw = verbatim_reply(resp)
    assert raw.startswith("<think>\n<tool_call>")
    assert "<function=fetch_page>" in raw


# ---------------------------------------------------------------------------
# Loop driver
# ---------------------------------------------------------------------------

_FETCH = ToolDefinition(name="fetch_url", description="Fetch a URL", parameters={"url": {"type": "string"}})
_SEARCH = ToolDefinition(name="web_search", description="Search", parameters={"query": {"type": "string"}})


def _call(name: str, cid: str, **args: Any) -> Dict[str, Any]:
    return {"name": name, "arguments": dict(args), "call_id": cid}


def _drive(
    loop: str,
    script: List[Dict[str, Any]],
    *,
    tools: Optional[List[ToolDefinition]] = None,
    runtime_vars: Optional[Dict[str, Any]] = None,
    max_iterations: Optional[int] = None,
) -> Tuple[RunState, List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (state, llm payloads, steps, executed tool calls, ledger records)."""
    payloads: List[Dict[str, Any]] = []
    steps: List[Tuple[str, Dict[str, Any]]] = []
    executed: List[Dict[str, Any]] = []
    tools = [_FETCH, _SEARCH] if tools is None else tools

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        payloads.append(json.loads(json.dumps(payload, default=str)))
        idx = min(len(payloads) - 1, len(script) - 1)
        return EffectOutcome.completed(json.loads(json.dumps(script[idx])))

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        del run, default_next_node
        payload = effect.payload if isinstance(effect.payload, dict) else {}
        results = []
        for tc in payload.get("tool_calls") or []:
            executed.append(dict(tc))
            results.append({"call_id": tc.get("call_id"), "name": tc.get("name"), "success": True, "output": "ok", "error": None})
        return EffectOutcome.completed({"mode": "executed", "results": results})

    on_step = lambda step, data: steps.append((step, dict(data)))  # noqa: E731
    if loop == "react":
        wf = create_react_workflow(
            logic=ReActLogic(tools=tools), provider="stub", model="stub", on_step=on_step
        )
    elif loop == "codeact":
        wf = create_codeact_workflow(logic=CodeActLogic(tools=tools), on_step=on_step)
    else:
        wf = create_memact_workflow(logic=MemActLogic(tools=tools), on_step=on_step)

    ledger = InMemoryLedgerStore()
    runtime = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=ledger,
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    rt: Dict[str, Any] = {"inbox": []}
    rt.update(runtime_vars or {})
    vars_: Dict[str, Any] = {"context": {"task": "Write the daily digest", "messages": []}, "_runtime": rt}
    if max_iterations is not None:
        vars_["_limits"] = {"max_iterations": max_iterations}
    run_id = runtime.start(workflow=wf, vars=vars_, actor_id=None, session_id=f"sess-agx-{loop}")
    for _ in range(200):
        state = runtime.tick(workflow=wf, run_id=run_id, max_steps=1)
        if state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
    return runtime.get_state(run_id), payloads, steps, executed, ledger.list(run_id)


def _names(steps: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    return [s for s, _ in steps]


def _answer(state: RunState) -> str:
    return str((state.output or {}).get("answer") or "")


ANNOUNCE = {"content": XP_ANNOUNCEMENTS[0], "tool_calls": [], "finish_reason": "stop"}
FIRST_BATCH = {
    "content": "Searching.",
    "tool_calls": [_call("web_search", "c1", query="world news today")],
    "finish_reason": "tool_calls",
}
VISIBLE_CALLS = {
    "content": "",
    "tool_calls": [
        _call("fetch_url", "c2", url="https://example.org/oil"),
        _call("fetch_url", "c3", url="https://example.org/gold"),
    ],
    "finish_reason": "tool_calls",
}
DIGEST = {"content": "# Digest\n\n- Oil: $98.\n- Gold: record high.", "tool_calls": [], "finish_reason": "stop"}

# The operator's shape (backlog 0918): calls in the thinking block that core
# could NOT recover because the tool name was not offered.
THINK_MARKUP = (
    "I should fetch the sources.\n"
    "<tool_call>\n<function=fetch_page>\n<parameter=url>\nhttps://example.org/oil\n</parameter>\n"
    "</function>\n</tool_call>"
)
UNKNOWN_TOOL_IN_THINKING = {
    "content": "",
    "reasoning": THINK_MARKUP,
    "tool_calls": [],
    "finish_reason": "stop",
    "metadata": {
        "reasoning": THINK_MARKUP,
        "warnings": [
            "Tool call not recognized: the model called 'fetch_page' but no such tool is available "
            "(available: fetch_url, web_search). The response was returned as plain content."
        ],
    },
}


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------


def test_react_announcement_is_reprompted_once_with_the_verbatim_reply_then_calls_run() -> None:
    state, payloads, steps, executed, ledger = _drive("react", [FIRST_BATCH, ANNOUNCE, VISIBLE_CALLS, DIGEST])

    assert state.status == RunStatus.COMPLETED
    names = _names(steps)
    assert names.count("parse_reprompt") == 1
    assert "parse_reprompt_failed" not in names
    reprompt = next(d for s, d in steps if s == "parse_reprompt")
    assert reprompt["reason"] == REASON_ANNOUNCED

    # The re-prompt call shows the model its own reply VERBATIM, then the corrective.
    msgs = payloads[2]["messages"]
    idx = next(i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("content") == ANNOUNCE["content"])
    corrective = msgs[idx + 1]
    assert corrective["role"] == "user"
    assert "announced tool calls but none ran" in corrective["content"]
    assert "call the tools now in the required format as your visible reply, or answer directly" in corrective["content"]

    # Ledger: the re-prompt call is marked on its effect payload.
    assert payloads[2]["_runtime_observability"]["reprompted"] == REASON_ANNOUNCED
    assert "_runtime_observability" not in payloads[1]
    llm_records = [r for r in ledger if (r.get("effect") or {}).get("type") == "llm_call"]
    marked = [r for r in llm_records if ((r["effect"].get("payload") or {}).get("_runtime_observability") or {}).get("reprompted")]
    assert len(marked) >= 1

    # The re-issued calls ran; the answer is the digest, never the announcement.
    assert [tc["name"] for tc in executed] == ["web_search", "fetch_url", "fetch_url"]
    assert _answer(state).startswith("# Digest")
    ptc = [d for s, d in steps if s == "parse_tool_calls"]
    assert ptc[-1].get("after_reprompt") == REASON_ANNOUNCED
    assert (state.output or {}).get("stop_reason", {}).get("code") == "final_answer"


def test_react_unknown_tool_in_thinking_is_reprompted_with_the_markup_visible() -> None:
    state, payloads, steps, executed, _ = _drive("react", [UNKNOWN_TOOL_IN_THINKING, VISIBLE_CALLS, DIGEST])

    assert state.status == RunStatus.COMPLETED
    reprompt = next(d for s, d in steps if s == "parse_reprompt")
    assert reprompt["reason"] == REASON_UNRUNNABLE
    assert "fetch_page" in reprompt["detail"]

    # The model sees its own think-held call — never the emptied record.
    msgs = payloads[1]["messages"]
    shown = [m for m in msgs if m.get("role") == "assistant" and "<function=fetch_page>" in str(m.get("content") or "")]
    assert len(shown) == 1
    assert shown[0]["content"].startswith("<think>\n")
    assert "cannot run" in msgs[msgs.index(shown[0]) + 1]["content"]

    assert [tc["name"] for tc in executed] == ["fetch_url", "fetch_url"]
    assert "<tool_call>" not in _answer(state)
    assert _answer(state).startswith("# Digest")


def test_react_double_failure_ends_the_step_with_a_visible_error_and_concludes() -> None:
    conclusion = {"content": "Best effort digest: oil near $98; gold at a record.", "tool_calls": [], "finish_reason": "stop"}
    state, payloads, steps, executed, _ = _drive("react", [FIRST_BATCH, ANNOUNCE, ANNOUNCE, conclusion])

    assert state.status == RunStatus.COMPLETED
    names = _names(steps)
    assert names.count("parse_reprompt") == 1
    assert names.count("parse_reprompt_failed") == 1
    # 4 calls: first batch, announcement, re-prompt (announcement again), conclusion.
    assert len(payloads) == 4
    assert "tools" not in payloads[3] or not payloads[3].get("tools")
    assert "announced tool calls but contained none" in json.dumps(payloads[3]["messages"])

    out = state.output or {}
    assert out["answer"] == conclusion["content"]
    assert ANNOUNCE["content"] not in out["answer"]
    sr = out["stop_reason"]
    assert sr["code"] == "no_tool_call" and sr["finished"] is False and sr["budget_exhausted"] is False
    assert "announced tool use without calling any tool" in sr["headline"]
    assert out["no_tool_call_stop"]["reason"] == REASON_ANNOUNCED
    assert any(n["code"] == "no_tool_call" and n["severity"] == "error" for n in out["notices"])
    assert "conclusion forced: announced_tool_use" in out["report"]
    assert "re-prompted once" in out["report"]
    assert [tc["name"] for tc in executed] == ["web_search"]


def test_react_double_failure_with_markup_never_publishes_markup() -> None:
    conclusion_also_announces = {"content": "Let me fetch the pages first.", "tool_calls": [], "finish_reason": "stop"}
    state, _, steps, executed, _ = _drive(
        "react", [UNKNOWN_TOOL_IN_THINKING, UNKNOWN_TOOL_IN_THINKING, conclusion_also_announces]
    )
    out = state.output or {}
    answer = out["answer"]
    assert "<tool_call>" not in answer and "fetch_page>" not in answer
    assert "Let me fetch the pages first." not in answer
    assert answer.startswith("Error: the model's reply announced tool calls it could not run")
    assert out["stop_reason"]["code"] == "no_tool_call"
    assert "tool call could not run" in out["stop_reason"]["label"]
    assert "conclusion_announcement_dropped" in _names(steps)
    assert executed == []


def test_react_unparsed_cut_off_call_is_reprompted() -> None:
    cut = {
        "content": "",
        "tool_calls": [],
        "finish_reason": "stop",
        "metadata": {
            "unparsed_tool_call": {
                "format": "qwen3_coder",
                "reason": "unterminated",
                "text": "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://exa",
            }
        },
    }
    state, payloads, steps, executed, _ = _drive("react", [cut, VISIBLE_CALLS, DIGEST])
    reprompt = next(d for s, d in steps if s == "parse_reprompt")
    assert reprompt["reason"] == REASON_UNRUNNABLE and "cut off" in reprompt["detail"]
    assert any("https://exa" in str(m.get("content") or "") for m in payloads[1]["messages"] if m.get("role") == "assistant")
    assert len(executed) == 2 and state.status == RunStatus.COMPLETED


@pytest.mark.parametrize("text", GENUINE_FINALS)
def test_react_genuine_final_is_not_reprompted(text: str) -> None:
    state, payloads, steps, _, _ = _drive("react", [{"content": text, "tool_calls": [], "finish_reason": "stop"}])
    assert "parse_reprompt" not in _names(steps)
    assert "parse_retry_plan_only" not in _names(steps)
    assert len(payloads) == 1
    assert _answer(state) == text.strip()


def test_react_markup_without_tools_says_no_tools_are_available() -> None:
    markup = {"content": "<tool_call>{\"name\": \"web_search\", \"arguments\": {}}</tool_call>", "tool_calls": []}
    _, payloads, steps, _, _ = _drive("react", [markup, DIGEST], tools=[])
    reprompt = next(d for s, d in steps if s == "parse_reprompt")
    assert reprompt["reason"] == REASON_UNRUNNABLE
    assert "no tools are available" in payloads[1]["messages"][-1]["content"]


def test_react_announcement_on_the_last_iteration_goes_to_the_conclusion() -> None:
    conclusion = {"content": "Best effort digest.", "tool_calls": [], "finish_reason": "stop"}
    state, payloads, steps, _, _ = _drive("react", [FIRST_BATCH, ANNOUNCE, conclusion], max_iterations=2)
    names = _names(steps)
    assert "parse_reprompt" not in names and "parse_reprompt_skipped" in names
    assert _answer(state) == "Best effort digest."
    assert len(payloads) == 3


def test_react_visit_lane_markup_reprompt_is_host_voiced() -> None:
    state, payloads, steps, _, _ = _drive(
        "react", [UNKNOWN_TOOL_IN_THINKING, DIGEST], runtime_vars={"suppress_loop_tail": True}
    )
    assert "parse_reprompt" in _names(steps)
    corrective = payloads[1]["messages"][-1]["content"]
    assert "tool" not in corrective.lower()
    assert "<tool_call>" not in _answer(state)


def test_react_recovered_thinking_calls_are_counted_on_every_host_surface() -> None:
    recovered = dict(VISIBLE_CALLS, metadata={"tool_calls_from_reasoning": 2})
    state, _, steps, executed, _ = _drive("react", [recovered, DIGEST])
    assert len(executed) == 2
    parse = next(d for s, d in steps if s == "parse" and d.get("has_tool_calls"))
    assert parse["tool_calls_from_reasoning"] == 2
    ptc = next(d for s, d in steps if s == "parse_tool_calls")
    assert ptc["from_reasoning"] == 2
    out = state.output or {}
    assert out["scratchpad"]["cycles"][0]["tool_calls_from_reasoning"] == 2
    assert "recovered 2 tool call(s) from the model's thinking" in out["report"]
    notice = next(n for n in out["notices"] if n["code"] == "tool_calls_from_reasoning")
    assert notice["severity"] == "info" and "2 tool call(s)" in notice["text"]


def test_react_check_plan_false_disables_both_checks() -> None:
    state, payloads, steps, _, _ = _drive("react", [ANNOUNCE], runtime_vars={"check_plan": False})
    assert "parse_reprompt" not in _names(steps) and len(payloads) == 1
    assert _answer(state) == ANNOUNCE["content"]


# ---------------------------------------------------------------------------
# CodeAct / MemAct share the pattern (a tool-free reply was final)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loop", ["codeact", "memact"])
def test_sibling_loops_reprompt_an_announcement_then_run_the_calls(loop: str) -> None:
    state, payloads, steps, executed, _ = _drive(loop, [ANNOUNCE, VISIBLE_CALLS, DIGEST])
    assert state.status == RunStatus.COMPLETED
    assert _names(steps).count("parse_reprompt") == 1
    assert payloads[1]["_runtime_observability"]["reprompted"] == REASON_ANNOUNCED
    assert any(
        m.get("role") == "assistant" and m.get("content") == ANNOUNCE["content"] for m in payloads[1]["messages"]
    )
    assert [tc["name"] for tc in executed] == ["fetch_url", "fetch_url"]
    assert ANNOUNCE["content"] not in _answer(state)


@pytest.mark.parametrize("loop", ["codeact", "memact"])
def test_sibling_loops_double_failure_is_a_visible_error(loop: str) -> None:
    state, _, steps, executed, _ = _drive(loop, [UNKNOWN_TOOL_IN_THINKING])
    assert state.status == RunStatus.COMPLETED
    assert "parse_reprompt_failed" in _names(steps)
    answer = _answer(state)
    assert answer.startswith("Error: the model's reply announced tool calls it could not run")
    assert "<tool_call>" not in answer
    assert (state.output or {})["no_tool_call_stop"]["reason"] == REASON_UNRUNNABLE
    assert executed == []


@pytest.mark.parametrize("loop", ["codeact", "memact"])
def test_sibling_loops_genuine_final_is_not_reprompted(loop: str) -> None:
    state, payloads, steps, _, _ = _drive(loop, [{"content": "Done.", "tool_calls": []}])
    assert "parse_reprompt" not in _names(steps)
    assert "Done." in _answer(state)
