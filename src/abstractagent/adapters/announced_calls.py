"""Replies that announce tool use but carry no runnable call (mission AGX, 2026-09-26).

Two shapes end a loop step with NO tool call while the model clearly meant to
act. Before this module both became the run's final answer:

1. ANNOUNCED — a short intent sentence and nothing else. Evidence
   (untracked/missions-2026-09-25/XP/REPORT.md, Qwen3.8-27B on MLX): at
   iteration 3 the model wrote "I have strong material. Let me verify a couple
   of key specifics (oil price level, gold, ...) before writing the digest."
   and stopped, 5/5 gateway runs and 15/15 provider-level generations. The run
   "completed" with that sentence as its answer and no digest.
2. UNRUNNABLE — tool-call markup the provider could not turn into a call: a
   tool name that was not offered, an envelope cut off mid-parameter, calls
   drafted inside the thinking block that abstractcore >= 2.16.0 could not
   recover (it recovers the clean trailing ones itself and reports the rest in
   `metadata.warnings` / `metadata.unparsed_tool_call`). Published as an
   answer, the user sees raw markup (backlog 0918).

The fix is ONE re-prompt that shows the model its own reply VERBATIM. The XP
evidence is specific about the verbatim part: re-prompting with the raw reply
made the model re-issue the same calls as visible calls 5/5; re-prompting with
the reply as the runtime records it (content "") made it believe the tools had
already run, 5/5. A second failure ends the step with a visible error — the
announcement or the markup is never published as the answer.

Pure functions only; each loop adapter owns its own routing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .transcripts import extract_reasoning_text

__all__ = [
    "NOT_AN_ANSWER_KINDS",
    "REASON_ANNOUNCED",
    "REASON_UNRUNNABLE",
    "classify_no_call_reply",
    "has_negated_intent",
    "is_not_an_answer",
    "resolve_checks",
    "looks_like_tool_announcement",
    "no_call_error_text",
    "no_call_stop_reason",
    "quote_reply",
    "reprompt_text",
    "tool_calls_from_reasoning",
    "verbatim_reply",
    "visible_content",
]

REASON_ANNOUNCED = "announced_tool_use"
REASON_UNRUNNABLE = "unrunnable_tool_call"

# Replies at or above this many prose characters are never judged
# "announcement-only": a real answer is longer than one intent sentence, and a
# false positive costs a model turn and, twice, the answer itself.
ANNOUNCEMENT_MAX_CHARS = 300

_FENCED_BLOCK_RE = re.compile(r"(?s)```[^\n`]*\n.*?```")
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?(</think>|$)")

# Openers of the tool-call envelopes the prompted parsers understand (mirrors
# abstractcore's `_TOOL_ENVELOPE_OPENER_RE`), plus Qwen's bare `<function=`.
_MARKUP_RE = re.compile(
    r"(?i)<tool_call\b|<\|tool_call\|>|<\|tool_call>|<\|tool_call_start\|>|<function_call>|```tool_code|<function=[\w.:-]+>"
)
_MARKUP_STRIP_RE = re.compile(
    r"(?is)"
    r"<tool_call\b.*?(</tool_call>|$)|"
    r"<\|tool_call\|>.*?(<\|/tool_call\|>|$)|"
    r"<\|tool_call>.*?(<tool_call\|>|$)|"
    r"<\|tool_call_start\|>.*?(<\|tool_call_end\|>|$)|"
    r"<function_call>.*?(</function_call>|$)|"
    r"```tool_code.*?(```|$)|"
    r"<function=[\w.:-]+>.*?(</function>|$)"
)

# abstractcore's warning sentences for tool syntax that produced no call
# (providers/base.py `_warn_unrecognized_tool_syntax`,
# `_recover_tool_calls_from_reasoning`; streaming.py unparsed envelopes).
_TOOL_WARNING_RE = re.compile(r"(?i)^\s*(tool call not recognized|tool-call syntax|unparsed tool call)")
_UNKNOWN_NAMES_RE = re.compile(r"the model called ((?:'[^']*'(?:, )?)+)")

# First-person, forward-looking intent. The apostrophe is REQUIRED in "I'll":
# the optional-apostrophe spelling matches the word "ill".
_INTENT_RE = re.compile(
    r"(?i)\b(let me|let['’]s|i will|i['’]ll|i am going to|i['’]m going to|i need to|i should|next,? i|now i)\b"
)
_NEGATED_INTENT_RE = re.compile(
    r"(?i)\b(i will not|i won['’]t|i['’]ll not|i cannot|i can['’]t|i do not need to|i don['’]t need to|"
    r"i need to stop|i should not|i shouldn['’]t|no need to)\b"
)
# Verbs that name work a tool would do. Broader than the long-reply heuristic
# in react_runtime (`_DEFERRED_ACTION_VERB_RE`) because it only applies to
# SHORT replies: every XP failure used one of verify/get/gather/confirm/fill/
# pull/do ... pass/check, none of which the old list knew.
_TOOL_VERB_RE = re.compile(
    r"(?i)\b("
    r"read|open|search|list|skim|inspect|explore|scan|run|execute|edit|fetch|download|creat(?:e|ing)|"
    r"verify|check|double-check|confirm|validate|get|gather|grab|pull|look(?:\s+up|\s+into|\s+at)?|find|"
    r"research|dig|fill|collect|query|call|test|browse|visit|retrieve|load|review|investigate|grep|"
    r"examine|analy[sz]e|compare|write|save|update|modify|patch|install|build|compile|apply|"
    r"do (?:one|a|another|two|some)"
    r")\b"
)
# The model waiting on the user is a legitimate final reply.
_WAITING_RE = re.compile(
    r"(?i)\b(let me know|what would you like|would you like|do you want|tell me|shall i|should i|"
    r"i['’]ll wait|i will wait|waiting for|if you (?:want|need|like|prefer)|"
    r"once you|when you|after you|unless you|until you|if you say|your (?:go|confirmation|approval))\b"
)
# Intent followed by a verb that TALKS rather than acts: "Let me summarize:",
# "Next, I recommend ...". Never an announcement of tool work.
_NON_ACTION_AFTER_INTENT_RE = re.compile(
    r"(?i)\b(?:let me|let['’]s|i will|i['’]ll|i am going to|i['’]m going to|i should|next,? i|now i)\s+"
    r"(?:also\s+|briefly\s+|just\s+)?"
    r"(?:summari[sz]e|explain|recommend|suggest|mention|note|clarify|add|point out|conclude|answer|say|"
    r"be (?:clear|brief|honest)|start by saying|wrap up)\b"
)
# Present-progressive or elliptical action statements with no first person:
# "Checking the remaining two sources now.", "Proceeding to fetch ...",
# "One more search to confirm ..., then the digest.", "First, a quick search for ...".
_ACTION_STATEMENT_RE = re.compile(
    r"(?i)^(?:"
    r"proceeding to \w+|"
    r"(?:checking|fetching|searching|looking up|reading|running|verifying|pulling|gathering|grabbing|opening|"
    r"loading|querying|downloading|inspecting|scanning|grepping|confirming|retrieving)\b.*\b(?:now|next|then|first)\b|"
    r"(?:one|two|a few) more (?:search|searches|check|checks|lookup|lookups|fetch|fetches|pass|read|reads)\b|"
    r"first,? (?:a|one) (?:quick )?(?:search|check|lookup|fetch|pass|read)\b|"
    r"i need (?:a bit |a little |some )?more (?:data|info|information|detail|details|context|evidence|sources)\b"
    r")"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!])\s+")
# A reply that STARTS by delivering is a final answer, whatever intent words follow.
_DELIVERY_START_RE = re.compile(
    r"(?i)^\s*(done|finished|completed|all set|all done|here (?:is|are)|here['’]s|final answer|in summary|"
    r"to summarize|summary)\b"
)
_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+\S|\*\*\S)")


# Transcript message kinds that hold a FAILED reply. Budget terminals that pick
# "the agent's last words" from the transcript must skip them — they are the
# replies this module exists to keep from being published.
NOT_AN_ANSWER_KINDS = frozenset({"reprompted_reply", "reprompt_failed_reply"})


def is_not_an_answer(message: Any) -> bool:
    meta = message.get("metadata") if isinstance(message, dict) else None
    return isinstance(meta, dict) and meta.get("kind") in NOT_AN_ANSWER_KINDS


def _flag(raw: Any) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def resolve_checks(runtime_ns: Any, *, suppressed: bool) -> Tuple[bool, bool]:
    """(check_announcement, check_unrunnable), two independent switches.

    - `_runtime.check_plan` controls ONLY the announcement heuristic (a guess
      about intent): absent = ON in task lanes, OFF in visit lanes, where
      "I will read that entry again" is musing.
    - `_runtime.check_unrunnable_calls` controls the unrunnable-markup check
      (a fact about the reply): absent = ON in every lane. Turning the
      heuristic off must never re-enable publishing raw tool markup.
    """
    ns = runtime_ns if isinstance(runtime_ns, dict) else {}
    raw_plan = ns.get("check_plan")
    check_announcement = (not suppressed) if raw_plan is None else _flag(raw_plan)
    raw_unrunnable = ns.get("check_unrunnable_calls")
    check_unrunnable = True if raw_unrunnable is None else _flag(raw_unrunnable)
    return check_announcement, check_unrunnable


def _prose(text: Any) -> str:
    s = str(text or "")
    s = _FENCED_BLOCK_RE.sub("", s)
    return s.strip()


def _strip_markup(text: str) -> str:
    try:
        return _MARKUP_STRIP_RE.sub("", text or "").strip()
    except Exception:
        return str(text or "").strip()


def _metadata(response: Any) -> Dict[str, Any]:
    meta = response.get("metadata") if isinstance(response, dict) else None
    return meta if isinstance(meta, dict) else {}


def tool_calls_from_reasoning(response: Any) -> int:
    """How many of this reply's tool calls abstractcore recovered from the thinking block."""
    try:
        n = int(_metadata(response).get("tool_calls_from_reasoning") or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def has_negated_intent(text: Any) -> bool:
    """"I will not run that", "I can't ...": a refusal, never an announcement."""
    return bool(_NEGATED_INTENT_RE.search(_prose(text)))


def _last_sentence(prose: str) -> str:
    lines = [ln.strip() for ln in prose.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else prose
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(last_line) if p.strip()]
    return parts[-1] if parts else last_line


def looks_like_tool_announcement(text: Any) -> bool:
    """True when a SHORT reply ENDS on an announcement of tool work.

    Conservative on purpose — a false positive costs a model turn and, twice,
    the answer. It fires only when the prose (fences and markup removed) is
    shorter than `ANNOUNCEMENT_MAX_CHARS` and its LAST sentence is the
    announcement: first-person intent followed by a tool verb ("Let me verify
    ..."), or an action statement ("Checking the remaining two sources now.").
    Every XP failure ends that way. Never an announcement: questions,
    replies waiting on the user ("... unless you say otherwise", "once you
    confirm"), intent that talks rather than acts ("Let me summarize: ..."),
    refusals ("I will not run that"), replies that start by delivering
    ("Done.", "Here is ..."), structured answers. English only (documented
    limit: docs/agents.md, backlog 0033).
    """
    prose = _strip_markup(_prose(text))
    if not prose or len(prose) >= ANNOUNCEMENT_MAX_CHARS:
        return False
    if prose.rstrip().endswith("?"):
        return False
    if _WAITING_RE.search(prose) or _DELIVERY_START_RE.search(prose) or _HEADING_RE.search(prose):
        return False
    if _NEGATED_INTENT_RE.search(prose):
        return False
    last = _last_sentence(prose)
    if _ACTION_STATEMENT_RE.search(last):
        return True
    intent = _INTENT_RE.search(last)
    if not intent:
        return False
    if _NON_ACTION_AFTER_INTENT_RE.search(last):
        return False
    return bool(_TOOL_VERB_RE.search(last[intent.end():]))


def _unrunnable_detail(response: Any, content: str) -> Optional[str]:
    """Name the tool syntax that produced no call, or None when there is none."""
    meta = _metadata(response)
    unparsed = meta.get("unparsed_tool_call")
    if unparsed:
        reason = unparsed.get("reason") if isinstance(unparsed, dict) else None
        return f"the tool call was cut off or malformed ({reason})" if reason else "the tool call was cut off or malformed"
    warnings = meta.get("warnings")
    if isinstance(warnings, list):
        for w in warnings:
            if not isinstance(w, str) or not _TOOL_WARNING_RE.search(w):
                continue
            names = _UNKNOWN_NAMES_RE.search(w)
            if names:
                return f"the model asked for {names.group(1)}, which is not an available tool"
            if "inside the model's reasoning" in w:
                return "the tool calls were written inside the model's thinking, where they cannot run"
            return "the tool-call syntax could not be parsed"
    raw_content = str(response.get("content") or "") if isinstance(response, dict) else content
    if _MARKUP_RE.search(_prose(_THINK_BLOCK_RE.sub("", raw_content))):
        return "the reply contains tool-call markup that did not become a tool call"
    # The thinking block only counts when there is no visible reply: a model
    # that considered a call while thinking and then answered has answered.
    if not _THINK_BLOCK_RE.sub("", raw_content).strip():
        for text in (raw_content, content, extract_reasoning_text(response)):
            if _MARKUP_RE.search(_prose(text)):
                return "the tool calls were written inside the model's thinking, where they cannot run"
    return None


def classify_no_call_reply(
    response: Any,
    content: Any,
    *,
    tools_offered: bool,
    check_announcement: bool = True,
    check_unrunnable: bool = True,
) -> Tuple[str, str]:
    """Classify a reply that carried NO tool call: (reason, detail) or ("", "").

    `content` is the loop's parsed content (which may already be the reasoning
    fallback when the visible content was empty). The caller must only ask
    about replies whose tool-call list is empty.
    """
    text = str(content or "")
    raw_content = str(response.get("content") or "") if isinstance(response, dict) else text
    visible = _strip_markup(_prose(_THINK_BLOCK_RE.sub("", raw_content)))

    if check_unrunnable:
        detail = _unrunnable_detail(response, text)
        # Markup/warnings with no visible answer, or with only a short one: the
        # reply is the failed call. A long visible answer that merely also
        # carries markup (docs about tool syntax, say) stays an answer.
        if detail and (not visible or len(visible) < ANNOUNCEMENT_MAX_CHARS):
            if not tools_offered:
                detail = f"{detail}; no tools are available in this step"
            return REASON_UNRUNNABLE, detail

    if check_announcement and tools_offered and looks_like_tool_announcement(text):
        return REASON_ANNOUNCED, "the reply announced tool use but contained no tool call"
    return "", ""


def visible_content(response: Any) -> str:
    """The reply's visible content exactly as the provider returned it (may be "")."""
    return str(response.get("content") or "") if isinstance(response, dict) else ""


def verbatim_reply(response: Any, *, fallback: str = "") -> str:
    """The assistant's reply as the provider returned it: reasoning AND content.

    Never the emptied record: the runtime's recorded turn for a think-held
    call is `content: ""`, and a model shown that believes its tools ran.
    Reasoning is re-wrapped in `<think>` (the tag every thinking family this
    framework serves uses); an unparsed envelope that the provider reported in
    metadata instead of content is appended so the model sees it too.
    """
    reasoning = extract_reasoning_text(response).strip()
    content = str(response.get("content") or "").strip() if isinstance(response, dict) else ""
    parts: List[str] = []
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if content:
        parts.append(content)
    unparsed = _metadata(response).get("unparsed_tool_call")
    if isinstance(unparsed, dict):
        utext = str(unparsed.get("text") or "").strip()
        if utext and utext not in reasoning and utext not in content:
            parts.append(utext)
    out = "\n\n".join(parts).strip()
    return out or str(fallback or "").strip()


def quote_reply(reply: str) -> str:
    """Fence `reply` so it survives as quoted DATA inside a user message."""
    longest = max((len(m) for m in re.findall(r"`+", reply or "")), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{reply}\n{fence}"


def reprompt_text(
    reason: str,
    *,
    tools_offered: bool,
    suppressed: bool = False,
    reply: str = "",
) -> str:
    """The corrective user message, with the failed reply QUOTED inside it.

    The reply rides here, not (only) in the assistant history turn: Qwen3.5 /
    Qwen3.6 chat templates strip `<think>…</think>` from every assistant turn
    before the last user message, so a think-held call carried by the history
    turn reaches the model as an EMPTY reply — the XP "as recorded" case that
    made the model believe its tools had run (5/5). User content is rendered
    verbatim by every template.
    """
    quoted = f"Your previous reply was, verbatim:\n{quote_reply(reply)}\n\n" if reply else ""
    if suppressed and reason == REASON_UNRUNNABLE:
        # Visit lanes: host-voiced, no loop vocabulary (c2447 lane honesty).
        # (The announcement check only runs there on an EXPLICIT
        # `check_plan=true`, which asks for the task-lane wording.)
        quoted = f"This is what came through:\n{quote_reply(reply)}\n\n" if reply else ""
        return quoted + (
            "It didn't come through as words — part of it was in a form that can't be carried "
            "out here, so nothing happened. Say it again: do what you meant to do now, or "
            "reply in plain words."
        )
    if not tools_offered:
        return quoted + (
            "That reply contained tool-call markup, but no tools are available here, so nothing "
            "ran. Reply again with a direct answer, without tool-call markup."
        )
    if reason == REASON_UNRUNNABLE:
        return quoted + (
            "That reply placed tool calls where they cannot run (inside your thinking, with an "
            "unknown tool name, or cut off), so none ran. Reply again: call the tools now in the "
            "required format as your visible reply, or answer directly."
        )
    return quoted + (
        "That reply announced tool calls but none ran: it contained no tool call. Reply again: "
        "call the tools now in the required format as your visible reply, or answer directly."
    )


def no_call_error_text(reason: str, detail: str, *, reprompted: bool = True) -> str:
    """The visible error that ends a step whose no-call reply could not be recovered."""
    what = (
        "tool calls it could not run"
        if reason == REASON_UNRUNNABLE
        else "tool use without calling any tool"
    )
    tail = f" ({detail})" if detail else ""
    how = (
        "and it did the same again after one re-prompt"
        if reprompted
        else "and no iteration was left to re-prompt it"
    )
    return (
        f"Error: the model's reply announced {what}{tail}, {how}. "
        "Nothing ran, and the reply was not used as an answer."
    )


def no_call_stop_reason(no_call: Dict[str, Any], *, iterations: int) -> Dict[str, Any]:
    """Host-facing `stop_reason` for a turn ended by an unrecovered no-call reply.

    One wording for all three loops (hosts render it; they never re-derive it).
    """
    unrunnable = str(no_call.get("reason") or "") == REASON_UNRUNNABLE
    reprompted = bool(no_call.get("reprompted", True))
    detail = str(no_call.get("detail") or "").strip()
    iters_txt = f" after {iterations} iterations" if iterations > 0 else ""
    what = "wrote tool calls that could not run" if unrunnable else "announced tool use without calling any tool"
    again = (
        ", and did the same after one re-prompt, so nothing ran."
        if reprompted
        else ", with no iteration left to re-prompt it, so nothing ran."
    )
    return {
        "code": "no_tool_call",
        "finished": False,
        "budget_exhausted": False,
        "iterations": iterations,
        "label": (
            f"stopped: tool call could not run{iters_txt}"
            if unrunnable
            else f"stopped: announced tools, made no call{iters_txt}"
        ),
        "headline": (
            f"The agent stopped early{iters_txt}: the model {what}"
            + (f" ({detail})" if detail else "")
            + again
        ),
        "remedy": (
            "Check that the tool the model asked for is enabled for this agent, then retry."
            if unrunnable
            else "Retry; if it recurs, use another model or turn thinking off for this agent "
            "(the model announces calls it does not emit)."
        ),
    }
