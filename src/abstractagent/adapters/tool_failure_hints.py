"""Turn a tool failure into something a model can ACT on.

Why this exists (operator, 2026-08-21): the loop's only reaction to a model
banging on a broken call was a generic scolding — *"you repeated the same tool
batch, change strategy"* — plus an abstract menu of options. That is not
actionable. It never named the parameter that was wrong, never said WHY it was
wrong, and never proposed a concrete call that could work. A model cannot
self-correct from a complaint; it can self-correct from a diagnosis.

The live incident is the specification. `gpt-5.4-mini` called
`open_attachment(artifact_id="a1", handle="attachment")` twelve times. Every
answer was `Error: no attachment matches handle 'attachment' in this session.`
The one fact that would have ended it on the first try — **"a1" is a value you
invented; it appears nowhere in this conversation, and this session has no
attachments at all, so no id can work; use read_file / search_files instead** —
was never said, though every ingredient was already in the run's own state.

Everything here is derived from information the runtime already holds at the
moment of the failure: the call's actual arguments, the verbatim error, the
tool specs in the loop's own toolset, and the transcript so far. Nothing is
guessed, and a diagnosis that cannot be made honestly is not made at all
(`None`) — a confident wrong hint is worse than the error alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "diagnose_tool_failure",
    "classify_tool_error",
    "render_hint_block",
    "output_reads_as_error",
]


# A LARGE share of this framework's tool failures arrive with `success: True`
# and an error sentence in the body — `abstractcore.tools.common_tools`
# returns `f"Error: File '{path}' does not exist"` as a plain string, and the
# 2026-08-21 `analyze_media` incident was nine of exactly those. A diagnosis
# layer that trusted the transport flag alone would be blind to them.
# Deliberately anchored at the START of the text: a tool that merely mentions
# the word "error" in a long report is not failing.
_ERROR_LEAD = re.compile(r"(?is)^\s*(?:\[[^\]]{1,40}\]\s*:?\s*)?(error\b|failed\b|exception\b|traceback\b|fatal\b)")


def output_reads_as_error(text: Any) -> bool:
    """True when a result's TEXT announces a failure, whatever the flag says."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_ERROR_LEAD.match(text))


# --- error classes -------------------------------------------------------
# Ordered: the FIRST match wins, so the specific families precede the generic
# ones. Every pattern is drawn from error strings this framework actually
# emits (abstractcore common_tools, abstractruntime session_attachments /
# workspace_scoped_tools) plus the usual provider/transport wording.

_CLASS_PATTERNS: Sequence[Tuple[str, re.Pattern]] = (
    (
        "transient",
        re.compile(
            r"(?i)\b(timed?\s*out|timeout|temporar\w+|rate[- ]limit|too many requests|"
            r"unavailable|503|502|504|connection (reset|refused|error)|please retry|try again later)\b"
        ),
    ),
    (
        "denied",
        re.compile(r"(?i)\b(permission denied|not permitted|forbidden|unauthorized|access denied|not allowed|outside the workspace|blocked by policy)\b"),
    ),
    (
        # "ambiguous" ONLY in the several-candidates sense. `edit_file` says
        # "end_line 0 is ambiguous" about a single bad ARGUMENT, and telling
        # the model to "narrow it" there sends it the wrong way.
        "ambiguous",
        re.compile(r"(?i)(\bmultiple\b.{0,40}\bmatch|\bmore than one\b|\bmatches several\b)"),
    ),
    (
        "not_found",
        re.compile(
            r"(?i)(\bno (attachment|file|match|results?)\b|\bnot found\b|\bdoes not exist\b|"
            r"\bno such\b|\bcannot find\b|\bunknown (tool|id|handle)\b|\bhas no stored\b)"
        ),
    ),
    (
        "invalid_argument",
        re.compile(
            r"(?i)(\binvalid\b|\bmalformed\b|\bmust be\b|\bexpected .*(?:got|received)\b|"
            r"\bis ambiguous\b|\bvalidation\b|\brequired (?:argument|parameter|field)\b|\bunexpected keyword\b)"
        ),
    ),
    (
        "conflict",
        re.compile(r"(?i)(\balready exists\b|\bwould overwrite\b|\bnot identical\b|\bno changes\b|\bpattern .*not found\b)"),
    ),
)

# Values that carry no identity and must never be reported as "invented".
# Deliberately NOT length-gated beyond one character: the live incident's
# invented identifiers were `a1`, `a2`, `a3`, and a two-character floor would
# have skipped exactly the values that mattered.
_UNREMARKABLE_VALUES = re.compile(r"(?i)^(|true|false|none|null|nan|\d{1,4}|[a-z]|\.|/|\*|-|\.\.)$")

# Tool families: when a call fails, the honest alternative is a SIBLING that is
# actually present in this run's toolset. Never suggest a tool the model does
# not have — the whole point is that the next call must be able to succeed.
_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "read": ("read_file", "open_attachment", "skim_files", "list_files", "search_files", "analyze_media"),
    "search": ("search_files", "list_files", "grep_files", "web_search", "fetch_url"),
    "web": ("web_search", "fetch_url"),
    "write": ("write_file", "edit_file"),
    "exec": ("execute_command", "run_command"),
}


def classify_tool_error(error_text: str) -> str:
    """The failure family, from the verbatim error text. `"unknown"` when no
    pattern matches — the caller then falls back to quoting, never guessing."""
    text = str(error_text or "")
    if not text.strip():
        return "unknown"
    for name, pattern in _CLASS_PATTERNS:
        if pattern.search(text):
            return name
    return "unknown"


def _families_of(tool: str) -> List[str]:
    return [fam for fam, members in _FAMILIES.items() if tool in members]


def _sibling_tools(tool: str, available: Iterable[str]) -> List[str]:
    have = [str(t) for t in available if str(t) and str(t) != tool]
    out: List[str] = []
    for fam in _families_of(tool):
        for member in _FAMILIES[fam]:
            if member != tool and member in have and member not in out:
                out.append(member)
    return out


_DISCOVERY_NAME = re.compile(r"(?i)(search|find|list|query|grep|lookup|index|browse|glob)")


def _discovery_tools(tool: str, available: Iterable[str]) -> List[str]:
    """Tools in THIS run that can locate a target rather than address it by
    name. Recognised by name, which is a heuristic — so they are offered as
    alternatives, never asserted to solve the problem."""
    return [str(t) for t in available if str(t) != tool and _DISCOVERY_NAME.search(str(t))]


def _required_params(spec: Any) -> List[str]:
    """Required parameter names of a tool spec, best-effort across the shapes
    this framework passes around (ToolDefinition, dict, JSON-schema)."""
    if spec is None:
        return []
    req = getattr(spec, "required_args", None)
    if isinstance(spec, dict):
        req = spec.get("required_args") or spec.get("required") or req
    if isinstance(req, (list, tuple)):
        return [str(r) for r in req if str(r)]
    params = getattr(spec, "parameters", None)
    if isinstance(spec, dict):
        params = spec.get("parameters", params)
    if isinstance(params, dict):
        inner = params.get("required")
        if isinstance(inner, (list, tuple)):
            return [str(r) for r in inner if str(r)]
        return [str(k) for k in params.keys()][:3]
    return []


def _spec_for(tool: str, tool_specs: Any) -> Any:
    for spec in tool_specs or []:
        name = spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", None)
        if str(name or "") == tool:
            return spec
    return None


def _example_call(tool: str, tool_specs: Any) -> str:
    spec = _spec_for(tool, tool_specs)
    params = _required_params(spec)
    if not params:
        return f"{tool}(...)"
    return f"{tool}({', '.join(f'{p}=...' for p in params[:3])})"


def _unsourced_arguments(arguments: Any, transcript_text: str) -> List[Tuple[str, str]]:
    """String argument values that appear NOWHERE earlier in the conversation.

    This is the check that would have ended the live incident on the first
    call: `artifact_id="a1"` was not a value the run had ever been given — the
    model minted it. The runtime can prove that, and saying it out loud turns
    "no attachment matches" into "you invented this identifier".
    """
    if not isinstance(arguments, dict) or not transcript_text:
        return []
    out: List[Tuple[str, str]] = []
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        v = value.strip()
        if len(v) < 2 or _UNREMARKABLE_VALUES.match(v):
            continue
        if v not in transcript_text:
            out.append((str(key), v))
    return out


def diagnose_tool_failure(
    *,
    name: str,
    arguments: Any,
    error_text: str,
    tool_specs: Any = None,
    available_tools: Iterable[str] = (),
    transcript_text: str = "",
) -> Optional[Dict[str, str]]:
    """A specific, parameter-level explanation of ONE failed call.

    Returns `{"class", "signature", "text"}`, or `None` when nothing can be
    said beyond the error the model already read — silence beats padding.
    """
    tool = str(name or "").strip() or "tool"
    err = str(error_text or "").strip()
    klass = classify_tool_error(err)
    args = arguments if isinstance(arguments, dict) else {}
    unsourced = _unsourced_arguments(args, transcript_text)

    cause: Optional[str] = None
    fix: Optional[str] = None

    if klass == "transient":
        cause = (
            "that is a SERVICE-side failure, not a problem with your arguments — the same call "
            "is the right call, but this loop has no wait in it, so calling it again now hits "
            "the same state."
        )
        fix = (
            "Do not re-issue it here: say in your answer that the service is unavailable and "
            "what you would run once it is back."
        )
    elif klass == "denied":
        cause = "the call was refused by policy, so no retry of the SAME call can succeed."
        fix = "Reach the goal inside what you are allowed to touch, or say plainly that it is out of bounds."
    elif klass == "ambiguous":
        cause = "the arguments matched several things at once, so the tool refused to choose."
        fix = "Narrow it with the disambiguating argument the error names — do not re-send the same arguments."
    elif klass == "not_found":
        if unsourced:
            named = ", ".join(f"{k}={v!r}" for k, v in unsourced[:3])
            cause = (
                f"{named} appears NOWHERE earlier in this conversation — no tool output and no message "
                "ever gave you that value, so it cannot resolve. You supplied it from memory or by pattern."
            )
            fix = (
                "Stop guessing identifiers. Either use a value that a previous tool output actually "
                "printed, or switch to a tool that discovers the target instead of addressing it by id."
            )
        else:
            shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3]) or "(no arguments)"
            cause = f"the target named by {shown} does not exist here — re-sending the same arguments cannot change that."
            fix = "Locate the real target first, then address it with the value that lookup returns."
    elif klass == "invalid_argument":
        shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:4]) or "(no arguments)"
        cause = f"the tool rejected the ARGUMENTS themselves: you sent {shown}."
        fix = "Fix the argument the error names and call it once — an identical retry fails identically."
    elif klass == "conflict":
        cause = "the tool refused because the target's current state does not match what the call assumed."
        fix = "Re-read the target, then build the call from what it actually contains now."
    elif unsourced:
        named = ", ".join(f"{k}={v!r}" for k, v in unsourced[:3])
        cause = f"{named} appears nowhere earlier in this conversation, so it may be an invented value."
        fix = "Check where that value was supposed to come from before calling again."

    if cause is None:
        return None

    suggestions = _sibling_tools(tool, available_tools)
    if klass == "not_found":
        # "Locate the real target first" is only actionable if the model is
        # told WHICH tool can locate it. Discovery tools are recognised by
        # name and only ever named when this run actually has them.
        for extra in _discovery_tools(tool, available_tools):
            if extra not in suggestions:
                suggestions.append(extra)
    if suggestions:
        alternatives = ", ".join(_example_call(s, tool_specs) for s in suggestions[:3])
        fix = f"{fix} Available here: {alternatives}."

    call_repr = f"{tool}({', '.join(f'{k}={v!r}' for k, v in list(args.items())[:6])})" if args else f"{tool}()"
    text = f"- {call_repr}\n  failed with: {err}\n  why: {cause}\n  do instead: {fix}"

    sig_src = f"{tool}|{klass}|{json.dumps(args, sort_keys=True, default=str)}"
    try:
        signature = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    except Exception:
        signature = f"{tool}|{klass}"
    return {"class": klass, "signature": signature, "text": text}


def render_hint_block(diagnoses: Sequence[Dict[str, str]]) -> str:
    """The message the loop hands the model. Empty when there is nothing to say."""
    lines = [str(d.get("text") or "") for d in diagnoses if d and d.get("text")]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""
    head = (
        "[tool failure] A call did not work. Read the diagnosis before your next move — "
        "re-sending the same call will fail the same way."
        if len(lines) == 1
        else f"[tool failure] {len(lines)} calls did not work. Read the diagnoses before your next move — "
        "re-sending the same calls will fail the same way."
    )
    return head + "\n" + "\n".join(lines)
