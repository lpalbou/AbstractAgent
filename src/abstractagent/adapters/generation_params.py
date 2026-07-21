"""Helpers for consistent generation params in AbstractAgent adapters.

These adapters build `EffectType.LLM_CALL` payloads for AbstractRuntime. We want
to expose a uniform `(temperature, seed, thinking)` interface across agents while keeping
backward compatibility with older runs that may not have these keys in
`vars["_runtime"]`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def normalize_seed(seed: Any) -> Optional[int]:
    """Return a provider-ready seed or None when unset/random.

    Policy:
    - None or any negative value -> None (meaning: do not send seed).
    - bool values are ignored (JSON booleans are ints in Python).
    - numeric-ish values -> int(seed) if >= 0.
    """
    try:
        if seed is None or isinstance(seed, bool):
            return None
        seed_i = int(seed)
        return seed_i if seed_i >= 0 else None
    except Exception:
        return None


# The thinking values core's provider boundary ACCEPTS (BaseProvider.
# _normalize_thinking_request). Core RAISES ValueError for any other string —
# forwarding a freeform value ("verbose", "deep") would hard-fail 100% of calls
# at the provider boundary, the same collision class as the prompt_cache_binding
# incident (2026-07-11, adversary finding P2-1). Values here are canonical; a
# drift-pin test runs every member through core's real normalizer.
_THINKING_ENABLE = {"on", "true", "yes"}
_THINKING_DISABLE = {"off", "false", "no", "none"}
_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}
_THINKING_XHIGH_ALIASES = {"extra_high", "x_high"}


def normalize_thinking(value: Any) -> Any:
    """Return a provider-SAFE thinking value or None when unset/unsupported.

    Policy matches `normalize_seed`: values core would reject are DROPPED
    (None = do not send; provider defaults apply) rather than forwarded into a
    guaranteed ValueError at the provider boundary. Known strings are emitted
    in canonical form.
    """
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s or s == "auto":
        # "auto" is core's explicit auto mode; empty means unset.
        return "auto" if s == "auto" else None
    if s in _THINKING_ENABLE:
        return True
    if s in _THINKING_DISABLE:
        return False
    s_norm = "_".join(part for part in s.replace("-", " ").split() if part)
    if s_norm in _THINKING_XHIGH_ALIASES:
        return "xhigh"
    if s_norm in _THINKING_LEVELS:
        return s_norm
    return None


def resolve_max_iterations(
    limits: Any,
    scratchpad: Any,
    *,
    default: int = 20,
) -> int:
    """Resolve the iteration budget with PRESENCE-based precedence.

    `_limits.max_iterations` wins when present, then the scratchpad seed, then
    `default`. Explicit values are honored even when falsy: 0 clamps to the
    loop's floor of 1 instead of silently falling open to the default
    (agency-caps adversary finding P2-2 — the `explicit-0 or DEFAULT` class:
    an explicit narrow budget must never silently widen).

    The default is 20 — the ONE ruled framework default (maintainer c726
    "i never said 50" + c786; runtime's agent-node pin and the published
    basic-agent bundle both carry 20). This package's old 25 was a third
    value in the ecosystem, the copied-default drift class. Explicit
    workflow/host values stay authoritative (the "workflow decides"
    amendment); the 100 calls/turn failsafe ceiling is enforced upstream
    (runtime compiler / gateway config), never silently here.
    """
    for source in (limits, scratchpad):
        if isinstance(source, dict) and source.get("max_iterations") is not None:
            value = coerce_iterations(source["max_iterations"])
            if value is None:
                continue
            return value if value >= 1 else 1
    return int(default)


# Side-effect classification for the repeat guard (0030 promoted 2026-07-14).
# ONE classifier, three signals, deny-safe by construction (a misclassified
# read-only tool only SKIPS re-executing an identical already-succeeded batch):
# (1) the curated local-name set; (2) the `mcp::` name prefix — MCP tools cross
# a boundary these loops cannot classify; (3) ToolDefinition.tags, so the day
# core tags MCP/mutating tools at registration the classifier lights up with
# zero changes here (origin-READY, not origin-dependent — core's tags field
# exists today, MCP tagging does not yet).
SIDE_EFFECT_TOOL_NAMES = frozenset(
    {
        "write_file",
        "edit_file",
        "execute_command",
        # fetch_url is never read-only-safe (2026-07-12 finding: the
        # model-controlled method can POST — remote_write_capable).
        "fetch_url",
        # Shell session tools (env-gated toolset; at least as side-effectful
        # as execute_command).
        "shell_exec",
        "shell_write_stdin",
        "shell_close",
        # Comms tools (side-effectful; avoid duplicate sends). The agora hub
        # tools are wired for unattended resident agents — exactly where the
        # repeat guard matters most (wave-F P2: a repeated identical
        # agora_post_message batch is the duplicate-send class verbatim).
        "send_email",
        "send_whatsapp_message",
        "send_telegram_message",
        "send_telegram_artifact",
        "agora_post_message",
        "agora_send_dm",
        "agora_ack_inbox",
    }
)

SIDE_EFFECT_TAGS = frozenset({"mcp", "side_effect", "mutating", "write"})


def is_side_effect_tool(
    name: Any,
    *,
    tool_tags: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when a tool call may have external side effects.

    `tool_tags` is an optional {tool_name: [tags]} map (built from the logic's
    ToolDefinitions) — absent or unknown names fall back to the name signals.
    """
    n = str(name or "").strip()
    if not n:
        return False
    if n in SIDE_EFFECT_TOOL_NAMES:
        return True
    if n.startswith("mcp::"):
        return True
    if isinstance(tool_tags, dict):
        tags = tool_tags.get(n)
        if isinstance(tags, (list, tuple, set, frozenset)) and any(
            str(t).strip().lower() in SIDE_EFFECT_TAGS for t in tags
        ):
            return True
    return False


def tool_tags_map(tools: Any) -> Dict[str, Any]:
    """{name: tags} from a list of ToolDefinitions (missing tags -> ())."""
    out: Dict[str, Any] = {}
    if isinstance(tools, (list, tuple)):
        for t in tools:
            name = getattr(t, "name", None)
            if isinstance(name, str) and name.strip():
                out[name.strip()] = tuple(getattr(t, "tags", None) or ())
    return out


# Executor classification for the verifier lane (R-Type experiment, commons
# c2725/c2735/c2736 2026-07-16/17): LLM-read review blessed a crash-on-first-
# bullet game, a split-brain ReferenceError, and a corner-ninth draw — only
# EXECUTION caught all three. The verifier can already FORCE tool calls
# (next_tool_calls -> act), so the missing piece is knowing WHICH tools
# execute artifacts and preferring them. Declaration-driven by design: a tool
# opts in via ToolDefinition.tags (the same channel is_side_effect_tool
# reads) — the loop never hardcodes executor names, because executors are
# environment capabilities (browser probe, pytest, cargo) that live and ship
# tool-side, never inside this package.
EXECUTOR_TAGS = frozenset({"executor"})


def executor_tool_names(
    allow: Any,
    *,
    tool_tags: Optional[Dict[str, Any]] = None,
) -> list:
    """Allowlisted tool names declared as artifact EXECUTORS.

    A tool declares itself an executor by carrying an `executor` tag
    (EXECUTOR_TAGS) in its ToolDefinition.tags. Order follows the allowlist;
    unknown names and junk shapes are skipped silently — absence of the tag
    simply means the verifier keeps its read-only behavior (byte-identical
    prompt), so misdeclaration fails safe in both directions.
    """
    out: list = []
    if not isinstance(allow, (list, tuple)):
        return out
    tags_map = tool_tags if isinstance(tool_tags, dict) else {}
    for name in allow:
        if not isinstance(name, str):
            continue
        n = name.strip()
        if not n:
            continue
        tags = tags_map.get(n)
        if isinstance(tags, (list, tuple, set, frozenset)) and any(
            str(t).strip().lower() in EXECUTOR_TAGS for t in tags
        ):
            out.append(n)
    return out


def verifier_execution_preference(executor_names: Any) -> str:
    """Verifier-prompt block: prefer EXECUTION over reading when executors exist.

    Empty when no executor-tagged tool is allowlisted — the verifier prompt
    stays byte-identical to the pre-seam text, so deployments without
    executor tools see zero behavior change. The mapping artifact -> executor
    -> arguments deliberately stays with the verifier LLM (it sees the answer,
    the observations, and the tool schemas); a structural forcing rule would
    need exactly the artifact-type special-casing that does not generalize.
    """
    names = [str(n).strip() for n in (executor_names or []) if str(n or "").strip()] if isinstance(executor_names, (list, tuple)) else []
    if not names:
        return ""
    return (
        f"Executor tools available: {', '.join(names)}.\n"
        "These tools EXECUTE artifacts and observe ground truth (runtime errors, liveness, test results).\n"
        "If the proposed final answer claims a runnable artifact (web page, script, test suite) and the tool\n"
        "outputs contain no successful execution of that artifact, the request is NOT fully satisfied:\n"
        "return next_tool_calls invoking the matching executor on the artifact instead of judging from\n"
        "reading alone. An artifact that was never executed is not verified.\n\n"
    )


# Ordered system-prompt slot table — ONE source for all three adapters
# (design adversary P1 2026-07-13: the composition was hand-copied into three
# files, the exact divergence class the system_prompt_extra fix had just paid
# for). Adding a future slot (phase directives, workspace policy) is one row
# here, never three parallel edits. Cache contract: slot VALUES must be
# byte-stable for the run (they live in the provider prefix).
PROMPT_SLOTS: tuple = (
    ("skills_block", "Available skills:"),
    ("system_prompt_extra", "Additional system instructions:"),
)

#: Palette profile keys the delegate branches understand. Anything else in a
#: granted profile is version skew and must warn loudly (silent half-applied
#: grants are the fall-open class).
DELEGATE_SUBSTRATE_KEYS = frozenset({"provider", "model", "description"})


def compose_prompt_slots(base: str, runtime_ns: Optional[Dict[str, Any]]) -> str:
    """Compose the named `_runtime` prompt slots onto a base system prompt.

    Fixed order per PROMPT_SLOTS (skills before behavioral directives), plus
    the delegate-substrate self-description: a granted palette the model
    cannot SEE manufactures dead `substrate=` guesses (the skills-doc argument
    in reverse), so granted names + host-authored descriptions render into the
    stable prefix. Provider/model strings deliberately do NOT render — tokens
    like locale/model names in prompts have flipped behavior before; the name
    is the model's whole vocabulary, resolution stays host-side.
    """
    sys = str(base or "").rstrip()
    ns = runtime_ns if isinstance(runtime_ns, dict) else {}
    for key, header in PROMPT_SLOTS:
        raw = ns.get(key)
        if isinstance(raw, str) and raw.strip():
            sys = f"{sys}\n\n{header}\n{raw.strip()}"
    palette = ns.get("delegate_substrates")
    if isinstance(palette, dict) and palette:
        lines: list = []
        for name in sorted(str(k) for k in palette.keys()):
            prof = palette.get(name)
            desc = str(prof.get("description") or "").strip() if isinstance(prof, dict) else ""
            lines.append(f"- {name}" + (f": {desc}" if desc else ""))
        if lines:
            sys = (
                f"{sys}\n\nAvailable delegate substrates (optional `substrate` argument to delegate_agent):\n"
                + "\n".join(lines)
            )
    return sys.strip()


def context_usage_warning(
    limits: Optional[Dict[str, Any]],
    scratchpad: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """One-shot context-budget warning payload (fable5 B-F3 2026-07-13).

    The runtime accounts real token usage into `_limits.estimated_tokens_used`
    after every LLM call, and the facades seed `warn_tokens_pct` — but nothing
    ever CHECKED it: the loops sail silently past the model's window until the
    provider 400s (OpenAI-compatible) or silently truncates (Ollama without
    num_ctx — the oldest content, i.e. the system prompt, goes first). This
    helper turns the existing accounting audible: returns the warning payload
    exactly ONCE when usage crosses `warn_tokens_pct`% of `max_tokens`
    (latched in the scratchpad), None otherwise. Emitting it is the caller's
    one line; it never blocks or mutates the run beyond the latch.
    """
    if not isinstance(limits, dict) or not isinstance(scratchpad, dict):
        return None
    try:
        used = int(limits.get("estimated_tokens_used") or 0)
        ceiling = int(limits.get("max_tokens") or 0)
        pct = int(limits.get("warn_tokens_pct") or 0)
    except (TypeError, ValueError):
        return None
    if used <= 0 or ceiling <= 0 or pct <= 0:
        return None
    # Two latches (wave adversary 2026-07-13): one at the warning threshold,
    # one when usage crosses the ceiling itself — a single 80% latch followed
    # by silence through 100% under-informs exactly when it matters most.
    # NOTE: this is ACCOUNTING/OBSERVABILITY ONLY — it never trims, gates, or
    # stops anything (the no-silent-truncation / no-token-budget ADR; the
    # loops run full-context by policy). The ceiling is registry-derived via
    # the runtime config wherever available; honest scope: usage is the
    # SERVER-REPORTED input tokens of the last call, so a server that is
    # already silently truncating plateaus below the threshold (see faq.md).
    if used >= ceiling and not scratchpad.get("context_exceeded_emitted"):
        scratchpad["context_exceeded_emitted"] = True
        scratchpad["context_warning_emitted"] = True
        return {
            "estimated_tokens_used": used,
            "max_tokens": ceiling,
            "warn_tokens_pct": pct,
            "exceeded": True,
            "warning": (
                "#FALLBACK context accounting ceiling EXCEEDED: reported input usage passed the "
                "configured window — expect provider 400s (OpenAI-compatible servers) or already-"
                "active silent truncation (servers without an explicit context size). Nothing is "
                "trimmed by this loop; this is observability only."
            ),
        }
    if scratchpad.get("context_warning_emitted"):
        return None
    threshold = (ceiling * pct) // 100
    if used < threshold:
        return None
    scratchpad["context_warning_emitted"] = True
    return {
        "estimated_tokens_used": used,
        "max_tokens": ceiling,
        "warn_tokens_pct": pct,
        "exceeded": False,
        "warning": (
            "#FALLBACK context accounting warning: estimated usage crossed the warning threshold "
            "of the configured window — long transcripts may overflow the model window (provider "
            "400, or silent truncation on servers without an explicit context size). Nothing is "
            "trimmed by this loop; this is observability only."
        ),
    }


def suppress_loop_tail(runtime_ns: Any) -> bool:
    """True when the host asked the loop-position tail to stay OUT of payloads.

    c2447 incident (2026-07-15): "[loop] iteration N of M." is task-agent
    chrome — loop-position awareness for a model working a bounded task. In
    composed entity visits the adapters' merge branch lands that tail INSIDE
    the visitor's user message (BRIDGE appends the visitor's words last), so
    the entity reads it as part of what the human said. Hosts composing these
    adapters for an entity set `_runtime.suppress_loop_tail` (runtime's
    BRIDGE, spelling agreed on the incident thread) and the whole tail block
    — iteration line and [plan] render — stays out. One shared predicate for
    all three adapters (bool/int/str spellings per the tool-arg coercion
    lesson); absent/falsy = unchanged task-agent behavior.
    """
    if not isinstance(runtime_ns, dict):
        return False
    val = runtime_ns.get("suppress_loop_tail")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return False


# Drained-guidance wrappers (c2447 residue closed 2026-07-17: proposal c2792,
# runtime voice-owner sign-off c2798, semantics vocabulary adoption c2796).
# The inbox mixes sources (gateway inject_guidance, hook steering, verifier
# next_prompt lines, the loops' own retry nudges) and items carry no source
# field, so the wrapper can only claim what is true for ALL of them.
#
# VISITOR-COUPLED SCOPE (semantics pin 1): the visit spelling is true exactly
# where the visit bridge sets `_runtime.suppress_loop_tail` today. If that
# knob ever extends to a visitor-less lane (own-time hardening), "your
# visitor" would assert a visitor that does not exist — that extension needs
# a SECOND spelling ruled through the same c2792 path, never a reuse.
#
# NEVER A PARSE ANCHOR (semantics pin 2): the wrapper is for the entity's
# reading only — a visitor can type the same bytes (marker-imitation class).
# Machine detection of drained guidance keys on the durable message metadata
# (kind="operator_guidance", which deliberately does NOT rename with the
# visible string), never on the bracket prose.
#
# ACKNOWLEDGED MISNOMER (semantics c2800): host-authored items (retry nudges,
# verifier lines) ride under kind="operator_guidance" too — the key means
# "drained from the inbox", NOT "the operator said this". No audit surface
# may key on it to answer "what did the operator inject"; the deferred
# source-split's `source` field becomes the audit key when a live incident
# demands it (runtime pre-approved directionally, c2798). The key never
# renames — that would break the consumers pin 2 protects.
GUIDANCE_WRAPPER_TASK = "[Operator guidance — this amends the task; the final answer must satisfy it]"
GUIDANCE_WRAPPER_VISIT = "[A note arrived during this conversation — not from your visitor]"


def guidance_wrapper(runtime_ns: Any) -> str:
    """The transcript wrapper for guidance drained at the reason boundary.

    Task lane (default): byte-identical to the historical string. Visit lane
    (`suppress_loop_tail` truthy): the c2447-honest spelling — a host retry
    nudge must not wear operator words, and task/final-answer vocabulary is
    the chrome class the knob exists to keep out of a visit.
    """
    return GUIDANCE_WRAPPER_VISIT if suppress_loop_tail(runtime_ns) else GUIDANCE_WRAPPER_TASK


def prompt_cache_capture(response: Any) -> Optional[Dict[str, Any]]:
    """The provider's prompt-cache telemetry struct from an LLM result dict.

    0030 residue (gate lifted 2026-07-15): core's local-cache providers record
    per-call cache telemetry into `GenerateResponse.metadata["prompt_cache"]`
    (`mode`, `key`, `outcome`, `cached_tokens`, `fed_tokens`, and a
    `#FALLBACK`-prefixed `degraded_reason` when the reuse degraded); the
    runtime's llm_client folds `metadata` into every LLM result dict. This
    helper lifts the struct for the parse emit — an ADDITIVE payload key,
    present only when the provider reported one (remote providers and older
    stacks simply don't carry it). Returns a copy or None; never raises.
    """
    if not isinstance(response, dict):
        return None
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return None
    struct = metadata.get("prompt_cache")
    if not isinstance(struct, dict) or not struct:
        return None
    return dict(struct)


def verifier_response_schema() -> Dict[str, Any]:
    """Verifier JSON schema shared by the ReAct and CodeAct review nodes.

    STRICT-MODE EXPRESSIBLE BY CONSTRUCTION (airelay 422 incident, 2026-07-15):
    OpenAI-strict validators (and subscription relays in front of them) require
    every object node to declare `properties`, a `required` array listing every
    key, and `additionalProperties: false`. A free-form dict — a bare
    `{"type": "object"}` without `properties` — violates those rules by
    construction and gets the WHOLE request refused with a deterministic 4xx.

    `next_tool_calls[].arguments` is therefore a JSON-ENCODED STRING, exactly
    like OpenAI's own function-calling wire format (which encodes tool
    arguments as a JSON string for the same reason). Parse sites accept both
    this string shape and the legacy object shape via
    `coerce_verifier_tool_arguments` (lenient providers and older transcripts
    may still carry objects).
    """
    return {
        "type": "object",
        "properties": {
            "complete": {"type": "boolean"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "next_prompt": {"type": "string"},
            "next_tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {
                            "type": "string",
                            "description": (
                                "Tool arguments as a JSON-encoded object string, "
                                "e.g. \"{\\\"path\\\": \\\"notes.txt\\\"}\". "
                                "Use \"{}\" when the tool takes no arguments."
                            ),
                        },
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["complete", "missing", "next_prompt", "next_tool_calls"],
        "additionalProperties": False,
    }


def coerce_verifier_tool_arguments(args: Any) -> Dict[str, Any]:
    """Normalize a verifier-proposed tool-call `arguments` value to a dict.

    The verifier schema declares `arguments` as a JSON-encoded string (see
    `verifier_response_schema`), but parse sites stay liberal: dicts pass
    through unchanged (legacy shape; lenient providers that ignored the
    schema), JSON-object strings are decoded, and anything else — including
    JSON that decodes to a non-object — degrades to `{}` exactly like the
    previous non-dict handling did.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def coerce_iterations(raw: Any) -> Optional[int]:
    """Coerce an iteration budget that may arrive as a non-int (tool-call arg
    coercion class: several tool-call formats preserve raw strings, so budgets
    arrive as "25", "8.5", or floats). `int("8.5")` raises — before this
    helper, a string-float EXPLICIT budget silently fell open to the default
    (the explicit-narrow-budget-must-never-silently-widen class, again).
    Booleans are refused (True would coerce to a 1-iteration child — never an
    operator's intent). Returns None when unparseable; callers fall back
    loudly or by their documented default."""
    if isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(raw))
    except (TypeError, ValueError, OverflowError):
        return None


def runtime_llm_params(
    runtime_ns: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
    default_temperature: float = 0.7,
) -> Dict[str, Any]:
    """Merge `runtime_ns` sampling controls into an LLM_CALL params dict.

    Precedence:
    1) `runtime_ns.temperature` / `runtime_ns.seed` when present
    2) `extra.temperature` / `extra.seed` (step-specific defaults)
    3) `default_temperature` (only for temperature)

    `thinking` is copied from `extra.thinking` when explicitly set, otherwise
    from `runtime_ns.thinking`. Empty strings are treated as unset.
    """
    out: Dict[str, Any] = dict(extra or {})

    # Temperature: always provide a float (provider-agnostic).
    temp_val = runtime_ns.get("temperature") if isinstance(runtime_ns, dict) else None
    if temp_val is None:
        temp_val = out.get("temperature")
    if temp_val is None:
        temp_val = default_temperature
    try:
        out["temperature"] = float(temp_val)
    except Exception:
        out["temperature"] = float(default_temperature)

    # Seed: only include when explicitly set (>= 0).
    seed_val = runtime_ns.get("seed") if isinstance(runtime_ns, dict) else None
    if seed_val is None:
        seed_val = out.get("seed")
    seed_norm = normalize_seed(seed_val)
    if seed_norm is not None:
        out["seed"] = seed_norm
    else:
        out.pop("seed", None)

    thinking_val = out.get("thinking")
    thinking_norm = normalize_thinking(thinking_val)
    if thinking_norm is None and isinstance(runtime_ns, dict):
        thinking_norm = normalize_thinking(runtime_ns.get("thinking"))
    if thinking_norm is not None:
        out["thinking"] = thinking_norm
    else:
        out.pop("thinking", None)

    # Pass-through media policies (runtime-owned defaults).
    #
    # This keeps thin clients simple: they can set `_runtime.audio_policy` (and
    # optional language hints) once at run start, and all LLM_CALL steps inherit it.
    if isinstance(runtime_ns, dict):
        audio_policy = runtime_ns.get("audio_policy")
        if "audio_policy" not in out and isinstance(audio_policy, str) and audio_policy.strip():
            out["audio_policy"] = audio_policy.strip()

        stt_language = runtime_ns.get("stt_language")
        if stt_language is None:
            stt_language = runtime_ns.get("audio_language")
        if "stt_language" not in out and isinstance(stt_language, str) and stt_language.strip():
            out["stt_language"] = stt_language.strip()

        binding = runtime_ns.get("prompt_cache_binding")
        if "prompt_cache_binding" not in out:
            if isinstance(binding, dict) and binding:
                out["prompt_cache_binding"] = dict(binding)
            elif isinstance(binding, str) and binding.strip():
                out["prompt_cache_binding"] = binding.strip()

        cache_key = runtime_ns.get("prompt_cache_key")
        if "prompt_cache_key" not in out and isinstance(cache_key, str) and cache_key.strip():
            out["prompt_cache_key"] = cache_key.strip()

        # Streaming passthrough (code seat c1007): `_runtime.stream = True`
        # reaches the LLM call as params.stream so same-process hosts get
        # runtime's on_token deltas (d3f6f87). STRICT bool True only — the
        # tool-args lesson (2026-02-20): strings like "false"/"0" are truthy
        # in Python, so anything but `True` is treated as unset (absent =
        # provider default, no key emitted; never a truthy-string fall-open).
        if "stream" not in out and runtime_ns.get("stream") is True:
            out["stream"] = True

    # Vocabulary collision guard (live-proven 2026-07-11, agency c509): core's
    # `prompt_cache_binding` is the STRICT durable-bloc artifact binding — a dict
    # with binding meta whose validation unconditionally raises without a loaded
    # bloc — while a bare STRING here has always meant per-session cache identity,
    # which in core's vocabulary is `prompt_cache_key` (best-effort). Emitting the
    # string under the strict name failed 100% of live calls at the provider
    # boundary (the entity visit lane's turn-1 failure). Strings therefore ride
    # `prompt_cache_key`; only dict bindings keep the strict name.
    bound = out.get("prompt_cache_binding")
    if isinstance(bound, str):
        out.pop("prompt_cache_binding", None)
        key = bound.strip()
        if key and "prompt_cache_key" not in out:
            out["prompt_cache_key"] = key

    return out
