"""Helpers for consistent generation params in AbstractAgent adapters.

These adapters build `EffectType.LLM_CALL` payloads for AbstractRuntime. We want
to expose a uniform `(temperature, seed, thinking)` interface across agents while keeping
backward compatibility with older runs that may not have these keys in
`vars["_runtime"]`.
"""

from __future__ import annotations

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
            try:
                value = int(source["max_iterations"])
            except (TypeError, ValueError):
                continue
            return value if value >= 1 else 1
    return int(default)


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
