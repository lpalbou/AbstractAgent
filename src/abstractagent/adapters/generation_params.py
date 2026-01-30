"""Helpers for consistent generation params in AbstractAgent adapters.

These adapters build `EffectType.LLM_CALL` payloads for AbstractRuntime. We want
to expose a uniform `(temperature, seed)` interface across agents while keeping
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

    return out
