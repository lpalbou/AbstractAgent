import pytest


@pytest.mark.basic
def test_runtime_llm_params_includes_audio_policy_and_stt_language_from_runtime_ns() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"audio_policy": "auto", "stt_language": "fr"},
        extra={"temperature": 0.2},
    )

    assert out["audio_policy"] == "auto"
    assert out["stt_language"] == "fr"


@pytest.mark.basic
def test_runtime_llm_params_does_not_override_explicit_policy_in_extra() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"audio_policy": "auto", "stt_language": "fr"},
        extra={"audio_policy": "native_only", "stt_language": "en"},
    )

    assert out["audio_policy"] == "native_only"
    assert out["stt_language"] == "en"


@pytest.mark.basic
def test_runtime_llm_params_forwards_prompt_cache_binding_from_runtime_ns() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    binding = {"binding_id": "bind-1", "key": "work:orbit"}

    out = runtime_llm_params({"prompt_cache_binding": binding}, extra={"temperature": 0.2})

    assert out["prompt_cache_binding"] == binding
    assert out["prompt_cache_binding"] is not binding


@pytest.mark.basic
def test_runtime_llm_params_string_binding_rides_prompt_cache_key() -> None:
    """Vocabulary-collision guard (agency c509, live turn-1 failure): core's
    `prompt_cache_binding` is the STRICT durable-bloc dict; a bare string means
    per-session cache identity, which is core's `prompt_cache_key`. Strings must
    NEVER reach the wire under the strict name — they failed 100% of live calls."""
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"prompt_cache_binding": "entity:voyager|visit-abc"},
        extra={"temperature": 0.2},
    )

    assert "prompt_cache_binding" not in out
    assert out["prompt_cache_key"] == "entity:voyager|visit-abc"


@pytest.mark.basic
def test_runtime_llm_params_explicit_string_binding_override_also_converts() -> None:
    """The guard applies at the OUTPUT boundary: even an explicit extra override
    that is a bare string converts to prompt_cache_key (the string shape is the
    trap regardless of which caller supplied it)."""
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"prompt_cache_binding": {"binding_id": "runtime"}},
        extra={"prompt_cache_binding": "explicit-session-key"},
    )

    assert "prompt_cache_binding" not in out
    assert out["prompt_cache_key"] == "explicit-session-key"


@pytest.mark.basic
def test_runtime_llm_params_forwards_prompt_cache_key_from_runtime_ns() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"prompt_cache_key": " session-42 "},
        extra={"temperature": 0.2},
    )

    assert out["prompt_cache_key"] == "session-42"


@pytest.mark.basic
def test_runtime_llm_params_key_does_not_clobber_explicit_extra_key() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"prompt_cache_key": "runtime-key", "prompt_cache_binding": "runtime-binding"},
        extra={"prompt_cache_key": "explicit-key"},
    )

    # The explicit key wins; the string binding never overwrites it and never
    # survives under the strict name.
    assert out["prompt_cache_key"] == "explicit-key"
    assert "prompt_cache_binding" not in out


@pytest.mark.basic
def test_runtime_llm_params_forwards_thinking_from_runtime_ns() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params({"thinking": " high "}, extra={"temperature": 0.2})

    assert out["thinking"] == "high"


@pytest.mark.basic
def test_normalize_thinking_drops_freeform_strings_core_would_reject() -> None:
    """Adversary P2-1 (2026-07-11): core RAISES ValueError for non-enum thinking
    strings — forwarding freeform values would hard-fail 100% of calls at the
    provider boundary (the prompt_cache_binding collision class). Unknown
    strings are DROPPED (None = do not send), like normalize_seed."""
    from abstractagent.adapters.generation_params import normalize_thinking

    for junk in ("verbose", "deep", "reasoning", "max", "ultra high plus"):
        assert normalize_thinking(junk) is None, junk

    # Known values pass, canonicalized.
    assert normalize_thinking(True) is True
    assert normalize_thinking(False) is False
    assert normalize_thinking(" AUTO ") == "auto"
    assert normalize_thinking("on") is True
    assert normalize_thinking("Off") is False
    assert normalize_thinking("none") is False
    assert normalize_thinking("Extra High") == "xhigh"
    assert normalize_thinking("x-high") == "xhigh"
    assert normalize_thinking("MEDIUM") == "medium"


@pytest.mark.basic
def test_normalize_thinking_output_always_accepted_by_core() -> None:
    """Drift pin: every value this adapter emits must pass core's REAL
    normalizer without raising. If core widens/narrows its enum, this test
    names the drift (the diary_type-clamp lesson applied to params)."""
    from abstractagent.adapters.generation_params import normalize_thinking
    from abstractcore.providers.base import BaseProvider

    candidates = [
        True, False, "auto", "on", "off", "none", "true", "no",
        "minimal", "low", "medium", "high", "xhigh",
        "extra high", "extra-high", "x_high", "verbose", "deep", "", "  ",
    ]
    for candidate in candidates:
        emitted = normalize_thinking(candidate)
        if emitted is None:
            continue  # dropped values never reach core
        # Must never raise.
        BaseProvider._normalize_thinking_request(emitted)


@pytest.mark.basic
def test_stream_passthrough_strict_bool_only() -> None:
    """code c1007: `_runtime.stream = True` rides params so on_token deltas
    fire end-to-end. STRICT bool — truthy strings ("true", "1") are the
    tool-args coercion class and must NOT enable streaming; absent/False
    emits no key (provider default)."""
    from abstractagent.adapters.generation_params import runtime_llm_params

    assert runtime_llm_params({"stream": True})["stream"] is True
    for not_a_stream in (False, None, "true", "false", "1", 1, 0, [], {}):
        out = runtime_llm_params({"stream": not_a_stream})
        assert "stream" not in out, repr(not_a_stream)
    # Absent key entirely: no stream key emitted.
    assert "stream" not in runtime_llm_params({})
    # extra wins / is preserved when a step sets it explicitly.
    assert runtime_llm_params({}, extra={"stream": True})["stream"] is True


@pytest.mark.basic
def test_resolve_max_iterations_explicit_zero_clamps_never_falls_open() -> None:
    """Adversary P2-2 (agency-caps class): an explicit narrow budget must never
    silently widen. `0 or DEFAULT` treated explicit 0 as unspecified → default."""
    from abstractagent.adapters.generation_params import resolve_max_iterations

    # Explicit 0 in limits clamps to the loop floor 1 — never the default.
    assert resolve_max_iterations({"max_iterations": 0}, {"max_iterations": 20}) == 1
    # Presence precedence: limits wins over scratchpad.
    assert resolve_max_iterations({"max_iterations": 3}, {"max_iterations": 20}) == 3
    # Absent in limits -> scratchpad seed.
    assert resolve_max_iterations({}, {"max_iterations": 7}) == 7
    # Absent everywhere -> the ONE ruled framework default (c726/c786: 20).
    assert resolve_max_iterations({}, {}) == 20
    assert resolve_max_iterations(None, None) == 20
    # Unparseable explicit value skips to the next source, never crashes.
    assert resolve_max_iterations({"max_iterations": "abc"}, {"max_iterations": 4}) == 4


@pytest.mark.basic
def test_runtime_llm_params_keeps_explicit_thinking_override() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params({"thinking": "high"}, extra={"thinking": "low", "temperature": 0.2})

    assert out["thinking"] == "low"


@pytest.mark.basic
def test_runtime_llm_params_forwards_boolean_thinking() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params({"thinking": False}, extra={"temperature": 0.2})

    assert out["thinking"] is False
