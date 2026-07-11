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
def test_runtime_llm_params_keeps_explicit_thinking_override() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params({"thinking": "high"}, extra={"thinking": "low", "temperature": 0.2})

    assert out["thinking"] == "low"


@pytest.mark.basic
def test_runtime_llm_params_forwards_boolean_thinking() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params({"thinking": False}, extra={"temperature": 0.2})

    assert out["thinking"] is False
