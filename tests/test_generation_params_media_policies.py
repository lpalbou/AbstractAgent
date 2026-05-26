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
def test_runtime_llm_params_keeps_explicit_prompt_cache_binding_override() -> None:
    from abstractagent.adapters.generation_params import runtime_llm_params

    out = runtime_llm_params(
        {"prompt_cache_binding": {"binding_id": "runtime"}},
        extra={"prompt_cache_binding": "explicit-binding-id"},
    )

    assert out["prompt_cache_binding"] == "explicit-binding-id"
