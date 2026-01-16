from abstractagent.adapters.react_runtime import _looks_like_plan


def test_looks_like_plan_does_not_flag_plan_mode_mentions() -> None:
    text = (
        "This CLI supports plan mode (generate TODOs) and review mode. "
        "Use /plan to toggle plan mode."
    )
    assert _looks_like_plan(text) is False


def test_looks_like_plan_flags_explicit_planning() -> None:
    text = "Plan: First, read the file. Next, summarize it. Then, answer the user."
    assert _looks_like_plan(text) is True

