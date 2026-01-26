from abstractagent.adapters.react_runtime import _looks_like_deferred_action


def test_looks_like_deferred_action_does_not_flag_plan_mode_mentions() -> None:
    text = (
        "This CLI supports plan mode (generate TODOs) and review mode. "
        "Use /plan to toggle plan mode."
    )
    assert _looks_like_deferred_action(text) is False


def test_looks_like_deferred_action_does_not_flag_non_first_person_plans() -> None:
    text = "Plan: First, read the file. Next, summarize it. Then, answer the user."
    assert _looks_like_deferred_action(text) is False


def test_looks_like_deferred_action_does_not_flag_waiting_responses() -> None:
    text = "I understand. I'm ready to help. Let me know your next step. No tool calls needed yet."
    assert _looks_like_deferred_action(text) is False


def test_looks_like_deferred_action_flags_first_person_intent_to_act() -> None:
    text = "To proceed meaningfully, I'll explore the codebase structure and read the relevant files."
    assert _looks_like_deferred_action(text) is True


def test_looks_like_deferred_action_does_not_flag_writing_the_answer() -> None:
    text = "I'll write the final answer now."
    assert _looks_like_deferred_action(text) is False
