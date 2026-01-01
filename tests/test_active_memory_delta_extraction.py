from __future__ import annotations

from abstractagent.adapters.memory_delta import extract_active_memory_delta


def test_extract_active_memory_delta_removes_block_and_parses_json() -> None:
    text = (
        "Hello\n\n"
        "```active_memory_delta\n"
        "{\n"
        "  \"current_tasks\": {\"upsert\": [{\"task_id\": \"t_1\", \"title\": \"Do\"}]}\n"
        "}\n"
        "```\n"
    )
    cleaned, delta = extract_active_memory_delta(text)
    assert "active_memory_delta" not in cleaned
    assert isinstance(delta, dict)
    assert delta.get("current_tasks", {}).get("upsert", [])[0]["task_id"] == "t_1"


def test_extract_active_memory_delta_strips_all_blocks_and_uses_last_valid() -> None:
    text = (
        "Hello\n"
        "```active_memory_delta\n"
        "{ \"current_context\": {\"upsert\": [{\"title\": \"A\"}]}}\n"
        "```\n"
        "World\n"
        "```active_memory_delta\n"
        "{ \"key_history\": {\"add\": [{\"kind\": \"event\", \"summary\": \"B\"}]}}\n"
        "```\n"
        "Done\n"
    )
    cleaned, delta = extract_active_memory_delta(text)
    assert "```active_memory_delta" not in cleaned
    assert "active_memory_delta" not in cleaned
    assert "Hello" in cleaned and "World" in cleaned and "Done" in cleaned
    assert isinstance(delta, dict)
    assert delta.get("key_history", {}).get("add", [])[0]["summary"] == "B"


def test_extract_active_memory_delta_ignores_invalid_json_but_still_strips() -> None:
    text = (
        "Hello\n"
        "```active_memory_delta\n"
        "{ \"current_context\": {\"upsert\": [{\"title\": \"A\"}]}}\n"
        "```\n"
        "```active_memory_delta\n"
        "{ \"current_context\":\n"  # truncated/invalid JSON (missing closing braces)
        "```\n"
        "Done\n"
    )
    cleaned, delta = extract_active_memory_delta(text)
    assert "active_memory_delta" not in cleaned
    assert "Hello" in cleaned and "Done" in cleaned
    assert isinstance(delta, dict)
    assert delta.get("current_context", {}).get("upsert", [])[0]["title"] == "A"
