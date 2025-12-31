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

