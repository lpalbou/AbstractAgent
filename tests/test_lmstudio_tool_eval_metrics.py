from __future__ import annotations

from abstractagent.scripts.lmstudio_tool_eval import ToolCallEvent, _summarize_tool_usage


def _event(ts: str, name: str, arguments: dict[str, object], *, success: bool | None = True) -> ToolCallEvent:
    return ToolCallEvent(ts=ts, name=name, arguments=arguments, success=success, error=None)


def test_summarize_tool_usage_counts_search_churn_after_full_read_and_reread() -> None:
    events = [
        _event("2026-08-17T00:00:01Z", "read_file", {"path": "game.js"}),
        _event("2026-08-17T00:00:02Z", "search_files", {"query": "MEADOW_EAST"}),
        _event("2026-08-17T00:00:03Z", "search_files", {"query": "FOREST_GHOST"}),
        _event("2026-08-17T00:00:04Z", "search_files", {"query": "RIVER_FISHER"}),
        _event("2026-08-17T00:00:05Z", "read_file", {"path": "game.js"}),
    ]

    summary = _summarize_tool_usage(events)

    assert summary["read_file_full"] == 2
    assert summary["search_files_calls"] == 3
    assert summary["search_after_full_read_calls"] == 3
    assert summary["full_file_rereads_after_full_read"] == 1


def test_summarize_tool_usage_counts_combined_searches_without_full_reread() -> None:
    events = [
        _event("2026-08-17T00:00:01Z", "read_file", {"path": "game.js"}),
        _event("2026-08-17T00:00:02Z", "search_files", {"query": "MEADOW_EAST|FOREST_GHOST|RIVER_FISHER", "output_mode": "lines"}, success=False),
        _event("2026-08-17T00:00:03Z", "search_files", {"query": "MEADOW_EAST|FOREST_GHOST|RIVER_FISHER", "output_mode": "content"}),
        _event("2026-08-17T00:00:04Z", "search_files", {"query": "MEADOW"}),
    ]

    summary = _summarize_tool_usage(events)

    assert summary["read_file_full"] == 1
    assert summary["search_files_calls"] == 3
    assert summary["search_after_full_read_calls"] == 3
    assert summary["full_file_rereads_after_full_read"] == 0
