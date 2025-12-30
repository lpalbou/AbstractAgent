from __future__ import annotations

from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition


def _base_vars() -> dict:
    return {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 1, "max_iterations": 3},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {
            "max_iterations": 3,
            "current_iteration": 1,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }


def test_react_prompt_includes_runtime_node_trace_scratchpad() -> None:
    fetch_url = ToolDefinition(
        name="fetch_url",
        description="Fetch a URL and return content (stubbed).",
        parameters={"url": {"type": "string"}},
    )
    logic = ReActLogic(tools=[fetch_url])

    vars = _base_vars()
    runtime = vars.setdefault("_runtime", {})
    assert isinstance(runtime, dict)
    runtime["node_traces"] = {
        "reason": {
            "node_id": "reason",
            "steps": [
                {
                    "ts": "2025-01-01T00:00:00+00:00",
                    "node_id": "reason",
                    "status": "completed",
                    "effect": {"type": "llm_call", "payload": {}, "result_key": "_temp.llm_response"},
                    "result": {"content": "I will fetch the page once and then write the report."},
                }
            ],
        },
        "act": {
            "node_id": "act",
            "steps": [
                {
                    "ts": "2025-01-01T00:00:01+00:00",
                    "node_id": "act",
                    "status": "completed",
                    "effect": {
                        "type": "tool_calls",
                        "payload": {
                            "tool_calls": [
                                {
                                    "name": "fetch_url",
                                    "arguments": {"url": "https://example.com/a"},
                                    "call_id": "call_1",
                                }
                            ]
                        },
                        "result_key": "_temp.tool_results",
                    },
                    "result": {
                        "mode": "executed",
                        "results": [
                            {
                                "call_id": "call_1",
                                "name": "fetch_url",
                                "success": True,
                                "output": "OK",
                                "error": None,
                            }
                        ],
                    },
                }
            ],
        },
    }

    req = logic.build_request(
        task="t",
        messages=[{"role": "user", "content": "do research and write rtype-report.md"}],
        iteration=2,
        max_iterations=3,
        vars=vars,
    )

    assert "Scratchpad (runtime; tool calls + results):" in req.prompt
    assert "fetch_url" in req.prompt
    assert "https://example.com/a" in req.prompt
    assert "OK" in req.prompt


def test_react_prompt_includes_full_tool_call_arguments() -> None:
    write_file = ToolDefinition(
        name="write_file",
        description="Write a file (stubbed).",
        parameters={"file_path": {"type": "string"}, "content": {"type": "string"}},
    )
    logic = ReActLogic(tools=[write_file])

    vars = _base_vars()
    runtime = vars.setdefault("_runtime", {})
    assert isinstance(runtime, dict)

    huge = "A" * 10_000
    runtime["node_traces"] = {
        "act": {
            "node_id": "act",
            "steps": [
                {
                    "ts": "2025-01-01T00:00:00+00:00",
                    "node_id": "act",
                    "status": "completed",
                    "effect": {
                        "type": "tool_calls",
                        "payload": {
                            "tool_calls": [
                                {
                                    "name": "write_file",
                                    "arguments": {"file_path": "snake_game.py", "content": huge},
                                    "call_id": "call_1",
                                }
                            ]
                        },
                        "result_key": "_temp.tool_results",
                    },
                    "result": {
                        "mode": "executed",
                        "results": [
                            {
                                "call_id": "call_1",
                                "name": "write_file",
                                "success": True,
                                "output": "OK",
                                "error": None,
                            }
                        ],
                    },
                }
            ],
        }
    }

    req = logic.build_request(
        task="t",
        messages=[{"role": "user", "content": "write a file"}],
        iteration=2,
        max_iterations=3,
        vars=vars,
    )

    # The scratchpad is runtime-owned and injected verbatim so the model can reliably
    # ground subsequent steps in what was actually executed.
    assert "write_file" in req.prompt
    assert "snake_game.py" in req.prompt
    assert huge in req.prompt
