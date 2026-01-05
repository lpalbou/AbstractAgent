from __future__ import annotations


def _required_arg_names(parameters: dict) -> list[str]:
    required: list[str] = []
    for name, meta in parameters.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(meta, dict) or "default" not in meta:
            required.append(name)
    return sorted(required)


def test_builtin_tool_specs_mark_optional_args_with_defaults() -> None:
    from abstractagent.logic import builtins

    assert _required_arg_names(builtins.ASK_USER_TOOL.parameters) == ["question"]
    assert _required_arg_names(builtins.RECALL_MEMORY_TOOL.parameters) == []
    assert _required_arg_names(builtins.INSPECT_VARS_TOOL.parameters) == []
    assert _required_arg_names(builtins.REMEMBER_TOOL.parameters) == ["span_id", "tags"]
    assert _required_arg_names(builtins.REMEMBER_NOTE_TOOL.parameters) == ["note"]
    assert _required_arg_names(builtins.COMPACT_MEMORY_TOOL.parameters) == []
