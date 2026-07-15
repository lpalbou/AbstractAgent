"""Verifier schema strict-mode expressibility (airelay 422 incident, 2026-07-15).

OpenAI-strict validators (and subscription relays in front of them) require
every object schema node to declare `properties`, a `required` array listing
every property key, and `additionalProperties: false`. The old verifier schema
carried `next_tool_calls[].arguments` as a bare `{"type": "object"}` free-form
dict — not expressible under those rules — so the WHOLE request 422'd on such
backends. `arguments` now rides as a JSON-encoded string (OpenAI's own
function-calling precedent) and the parse sites accept both shapes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from abstractagent.adapters.generation_params import (
    coerce_verifier_tool_arguments,
    verifier_response_schema,
)
from abstractagent.adapters.react_runtime import create_react_workflow
from abstractagent.logic.react import ReActLogic
from abstractcore.tools import ToolDefinition
from abstractruntime.core.models import Effect, EffectType, RunState, RunStatus  # noqa: F401
from abstractruntime.core.runtime import EffectOutcome, Runtime
from abstractruntime.storage.in_memory import InMemoryLedgerStore, InMemoryRunStore


# ---------------------------------------------------------------------------
# Strict-rules checker (OpenAI structured-outputs published rules)
# ---------------------------------------------------------------------------

def _assert_strict_expressible(node: Any, path: str = "$") -> None:
    """Every object node must declare properties, require every key, and set
    additionalProperties to false; free-form dicts are forbidden anywhere."""
    if isinstance(node, list):
        for i, item in enumerate(node):
            _assert_strict_expressible(item, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return

    if node.get("type") == "object":
        props = node.get("properties")
        assert isinstance(props, dict) and props, f"{path}: object without properties (free-form dict)"
        required = node.get("required")
        assert isinstance(required, list) and set(required) == set(props.keys()), (
            f"{path}: strict mode requires `required` to list every property key"
        )
        assert node.get("additionalProperties") is False, (
            f"{path}: strict mode requires additionalProperties: false"
        )

    for key, value in node.items():
        _assert_strict_expressible(value, f"{path}.{key}")


def test_verifier_schema_is_strict_expressible() -> None:
    _assert_strict_expressible(verifier_response_schema())


def test_verifier_schema_has_no_free_form_dict_after_runtime_round_trip() -> None:
    """The runtime regenerates the schema through a dynamic pydantic model
    before it reaches the provider wire; the regenerated schema must contain
    no free-form object (`type: object` without properties) and no
    `additionalProperties: true` — the two constructs strict validators refuse
    outright (the pydantic round trip legitimately drops the explicit
    `additionalProperties: false`, which permissive-strict backends fill)."""
    from abstractruntime.integrations.abstractcore.effect_handlers import (
        _pydantic_model_from_json_schema,
    )

    model = _pydantic_model_from_json_schema(verifier_response_schema(), name="ReActVerifier")
    wire = model.model_json_schema()

    def walk(node: Any, path: str = "$") -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            assert isinstance(node.get("properties"), dict) and node["properties"], (
                f"{path}: free-form object leaked into the wire schema"
            )
        assert node.get("additionalProperties") is not True, (
            f"{path}: additionalProperties true is refused by strict validators"
        )
        for key, value in node.items():
            walk(value, f"{path}.{key}")

    walk(wire)


# ---------------------------------------------------------------------------
# Argument coercion (liberal parse over the strict wire contract)
# ---------------------------------------------------------------------------

def test_coerce_verifier_tool_arguments_shapes() -> None:
    assert coerce_verifier_tool_arguments({"path": "a.txt"}) == {"path": "a.txt"}
    assert coerce_verifier_tool_arguments('{"path": "a.txt"}') == {"path": "a.txt"}
    assert coerce_verifier_tool_arguments("{}") == {}
    assert coerce_verifier_tool_arguments("") == {}
    assert coerce_verifier_tool_arguments("   ") == {}
    assert coerce_verifier_tool_arguments("not json") == {}
    assert coerce_verifier_tool_arguments("[1, 2]") == {}  # non-object JSON degrades to {}
    assert coerce_verifier_tool_arguments(None) == {}
    assert coerce_verifier_tool_arguments(42) == {}


# ---------------------------------------------------------------------------
# Integration: verifier round-trip with string-encoded arguments
# ---------------------------------------------------------------------------

def _base_vars(*, task: str, runtime_ns: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "context": {"task": task, "messages": []},
        "_runtime": dict({"inbox": []}, **(runtime_ns or {})),
    }


def test_verifier_string_arguments_execute_as_parsed_dicts() -> None:
    """A verifier verdict carrying JSON-STRING arguments (the new schema shape)
    must drive the forced tool call with the PARSED dict arguments."""
    llm_calls: list[dict[str, Any]] = []
    executed_tool_calls: list[dict[str, Any]] = []

    def llm_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        payload = dict(effect.payload or {})
        is_review = str(payload.get("response_schema_name") or "") == "ReActVerifier"
        llm_calls.append({"review": is_review})
        if is_review:
            n_reviews = sum(1 for c in llm_calls if c["review"])
            if n_reviews == 1:
                return EffectOutcome.completed(
                    {
                        "data": {
                            "complete": False,
                            "missing": ["must list files"],
                            "next_prompt": "",
                            # NEW SHAPE: arguments as a JSON-encoded string.
                            "next_tool_calls": [
                                {"name": "list_files", "arguments": '{"directory_path": "."}'}
                            ],
                        }
                    }
                )
            return EffectOutcome.completed(
                {"data": {"complete": True, "missing": [], "next_prompt": "", "next_tool_calls": []}}
            )
        return EffectOutcome.completed({"content": "All done.", "tool_calls": []})

    def tool_handler(run: RunState, effect: Effect, default_next_node: Optional[str]) -> EffectOutcome:
        tcs = effect.payload.get("tool_calls") or []
        executed_tool_calls.extend(dict(tc) for tc in tcs)
        return EffectOutcome.completed(
            {
                "mode": "executed",
                "results": [
                    {
                        "call_id": tc.get("call_id"),
                        "name": tc.get("name"),
                        "success": True,
                        "output": "ok",
                        "error": None,
                    }
                    for tc in tcs
                ],
            }
        )

    rt = Runtime(
        run_store=InMemoryRunStore(),
        ledger_store=InMemoryLedgerStore(),
        effect_handlers={EffectType.LLM_CALL: llm_handler, EffectType.TOOL_CALLS: tool_handler},
    )
    wf = create_react_workflow(
        logic=ReActLogic(tools=[ToolDefinition(name="list_files", description="List", parameters={})]),
        workflow_id="wf",
        provider="stub",
        model="stub",
        allowed_tools=["list_files"],
    )
    rid = rt.start(
        workflow=wf,
        vars=_base_vars(task="Explore", runtime_ns={"review_mode": True, "review_max_rounds": 2}),
    )
    for _ in range(60):
        st = rt.tick(workflow=wf, run_id=rid, max_steps=1)
        if st.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break

    assert rt.get_state(rid).status == RunStatus.COMPLETED
    forced = [tc for tc in executed_tool_calls if str(tc.get("name")) == "list_files"]
    assert forced, "verifier-forced tool call did not execute"
    assert forced[-1].get("arguments") == {"directory_path": "."}, (
        "string-encoded arguments must be parsed into the executed tool call"
    )
