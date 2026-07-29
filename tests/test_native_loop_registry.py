"""Tests for gateway native-loop factory materialization."""

from __future__ import annotations

import json
import zipfile

import pytest

from abstractagent.adapters.native_loop_registry import (
    AGENT_V1_INTERFACE,
    NativeLoopManifestError,
    audit_native_loop_manifest,
    build_native_loop_manifest,
    materialize_native_loop_spec,
    pack_native_loop_bundle,
)
from abstractruntime import EffectType, RunState, RunStatus


class _Ctx:
    def now_iso(self) -> str:
        return "2025-01-01T00:00:00+00:00"


def _base_vars() -> dict:
    return {
        "context": {"task": "t", "messages": []},
        "scratchpad": {"iteration": 0, "max_iterations": 2},
        "_runtime": {"inbox": []},
        "_temp": {},
        "_limits": {
            "max_iterations": 2,
            "current_iteration": 0,
            "max_history_messages": -1,
            "max_tokens": 1024,
        },
    }


def _run(*, vars: dict, current_node: str = "init") -> RunState:
    return RunState(
        run_id="run",
        workflow_id="wf",
        status=RunStatus.RUNNING,
        current_node=current_node,
        vars=vars,
        waiting=None,
        output=None,
        error=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
        actor_id=None,
        session_id=None,
        parent_run_id=None,
    )


def test_materialize_react_spec_uses_catalog_workflow_id():
    spec = materialize_native_loop_spec(
        "react",
        bundle_ref="react-agent@0.1.0",
        entrypoint="react",
        metadata={
            "native_loop_factory": "react",
            "loop_family": "react",
            "headless_policy": "refuse_ask",
        },
    )
    assert spec.workflow_id == "react-agent@0.1.0:react"
    assert spec.entry_node == "init"
    assert "reason" in spec.nodes


def test_audit_rejects_unknown_metadata_keys():
    with pytest.raises(NativeLoopManifestError, match="unknown manifest metadata"):
        audit_native_loop_manifest(
            metadata={"loop_family": "react", "unexpected": True},
            interfaces=[AGENT_V1_INTERFACE],
            flows=[],
        )


def test_audit_rejects_poller_flows():
    with pytest.raises(NativeLoopManifestError, match="poller"):
        audit_native_loop_manifest(
            metadata={"loop_family": "react", "native_loop_factory": "react"},
            interfaces=[AGENT_V1_INTERFACE],
            flows=[
                {
                    "nodes": [
                        {"id": "poll", "type": "wait_until", "effect": {"type": "wait_until"}},
                    ]
                }
            ],
        )


def test_audit_requires_agent_v1_interface():
    with pytest.raises(NativeLoopManifestError, match=AGENT_V1_INTERFACE):
        audit_native_loop_manifest(
            metadata={"loop_family": "react", "native_loop_factory": "react"},
            interfaces=["other.interface"],
            flows=[],
        )


def test_audit_passes_shipped_react_agent_manifest():
    """Manifest-only react-agent@0.1.0 shape (gateway unit5 / pass C)."""
    audit_native_loop_manifest(
        metadata={
            "native_loop_factory": "react",
            "loop_family": "react",
            "headless_policy": "refuse_ask",
            "max_iterations_default": 50,
        },
        interfaces=[AGENT_V1_INTERFACE],
        flows=[],
    )
    spec = materialize_native_loop_spec(
        "react",
        bundle_ref="react-agent@0.1.0",
        entrypoint="react",
        metadata={
            "native_loop_factory": "react",
            "loop_family": "react",
            "headless_policy": "refuse_ask",
        },
    )
    assert spec.workflow_id == "react-agent@0.1.0:react"
    assert spec.entry_node == "init"
    assert "done" in spec.nodes


def test_materialize_react_inherits_host_default_tools_when_unset():
    """Native react loops must not materialize with an empty tool surface."""
    spec = materialize_native_loop_spec(
        "react",
        bundle_ref="react-agent@0.1.0",
        entrypoint="react",
        metadata={
            "native_loop_factory": "react",
            "loop_family": "react",
            "headless_policy": "refuse_ask",
        },
    )
    run = _run(vars=_base_vars())
    init_plan = spec.get_node("init")(run, _Ctx())
    assert init_plan.next_node == "reason"

    allow = run.vars["_runtime"].get("allowed_tools")
    assert isinstance(allow, list)
    assert "write_file" in allow
    assert "read_file" in allow

    specs = run.vars["_runtime"].get("tool_specs")
    assert isinstance(specs, list)
    assert len(specs) >= 1
    names = {s.get("name") for s in specs if isinstance(s, dict)}
    assert "write_file" in names


def test_materialize_react_respects_manifest_allowed_tools():
    spec = materialize_native_loop_spec(
        "react",
        bundle_ref="react-agent@0.1.0",
        entrypoint="react",
        metadata={
            "native_loop_factory": "react",
            "loop_family": "react",
            "allowed_tools": ["read_file"],
        },
    )
    run = _run(vars=_base_vars())
    spec.get_node("init")(run, _Ctx())
    allow = run.vars["_runtime"].get("allowed_tools")
    assert allow == ["read_file"]
    specs = run.vars["_runtime"].get("tool_specs")
    names = {s.get("name") for s in specs if isinstance(s, dict)}
    assert names == {"read_file"}


def test_materialize_codeact_inherits_execute_python_default():
    spec = materialize_native_loop_spec(
        "codeact",
        bundle_ref="codeact-agent@0.1.0",
        entrypoint="codeact",
        metadata={"native_loop_factory": "codeact", "loop_family": "codeact"},
    )
    run = _run(vars=_base_vars())
    init_plan = spec.get_node("init")(run, _Ctx())
    assert init_plan.next_node == "reason"
    allow = run.vars["_runtime"].get("allowed_tools")
    assert isinstance(allow, list)
    assert "execute_python" in allow


@pytest.mark.parametrize(
    ("bundle_id", "factory"),
    [
        ("react-agent", "react"),
        ("codeact-agent", "codeact"),
        ("memact-agent", "memact"),
    ],
)
def test_build_native_loop_manifest_for_all_factories(bundle_id, factory):
    manifest = build_native_loop_manifest(bundle_id, factory)
    assert manifest["bundle_id"] == bundle_id
    assert manifest["metadata"]["native_loop_factory"] == factory
    assert manifest["metadata"]["loop_family"] == factory
    assert manifest["flows"] == {}
    assert manifest["bundle_format_version"] == "1"
    assert manifest["default_entrypoint"] == factory
    assert AGENT_V1_INTERFACE in manifest["interfaces"]
    spec = materialize_native_loop_spec(
        factory,
        bundle_ref=f"{bundle_id}@0.1.0",
        entrypoint=factory,
        metadata=manifest["metadata"],
    )
    assert spec.entry_node == "init"


@pytest.mark.parametrize(
    ("bundle_id", "factory"),
    [
        ("codeact-agent", "codeact"),
        ("memact-agent", "memact"),
    ],
)
def test_materialize_namespaced_workflow_id(bundle_id, factory):
    spec = materialize_native_loop_spec(
        factory,
        bundle_ref=f"{bundle_id}@0.1.0",
        entrypoint=factory,
        metadata={"native_loop_factory": factory, "loop_family": factory},
    )
    assert spec.workflow_id == f"{bundle_id}@0.1.0:{factory}"


@pytest.mark.parametrize(
    ("bundle_id", "factory"),
    [
        ("codeact-agent", "codeact"),
        ("memact-agent", "memact"),
    ],
)
def test_pack_native_loop_bundle_writes_manifest_only_zip(tmp_path, bundle_id, factory):
    manifest = build_native_loop_manifest(
        bundle_id,
        factory,
        created_at="2026-07-28T08:00:00+00:00",
    )
    out = tmp_path / f"{bundle_id}@0.1.0.flow"
    pack_native_loop_bundle(out, manifest)
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["manifest.json"]
        loaded = json.loads(zf.read("manifest.json"))
    assert loaded["bundle_id"] == bundle_id
    assert loaded["metadata"]["native_loop_factory"] == factory


def test_top_level_package_exports_gateway_loader_surface():
    """Gateway imports ``abstractagent.materialize_native_loop_spec`` (not adapters)."""
    import abstractagent

    for name in (
        "audit_native_loop_manifest",
        "materialize_native_loop_spec",
        "build_native_loop_manifest",
        "pack_native_loop_bundle",
    ):
        assert hasattr(abstractagent, name), name
        assert callable(getattr(abstractagent, name))
