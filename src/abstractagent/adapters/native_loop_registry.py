"""Native loop factory registry for gateway-hosted agent bundles.

Gateway loads manifest-only bundles (``metadata.native_loop_factory``) and
materializes a :class:`~abstractruntime.WorkflowSpec` through this module
instead of compiling embedded VisualFlow graphs.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from abstractcore.tools import ToolDefinition
from abstractruntime import WorkflowSpec

from ..logic.builtins import (
    ASK_USER_TOOL,
    COMPACT_MEMORY_TOOL,
    DELEGATE_AGENT_TOOL,
    INSPECT_VARS_TOOL,
    OPEN_ATTACHMENT_TOOL,
    RECALL_MEMORY_TOOL,
    REMEMBER_NOTE_TOOL,
    REMEMBER_TOOL,
)
from ..logic.codeact import CodeActLogic
from ..logic.memact import MemActLogic
from ..logic.react import ReActLogic
from .codeact_runtime import create_codeact_workflow
from .memact_runtime import create_memact_workflow
from .react_runtime import create_react_workflow

AGENT_V1_INTERFACE = "abstractcode.agent.v1"

ALLOWED_MANIFEST_METADATA_KEYS = frozenset(
    {
        "loop_family",
        "native_loop_factory",
        "max_iterations_default",
        "headless_policy",
        "allowed_tools",
        "publisher",
    }
)

NATIVE_LOOP_FACTORIES = frozenset({"react", "codeact", "memact"})

_POLLER_EFFECT_TYPES = frozenset({"wait_until"})


class NativeLoopManifestError(ValueError):
    """Raised when a native-loop bundle manifest fails pack audit."""


def _workflow_id(*, bundle_ref: str, entrypoint: str) -> str:
    bundle_ref = str(bundle_ref or "").strip()
    entrypoint = str(entrypoint or "").strip()
    if bundle_ref and entrypoint:
        return f"{bundle_ref}:{entrypoint}"
    if bundle_ref:
        return bundle_ref
    return entrypoint or "native_loop"


def _tool_definitions_from_callables(tools: Iterable[Any]) -> List[ToolDefinition]:
    """Build deduped ToolDefinitions from host tool callables."""
    out: List[ToolDefinition] = []
    seen: set[str] = set()
    for tool in tools:
        tool_def = getattr(tool, "_tool_definition", None)
        if tool_def is None:
            tool_def = ToolDefinition.from_function(tool)
        name = str(getattr(tool_def, "name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(tool_def)
    return out


def _default_react_memact_tool_definitions() -> List[ToolDefinition]:
    """Host default toolset for gateway-native react/memact loops."""
    from ..tools import ALL_TOOLS

    return _tool_definitions_from_callables(ALL_TOOLS)


def _default_codeact_tool_definitions() -> List[ToolDefinition]:
    """Host default toolset for gateway-native codeact loops."""
    from ..tools.code_execution import execute_python

    return [
        ASK_USER_TOOL,
        OPEN_ATTACHMENT_TOOL,
        RECALL_MEMORY_TOOL,
        INSPECT_VARS_TOOL,
        REMEMBER_TOOL,
        REMEMBER_NOTE_TOOL,
        COMPACT_MEMORY_TOOL,
        DELEGATE_AGENT_TOOL,
        *_tool_definitions_from_callables([execute_python]),
    ]


def _coerce_allowed_tools(metadata: Mapping[str, Any]) -> Optional[List[str]]:
    raw = metadata.get("allowed_tools")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise NativeLoopManifestError("metadata.allowed_tools must be a list of tool names")
    out: List[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name:
            out.append(name)
    return out or None


def _unknown_metadata_keys(metadata: Mapping[str, Any]) -> List[str]:
    return sorted(
        key
        for key in metadata.keys()
        if key not in ALLOWED_MANIFEST_METADATA_KEYS
    )


def _flow_has_poller(flow: Mapping[str, Any]) -> bool:
    nodes = flow.get("nodes")
    if not isinstance(nodes, list):
        return False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").strip().lower()
        if node_type in _POLLER_EFFECT_TYPES:
            return True
        effect = node.get("effect")
        if isinstance(effect, dict):
            effect_type = str(effect.get("type") or "").strip().lower()
            if effect_type in _POLLER_EFFECT_TYPES:
                return True
    return False


def audit_native_loop_manifest(
    *,
    metadata: Mapping[str, Any],
    interfaces: Sequence[str],
    flows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate a native-loop bundle manifest. Raises on failure."""
    unknown = _unknown_metadata_keys(metadata)
    if unknown:
        raise NativeLoopManifestError(
            f"unknown manifest metadata keys: {', '.join(unknown)}"
        )

    factory = str(
        metadata.get("native_loop_factory") or metadata.get("loop_family") or ""
    ).strip().lower()
    if factory not in NATIVE_LOOP_FACTORIES:
        raise NativeLoopManifestError(
            f"unsupported native_loop_factory {factory!r}; "
            f"expected one of {sorted(NATIVE_LOOP_FACTORIES)}"
        )

    if metadata.get("loop_family") is not None:
        loop_family = str(metadata.get("loop_family") or "").strip().lower()
        if loop_family != factory:
            raise NativeLoopManifestError(
                f"metadata.loop_family ({loop_family!r}) "
                f"does not match native_loop_factory ({factory!r})"
            )

    interface_set = {str(item or "").strip() for item in interfaces if str(item or "").strip()}
    if AGENT_V1_INTERFACE not in interface_set:
        raise NativeLoopManifestError(
            f"interfaces must include {AGENT_V1_INTERFACE!r}"
        )

    if flows:
        for index, flow in enumerate(flows):
            if not isinstance(flow, Mapping):
                raise NativeLoopManifestError(f"flows[{index}] must be an object")
            if _flow_has_poller(flow):
                raise NativeLoopManifestError(
                    "native-loop bundles must not ship VisualFlow poller nodes "
                    f"(wait_until found in flows[{index}])"
                )


def build_native_loop_manifest(
    bundle_id: str,
    factory: str,
    *,
    version: str = "0.1.0",
    max_iterations_default: int = 50,
    headless_policy: str = "refuse_ask",
    allowed_tools: Optional[Sequence[str]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a manifest-only native-loop bundle document for gateway packing.

    Gateway zips the returned dict as ``manifest.json`` inside a ``.flow`` bundle.
    The manifest is audited before return so pack scripts fail fast.
    """
    factory_key = str(factory or "").strip().lower()
    bundle_id = str(bundle_id or "").strip()
    if not bundle_id:
        raise NativeLoopManifestError("bundle_id is required")
    entrypoint = factory_key
    bundle_ref = f"{bundle_id}@{version}"
    metadata: Dict[str, Any] = {
        "native_loop_factory": factory_key,
        "loop_family": factory_key,
        "max_iterations_default": max_iterations_default,
        "headless_policy": headless_policy,
    }
    if allowed_tools is not None:
        metadata["allowed_tools"] = list(allowed_tools)
    manifest: Dict[str, Any] = {
        "bundle_format_version": "1",
        "bundle_id": bundle_id,
        "bundle_version": version,
        "created_at": created_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metadata": metadata,
        "interfaces": [AGENT_V1_INTERFACE],
        "flows": {},
        "artifacts": {},
        "assets": {},
        "default_entrypoint": entrypoint,
        "entrypoints": [
            {
                "flow_id": entrypoint,
                "name": entrypoint,
                "description": f"Native {factory_key} agent loop (abstractagent)",
                "interfaces": [AGENT_V1_INTERFACE],
            }
        ],
    }
    audit_native_loop_manifest(
        metadata=metadata,
        interfaces=manifest["interfaces"],
        flows=[],
    )
    # Prove materialization succeeds for the declared factory.
    materialize_native_loop_spec(
        factory_key,
        bundle_ref=bundle_ref,
        entrypoint=entrypoint,
        metadata=metadata,
    )
    return manifest


def pack_native_loop_bundle(
    output_path: Union[str, Path],
    manifest: Mapping[str, Any],
) -> Path:
    """Write a manifest-only ``.flow`` zip understood by the gateway loader."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = manifest.get("metadata") or {}
    audit_native_loop_manifest(
        metadata=metadata,
        interfaces=list(manifest.get("interfaces") or []),
        flows=[],
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(dict(manifest), indent=2))
        names = zf.namelist()
    if names != ["manifest.json"]:
        raise NativeLoopManifestError(
            f"native-loop bundle must contain only manifest.json, got {names!r}"
        )
    return path


def materialize_native_loop_spec(
    factory: str,
    *,
    bundle_ref: str,
    entrypoint: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WorkflowSpec:
    """Materialize a gateway-hosted native loop as a runtime workflow spec."""
    meta = dict(metadata or {})
    factory_key = str(factory or "").strip().lower()
    meta.setdefault("native_loop_factory", factory_key)
    meta.setdefault("loop_family", factory_key)
    audit_native_loop_manifest(
        metadata=meta,
        interfaces=[AGENT_V1_INTERFACE],
        flows=[],
    )

    if factory_key not in NATIVE_LOOP_FACTORIES:
        raise NativeLoopManifestError(
            f"unsupported native loop factory {factory!r}"
        )

    workflow_id = _workflow_id(bundle_ref=bundle_ref, entrypoint=entrypoint)
    allowed_tools = _coerce_allowed_tools(meta)

    if factory_key == "react":
        return create_react_workflow(
            logic=ReActLogic(tools=_default_react_memact_tool_definitions()),
            workflow_id=workflow_id,
            allowed_tools=allowed_tools,
            final_next_node=None,
        )
    if factory_key == "codeact":
        return create_codeact_workflow(
            logic=CodeActLogic(tools=_default_codeact_tool_definitions()),
            workflow_id=workflow_id,
        )
    return create_memact_workflow(
        logic=MemActLogic(tools=_default_react_memact_tool_definitions()),
        workflow_id=workflow_id,
        allowed_tools=allowed_tools,
    )
