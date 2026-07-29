#!/usr/bin/env python3
"""Smoke-check the gateway-facing abstractagent native-loop import surface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstractagent import (  # noqa: E402
    audit_native_loop_manifest,
    build_native_loop_manifest,
    materialize_native_loop_spec,
)

_EXPORTS = (
    audit_native_loop_manifest,
    build_native_loop_manifest,
    materialize_native_loop_spec,
)


def main() -> int:
    for fn in _EXPORTS:
        if not callable(fn):
            print(f"FAIL: {fn!r} is not callable", file=sys.stderr)
            return 1

    manifest = build_native_loop_manifest("react-agent", "react")
    audit_native_loop_manifest(
        metadata=manifest["metadata"],
        interfaces=manifest["interfaces"],
        flows=manifest.get("flows") or [],
    )
    spec = materialize_native_loop_spec(
        "react",
        bundle_ref="react-agent@0.1.0",
        entrypoint="react",
        metadata=manifest["metadata"],
    )
    if spec.workflow_id != "react-agent@0.1.0:react":
        print(f"FAIL: unexpected workflow_id {spec.workflow_id!r}", file=sys.stderr)
        return 1

    print("OK: gateway native-loop import surface (audit + materialize react)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
