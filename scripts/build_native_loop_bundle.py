#!/usr/bin/env python3
"""Build manifest-only native-loop .flow bundles for the gateway loader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstractagent.adapters.native_loop_registry import (
    build_native_loop_manifest,
    pack_native_loop_bundle,
)

DEFAULT_PACKS = (
    ("codeact-agent", "codeact"),
    ("memact-agent", "memact"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack abstractagent native-loop manifests into gateway .flow bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for *.flow bundle files",
    )
    parser.add_argument(
        "--version",
        default="0.1.0",
        help="Bundle semver (default: 0.1.0)",
    )
    parser.add_argument(
        "--bundle-id",
        dest="bundles",
        action="append",
        nargs=2,
        metavar=("BUNDLE_ID", "FACTORY"),
        help="One pack; repeat for multiple (default: codeact-agent + memact-agent)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    packs = args.bundles or list(DEFAULT_PACKS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for bundle_id, factory in packs:
        manifest = build_native_loop_manifest(
            bundle_id,
            factory,
            version=args.version,
        )
        filename = f"{bundle_id}@{args.version}.flow"
        out = args.output_dir / filename
        pack_native_loop_bundle(out, manifest)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
