"""Deprecated CLI entrypoint for AbstractAgent.

The interactive REPL and UX components were extracted into **AbstractCode** to
avoid mixing UI concerns with agent patterns.

Use:
  abstractcode --agent react --provider <provider> --model <model>
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="react-agent",
        description="Deprecated: the interactive REPL moved to AbstractCode.",
    )
    # Behavior-env-vars ruling (commons c4157, 2026-07-21): this stub used to
    # sniff ABSTRACTCODE_PROVIDER/MODEL just to echo them into the printed
    # suggestion — a cross-package env read for zero behavior. Plain defaults;
    # the real selection lives in abstractcode's own config.
    parser.add_argument("--provider", default="<provider>")
    parser.add_argument("--model", default="<model>")
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("The AbstractAgent interactive REPL has moved to AbstractCode.\n")
    print("Run:")
    print(f"  abstractcode --agent react --provider {args.provider} --model {args.model}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

