"""Deprecated CLI entrypoint for AbstractAgent.

The interactive REPL and UX components were extracted into **AbstractCode** to
avoid mixing UI concerns with agent patterns.

Use:
  abstractcode --agent react --provider <provider> --model <model>
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="react-agent",
        description="Deprecated: the interactive REPL moved to AbstractCode.",
    )
    parser.add_argument("--provider", default=os.getenv("ABSTRACTCODE_PROVIDER", "ollama"))
    parser.add_argument("--model", default=os.getenv("ABSTRACTCODE_MODEL", "qwen3:4b-instruct-2507-q4_K_M"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("The AbstractAgent interactive REPL has moved to AbstractCode.\n")
    print("Run:")
    print(f"  abstractcode --agent react --provider {args.provider} --model {args.model}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

