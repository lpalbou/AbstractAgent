"""phase:review seam arithmetic for abstractagent (three-column form, release#220).

Enumerates every reference abstractagent writes into abstractcore / abstractruntime /
abstractmemory / abstractflow, then resolves each one against a given interpreter's
site-packages. Run it once per environment:

  (b)  sibling SOURCE trees        -> the dev interpreter (core 2.13.39, runtime 0.4.30)
  (b') PUBLISHED wheels            -> untracked/release-audit/venv-published
                                      (core 2.13.38, runtime 0.4.29 -- the newest on PyPI)

Usage: <python> seam_probe.py <src|tests|all>
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata as md
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGETS = ("abstractcore", "abstractruntime", "abstractmemory", "abstractflow")


def collect(root: pathlib.Path) -> dict[str, set[tuple[str, str]]]:
    """-> {package: {(module, symbol), ...}}. `symbol` is "" for a bare module import."""
    found: dict[str, set[tuple[str, str]]] = {t: set() for t in TARGETS}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                pkg = node.module.split(".")[0]
                if pkg in found:
                    for alias in node.names:
                        found[pkg].add((node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = alias.name.split(".")[0]
                    if pkg in found:
                        found[pkg].add((alias.name, ""))
    return found


def resolve(module: str, symbol: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - we are reporting, not handling
        return False, f"{type(exc).__name__}: {exc}"
    if not symbol:
        return True, "module"
    if hasattr(mod, symbol):
        return True, "ok"
    return False, "AttributeError: symbol absent from module"


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    roots = {"src": [REPO / "src"], "tests": [REPO / "tests"]}
    roots["all"] = roots["src"] + roots["tests"]

    print("ENVIRONMENT")
    for pkg in TARGETS + ("abstractagent",):
        try:
            print(f"  {pkg:18s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:18s} NOT INSTALLED")
    print(f"  python             {sys.version.split()[0]}")
    print(f"  scope              {which}\n")

    grand_written = grand_ok = 0
    for root in roots[which]:
        refs = collect(root)
        for pkg in TARGETS:
            items = sorted(refs[pkg])
            if not items:
                continue
            ok = 0
            failures: list[str] = []
            for module, symbol in items:
                good, why = resolve(module, symbol)
                if good:
                    ok += 1
                else:
                    name = f"{module}.{symbol}" if symbol else module
                    failures.append(f"      MISS {name}  <- {why}")
            grand_written += len(items)
            grand_ok += ok
            print(f"{root.name}/ -> {pkg}: {ok}/{len(items)} resolve")
            for line in failures:
                print(line)
            print()

    print(f"TOTAL ({which}): {grand_ok}/{grand_written} references resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
