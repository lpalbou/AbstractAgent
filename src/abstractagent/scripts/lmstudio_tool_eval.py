"""LMStudio tool-synergy evaluation harness (manual, real LLM).

Runs a few deterministic scenarios against a real provider/model (e.g. LMStudio
`qwen/qwen3-next-80b`) and prints a compact summary of tool usage patterns:
- whether the model uses analyze_code/search_files before bounded reads
- whether edits succeed without repeated identical retries
- how often it reads whole files vs slices

This is intentionally NOT a required CI test. It should be run manually.

Usage:
  ABSTRACTAGENT_TEST_PROVIDER=lmstudio \
  ABSTRACTAGENT_TEST_MODEL=qwen/qwen3-next-80b \
  ABSTRACTAGENT_TEST_BASE_URL=http://localhost:1234/v1 \
  python -m abstractagent.scripts.lmstudio_tool_eval
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def _llm_config() -> Tuple[str, str, Dict[str, Any]]:
    provider = os.getenv("ABSTRACTAGENT_TEST_PROVIDER", "lmstudio")
    model = os.getenv("ABSTRACTAGENT_TEST_MODEL", "qwen/qwen3-next-80b")
    base_url = os.getenv("ABSTRACTAGENT_TEST_BASE_URL")

    llm_kwargs: Dict[str, Any] = {"temperature": 0}
    llm_kwargs["seed"] = 42
    if base_url:
        llm_kwargs["base_url"] = base_url
    return provider, model, llm_kwargs


@dataclass(frozen=True)
class ToolCallEvent:
    ts: str
    name: str
    arguments: Dict[str, Any]
    success: Optional[bool]
    error: Optional[str]


def _iter_tool_calls(traces: Dict[str, Any]) -> List[ToolCallEvent]:
    events: List[ToolCallEvent] = []
    for _node_id, trace in (traces or {}).items():
        steps = trace.get("steps") if isinstance(trace, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            eff = step.get("effect")
            if not isinstance(eff, dict):
                continue
            if eff.get("type") != "tool_calls":
                continue
            payload = eff.get("payload") if isinstance(eff.get("payload"), dict) else {}
            tool_calls = payload.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            result = step.get("result") if isinstance(step.get("result"), dict) else {}
            results = result.get("results") if isinstance(result.get("results"), list) else []

            for idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                name = str(tc.get("name") or "").strip()
                if not name:
                    continue
                args = tc.get("arguments")
                if not isinstance(args, dict):
                    args = {}

                success = None
                error = None
                if idx < len(results) and isinstance(results[idx], dict):
                    success = results[idx].get("success")
                    error = results[idx].get("error")
                    if success is not None:
                        success = bool(success)
                    if error is not None:
                        error = str(error)

                events.append(
                    ToolCallEvent(
                        ts=str(step.get("ts") or ""),
                        name=name,
                        arguments=dict(args),
                        success=success,
                        error=error,
                    )
                )

    events.sort(key=lambda e: e.ts)
    return events


def _json_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _summarize_tool_usage(events: List[ToolCallEvent]) -> Dict[str, Any]:
    tool_counts: Dict[str, int] = {}
    full_reads = 0
    sliced_reads = 0
    repeated_calls: Dict[str, int] = {}
    edit_pattern_sizes: List[int] = []
    failures: Dict[str, int] = {}

    seen_call_keys: Dict[str, int] = {}

    for e in events:
        tool_counts[e.name] = tool_counts.get(e.name, 0) + 1
        if e.success is False:
            failures[e.name] = failures.get(e.name, 0) + 1

        key = f"{e.name}:{_json_key(e.arguments)}"
        seen_call_keys[key] = seen_call_keys.get(key, 0) + 1

        if e.name == "read_file":
            should_entire = e.arguments.get("should_read_entire_file", True)
            start = (
                e.arguments.get("start_line")
                if e.arguments.get("start_line") is not None
                else e.arguments.get("start_line_one_indexed", e.arguments.get("start", 1))
            )
            end = (
                e.arguments.get("end_line")
                if e.arguments.get("end_line") is not None
                else e.arguments.get("end_line_one_indexed_inclusive", e.arguments.get("end"))
            )
            try:
                start_i = int(start or 1)
            except Exception:
                start_i = 1

            # If a range was requested (even via aliases), treat as a slice read.
            if end is not None or start_i != 1 or bool(should_entire) is False:
                sliced_reads += 1
            else:
                full_reads += 1

        if e.name == "edit_file":
            pattern = e.arguments.get("pattern")
            if isinstance(pattern, str):
                edit_pattern_sizes.append(len(pattern))

    for k, v in seen_call_keys.items():
        if v > 1:
            repeated_calls[k] = v

    return {
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "failures": dict(sorted(failures.items(), key=lambda kv: (-kv[1], kv[0]))),
        "read_file_full": full_reads,
        "read_file_sliced": sliced_reads,
        "repeated_calls": dict(sorted(repeated_calls.items(), key=lambda kv: (-kv[1], kv[0]))),
        "edit_pattern_sizes": {
            "count": len(edit_pattern_sizes),
            "max": max(edit_pattern_sizes) if edit_pattern_sizes else 0,
            "p95": int(sorted(edit_pattern_sizes)[max(0, int(len(edit_pattern_sizes) * 0.95) - 1)]) if edit_pattern_sizes else 0,
        },
    }


@dataclass(frozen=True)
class Scenario:
    name: str
    build: Callable[[Path], Dict[str, Any]]
    prompt: Callable[[Dict[str, Any]], str]
    verify: Callable[[Dict[str, Any]], Tuple[bool, str]]


def _make_python_scenario() -> Scenario:
    def build(root: Path) -> Dict[str, Any]:
        path = root / "main.py"

        filler = "\n".join([f"# filler {i}" for i in range(1, 460)])  # >400 lines to refuse full read_file
        path.write_text(
            "\n".join(
                [
                    "class Player:",
                    "    def __init__(self):",
                    "        # BUG: this should store the provided color",
                    "        self.color = None",
                    "",
                    "def main():",
                    "    p = Player('blue')",
                    "    return p.color",
                    "",
                    filler,
                    "",
                    "if __name__ == '__main__':",
                    "    print(main())",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return {"file_path": str(path)}

    def prompt(ctx: Dict[str, Any]) -> str:
        return (
            "Fix the following runtime error:\n"
            "TypeError: Player.__init__() takes 1 positional argument but 2 were given\n\n"
            f"Target file: {ctx['file_path']}\n\n"
            "Constraints:\n"
            "- Do NOT try to read the entire file (it is >400 lines; read_file(full) will refuse).\n"
            "- Use analyze_code() first to locate the Player definition.\n"
            "- Then use read_file(start_line/end_line) around the relevant blocks.\n"
            "- Apply small, surgical edit_file() calls (short patterns; max_replacements=1).\n\n"
            "Expected fix:\n"
            "- Player.__init__ should accept an optional color parameter and set self.color\n"
            "- Replace the \"self.color = None\" line with \"self.color = color\".\n"
            "- Keep behavior otherwise unchanged.\n"
        )

    def verify(ctx: Dict[str, Any]) -> Tuple[bool, str]:
        text = Path(ctx["file_path"]).read_text(encoding="utf-8")
        def _sig_has_color(start_idx: int) -> bool:
            if start_idx < 0:
                return False
            window = text[start_idx : start_idx + 300]
            # Check only inside the signature portion (best-effort).
            sig_end = window.find("):")
            if sig_end == -1:
                sig_end = window.find("):\n")
            if sig_end == -1:
                sig_end = window.find("):\r\n")
            if sig_end == -1:
                sig_end = min(len(window), 120)
            signature = window[:sig_end]
            return "color" in signature

        player_idx = text.find("class Player")
        ok_player = player_idx != -1 and _sig_has_color(text.find("def __init__", player_idx))
        if ok_player:
            window = text[player_idx : player_idx + 300]
            ok_player = "self.color = color" in window or "self.color=color" in window

        if ok_player:
            return True, "Player constructor updated with color parameter."
        return False, "Did not find expected __init__ signature/body update for Player."

    return Scenario(name="python_ctor_mismatch_large_file", build=build, prompt=prompt, verify=verify)


def _make_js_scenario() -> Scenario:
    def build(root: Path) -> Dict[str, Any]:
        path = root / "app.js"
        filler = "\n".join([f"// filler {i}" for i in range(1, 460)])  # >400 lines to discourage full read_file
        path.write_text(
            "\n".join(
                [
                    "export function greet(name) {",
                    "  return `Hello ${name}`;",
                    "}",
                    "",
                    "export function run() {",
                    "  // BUG: wrong function name",
                    "  return greets('world');",
                    "}",
                    "",
                    filler,
                    "",
                    "console.log(run());",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {"file_path": str(path)}

    def prompt(ctx: Dict[str, Any]) -> str:
        return (
            "Fix the following JavaScript error:\n"
            "ReferenceError: greets is not defined\n\n"
            f"Target file: {ctx['file_path']}\n\n"
            "Constraints:\n"
            "- Use analyze_code() first to locate greet/run.\n"
            "- Use read_file(start_line/end_line) around the run() function.\n"
            "- Make a minimal edit_file() change: call greet('world') instead of greets('world').\n"
        )

    def verify(ctx: Dict[str, Any]) -> Tuple[bool, str]:
        text = Path(ctx["file_path"]).read_text(encoding="utf-8")
        if "return greet('world');" in text:
            return True, "Fixed call to greet()."
        return False, "Did not find expected replacement greets(...) -> greet(...)."

    return Scenario(name="js_reference_error", build=build, prompt=prompt, verify=verify)


def _print_heading(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _run_scenarios(*, scenarios: List[Scenario]) -> int:
    provider, model, llm_kwargs = _llm_config()
    _print_heading(f"LMStudio Tool Eval | provider={provider} | model={model} | base_url={llm_kwargs.get('base_url') or '(default)'}")

    from abstractagent.agents.react import create_react_agent

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = Path(os.getenv("AF_TOOL_EVAL_OUT_DIR", "test_results/tool_eval")).expanduser().absolute()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"lmstudio_tool_eval_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []

    for s in scenarios:
        with tempfile.TemporaryDirectory(prefix="af_tool_eval_") as td:
            root = Path(td)
            ctx = s.build(root)
            scenario_dir = run_dir / s.name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            agent = create_react_agent(
                provider=provider,
                model=model,
                llm_kwargs=llm_kwargs,
                max_iterations=20,
                max_tokens=8192,
            )

            agent.start(s.prompt(ctx))
            state = agent.run_to_completion()

            traces = agent.get_node_traces()
            events = _iter_tool_calls(traces)
            summary = _summarize_tool_usage(events)

            ok, msg = s.verify(ctx)
            answer = ""
            try:
                answer = str((state.output or {}).get("answer") or "")
            except Exception:
                answer = ""

            record = {
                "scenario": s.name,
                "status": getattr(state.status, "value", str(state.status)),
                "verify_ok": ok,
                "verify_msg": msg,
                "tool_summary": summary,
                "answer": answer,
            }
            aggregate.append(record)

            _print_heading(f"Scenario: {s.name}")
            print(f"Run status: {record['status']} | verify_ok={ok}")
            print(f"Verify: {msg}")
            print("Tool counts:", summary["tool_counts"])
            if summary["failures"]:
                print("Tool failures:", summary["failures"])
            print(f"read_file(full)={summary['read_file_full']} | read_file(slice)={summary['read_file_sliced']}")
            if summary["repeated_calls"]:
                # Print only the top few repeated calls to keep output small.
                top = list(summary["repeated_calls"].items())[:3]
                print("Repeated identical calls (top 3):", {k: v for k, v in top})

            # Persist scenario traces + final file snapshot for deep inspection.
            try:
                (scenario_dir / "node_traces.json").write_text(_json_key(traces), encoding="utf-8")
                (scenario_dir / "summary.json").write_text(_json_key(record), encoding="utf-8")
                file_path = Path(str(ctx.get("file_path") or "")).expanduser()
                if file_path.exists() and file_path.is_file():
                    (scenario_dir / f"final_{file_path.name}").write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

    # Aggregate: report issues that repeat across scenarios.
    _print_heading("Aggregate (repetitive-only)")
    repeats: Dict[str, int] = {}
    for r in aggregate:
        ts = r.get("tool_summary") or {}
        if isinstance(ts, dict):
            if (ts.get("read_file_full") or 0) > 0:
                repeats["read_file_full_used"] = repeats.get("read_file_full_used", 0) + 1
            tool_counts = ts.get("tool_counts") or {}
            if isinstance(tool_counts, dict) and tool_counts.get("analyze_code", 0) == 0:
                repeats["analyze_code_not_used"] = repeats.get("analyze_code_not_used", 0) + 1
            failures = ts.get("failures") or {}
            if isinstance(failures, dict) and failures.get("edit_file", 0) > 0:
                repeats["edit_file_failures"] = repeats.get("edit_file_failures", 0) + 1

    if not repeats:
        print("No repetitive issues detected across scenarios (based on basic heuristics).")
    else:
        for k, v in sorted(repeats.items(), key=lambda kv: (-kv[1], kv[0])):
            if v > 1:
                print(f"- {k}: occurred in {v}/{len(aggregate)} scenarios")

    print(f"\nSaved eval artifacts to: {run_dir}")

    # Exit code: non-zero if any scenario verification failed.
    failed = [r for r in aggregate if not r.get("verify_ok")]
    if failed:
        print(f"\nFailures: {len(failed)}/{len(aggregate)} scenarios did not meet verification checks.")
        for r in failed:
            print(f"- {r['scenario']}: status={r['status']} verify={r['verify_msg']}")
        return 1
    return 0


def main() -> int:
    scenarios = [_make_python_scenario(), _make_js_scenario()]
    return _run_scenarios(scenarios=scenarios)


if __name__ == "__main__":
    raise SystemExit(main())
