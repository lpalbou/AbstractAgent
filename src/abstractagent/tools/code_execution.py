"""Code execution tool used by CodeAct agents."""

from __future__ import annotations

from typing import Optional

from abstractcore.tools import tool


# ADR-0026: this tool's stdout/stderr ARE the observation the model reads on the
# next turn. A 6000-char head+tail clip used to sit here and cut the MIDDLE out of
# every long traceback and every long print — the exact bytes the model needed.
# Tool output now flows WHOLE. A caller that genuinely wants a bound passes
# `max_output_chars` explicitly, and the clip is loud and in-band.


def _bound_output(text: str, *, max_chars: int) -> str:
    """Caller-requested, marked bound. `max_chars <= 0` (the default) = no bound."""
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    #[WARNING:TRUNCATION] explicit caller-supplied execute_python bound (max_output_chars)
    return text[:max_chars] + (
        f"\n... [#WARNING:TRUNCATION {len(text) - max_chars} of {len(text)} chars dropped by the "
        "caller-supplied max_output_chars bound on execute_python]"
    )


# Round 12: the one sentence when the installed AbstractCore has no command sandbox.
NO_CORE_SANDBOX = (
    "Commands are refused: the installed AbstractCore has no command sandbox "
    "(abstractcore.tools.sandbox, AbstractCore 2.25.0 or newer)."
)


@tool(
    name="execute_python",
    description="Execute a Python snippet in a subprocess, sandboxed to this run's workspaces. Returns stdout/stderr/exit_code in full.",
    when_to_use="When you need to compute, inspect files, or transform data using Python code",
    # Stamped by the host (AbstractRuntime) with the run's workspace set; never model-facing.
    hide_args=["_sandbox"],
)
def execute_python(code: str, timeout_s: float = 0.0, max_output_chars: int = 0, _sandbox: Optional[dict] = None) -> dict:
    """Execute Python code in a local subprocess.

    Notes:
    - Round 12: when the host stamps `_sandbox` (the run's effective workspace set), the
      interpreter runs inside AbstractCore's OS sandbox (macOS sandbox-exec, Linux bwrap/Landlock),
      starting in the run's private workspace with the host's scrubbed environment. No sandbox
      on the host (or an AbstractCore without one) = refused (`error` carries the sentence,
      nothing runs). A host that configured AbstractCore's command policy refuses an unstamped
      call. Library use without a host policy runs as before.
    - `timeout_s <= 0` (the default) means NO timeout (ADR-0027: no low default
      timeouts on correctness-critical paths — a 10s default silently killed
      legitimate long computations and reported them as errors). Pass a positive
      value to opt into a safeguard.
    - `max_output_chars <= 0` (the default) means output is returned WHOLE
      (ADR-0026).
    """
    code = str(code or "")
    if not code.strip():
        raise ValueError("code must be a non-empty string")

    from ..sandbox.local import LocalSandbox

    try:
        from abstractcore.tools import sandbox as core_sandbox
    except ImportError:
        return {"stdout": "", "stderr": "", "exit_code": 126, "duration_ms": 0.0, "error": NO_CORE_SANDBOX, "success": False}
    box = core_sandbox.sandbox_for_tool_call(_sandbox)
    if box.refuses:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 126,
            "duration_ms": 0.0,
            "error": box.refusal(),
            "success": False,
            "sandbox": box.describe(),
            "sandbox_line": box.rendered_line(),
        }
    bound = box if (box.spec is not None or box.kind != core_sandbox.KIND_UNSANDBOXED) else None
    sandbox = LocalSandbox(cwd=box.spec.private_workspace if box.spec is not None else None, command_sandbox=bound)
    #[WARNING:TIMEOUT] caller-supplied only; <= 0 = unlimited (ADR-0027)
    timeout = float(timeout_s)
    result = sandbox.execute(code, timeout_s=timeout if timeout > 0 else None)
    try:
        cap = int(max_output_chars)
    except (TypeError, ValueError):
        cap = 0
    out = {
        "stdout": _bound_output(result.stdout, max_chars=cap),
        "stderr": _bound_output(result.stderr, max_chars=cap),
        "exit_code": int(result.exit_code),
        "duration_ms": float(result.duration_ms),
        "error": result.error,
    }
    if bound is not None:
        out["sandbox"] = bound.describe()
        out["sandbox_line"] = bound.rendered_line()
    return out

