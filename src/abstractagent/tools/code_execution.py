"""Code execution tool used by CodeAct agents."""

from __future__ import annotations

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


@tool(
    name="execute_python",
    description="Execute a Python snippet in a subprocess sandbox. Returns stdout/stderr/exit_code in full.",
    when_to_use="When you need to compute, inspect files, or transform data using Python code",
)
def execute_python(code: str, timeout_s: float = 0.0, max_output_chars: int = 0) -> dict:
    """Execute Python code in a local subprocess.

    Notes:
    - This is a dev sandbox. It is not a hardened security boundary.
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

    sandbox = LocalSandbox()
    #[WARNING:TIMEOUT] caller-supplied only; <= 0 = unlimited (ADR-0027)
    timeout = float(timeout_s)
    result = sandbox.execute(code, timeout_s=timeout if timeout > 0 else None)
    try:
        cap = int(max_output_chars)
    except (TypeError, ValueError):
        cap = 0
    return {
        "stdout": _bound_output(result.stdout, max_chars=cap),
        "stderr": _bound_output(result.stderr, max_chars=cap),
        "exit_code": int(result.exit_code),
        "duration_ms": float(result.duration_ms),
        "error": result.error,
    }

