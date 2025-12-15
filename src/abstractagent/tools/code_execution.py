"""Code execution tool used by CodeAct agents."""

from __future__ import annotations

from abstractcore.tools import tool


def _truncate(text: str, *, limit: int = 6000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[:4000]
    tail = text[-1500:] if len(text) > 5500 else ""
    return head + f"\n... (truncated, {len(text)} chars total)\n" + tail


@tool(
    name="execute_python",
    description="Execute a Python snippet in a subprocess sandbox (timeout enforced). Returns stdout/stderr/exit_code.",
    when_to_use="When you need to compute, inspect files, or transform data using Python code",
)
def execute_python(code: str, timeout_s: float = 10.0) -> dict:
    """Execute Python code in a local subprocess.

    Notes:
    - This is a dev sandbox (timeout only). It is not a hardened security boundary.
    - Use small snippets and print what you need.
    """
    code = str(code or "")
    if not code.strip():
        raise ValueError("code must be a non-empty string")

    from ..sandbox.local import LocalSandbox

    sandbox = LocalSandbox()
    result = sandbox.execute(code, timeout_s=float(timeout_s))
    return {
        "stdout": _truncate(result.stdout),
        "stderr": _truncate(result.stderr),
        "exit_code": int(result.exit_code),
        "duration_ms": float(result.duration_ms),
        "error": result.error,
    }

