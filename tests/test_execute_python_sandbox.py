"""Round 12: execute_python runs inside AbstractCore's OS command sandbox.

Real macOS tests (sandbox-exec) with scratch folders and markers; the runtime path
(AbstractRuntime stamps `_sandbox` on the CodeAct tool call) is exercised end to end."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as core_sandbox
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec

from abstractagent.tools.code_execution import NO_CORE_SANDBOX, execute_python

MARK = "R12-AGENT-REFUSED-MARKER"

macos_only = pytest.mark.skipif(
    sys.platform != "darwin" or not os.access(core_sandbox.SANDBOX_EXEC, os.X_OK),
    reason="real sandbox test: this host is not macOS with /usr/bin/sandbox-exec (Linux backends are tested in abstractcore)",
)


@pytest.fixture(autouse=True)
def _fresh_host():
    core_sandbox._reset_host_for_tests()
    yield
    core_sandbox._reset_host_for_tests()


@pytest.fixture
def t(tmp_path, monkeypatch):
    root = Path(os.path.realpath(tmp_path))
    for d in ("home", "priv", "refused", "rw"):
        (root / d).mkdir()
    (root / "refused/secret.txt").write_text(MARK)
    monkeypatch.setenv("HOME", str(root / "home"))
    return root


def _stamp(t: Path, refused=True):
    return SandboxSpec(
        private_workspace=str(t / "priv"),
        posture="any_except_denied",
        allowed=(SandboxRow(str(t / "rw"), "rw"),),
        refused=(str(t / "refused"),) if refused else (),
    ).to_stamp()


def test_sandbox_arg_is_hidden_from_the_model():
    assert "_sandbox" not in execute_python._tool_definition.parameters


@macos_only
def test_refused_path_is_unreadable_from_python(t):
    code = f"print(open({str(t / 'refused/secret.txt')!r}).read())"
    res = execute_python(code, _sandbox=_stamp(t))
    assert MARK not in res["stdout"] and res["exit_code"] != 0
    assert "Operation not permitted" in res["stderr"]
    assert res["sandbox"]["kind"] == "macos-sandbox-exec" and res["sandbox_line"] == "Sandbox: macOS sandbox-exec"
    ok = execute_python(code, _sandbox=_stamp(t, refused=False))
    assert MARK in ok["stdout"]  # positive control


@macos_only
def test_starts_in_the_private_workspace_with_a_private_tmpdir(t):
    res = execute_python("import os,tempfile;open('w.txt','w').write('x');print(os.getcwd());print(tempfile.gettempdir())", _sandbox=_stamp(t))
    assert res["stdout"].split() == [str(t / "priv"), str(t / "priv/.tmp")]
    assert (t / "priv/w.txt").exists()


def test_fail_closed_without_a_sandbox(t, monkeypatch):
    monkeypatch.setattr(core_sandbox, "host_sandbox_kind", lambda posture="allowed_only": core_sandbox.KIND_NONE)
    res = execute_python(f"open({str(t / 'priv/ran')!r},'w')", _sandbox=_stamp(t))
    assert res["success"] is False and res["error"].startswith("Commands are not sandboxed on this gateway host")
    assert not (t / "priv/ran").exists()


def test_fail_closed_without_core_sandbox_module(t, monkeypatch):
    import abstractcore.tools as core_tools

    monkeypatch.setitem(sys.modules, "abstractcore.tools.sandbox", None)
    monkeypatch.delattr(core_tools, "sandbox", raising=False)
    res = execute_python(f"open({str(t / 'priv/ran')!r},'w')", _sandbox=_stamp(t))
    assert res["success"] is False and res["error"] == NO_CORE_SANDBOX
    assert not (t / "priv/ran").exists()


def test_configured_host_refuses_unstamped_call(t):
    core_sandbox.configure_host(env={"PATH": "/usr/bin:/bin"})
    res = execute_python(f"open({str(t / 'priv/ran')!r},'w')")
    assert res["success"] is False and res["error"] == core_sandbox.REFUSED_NO_SPEC
    assert not (t / "priv/ran").exists()


@macos_only
def test_codeact_tool_call_through_the_runtime_is_sandboxed(t):
    """A CodeAct-style TOOL_CALLS effect: the runtime stamps the run's workspaces on
    execute_python; reading a refused workspace fails inside the sandbox."""
    from abstractruntime.core.models import Effect, EffectType, RunState
    from abstractruntime.integrations.abstractcore.effect_handlers import make_tool_calls_handler
    from abstractruntime.integrations.abstractcore.tool_executor import MappingToolExecutor

    handler = make_tool_calls_handler(tools=MappingToolExecutor.from_tools([execute_python]))
    run_vars = {
        "workspace_root": str(t / "priv"),
        "workspace_access_mode": "all_except_ignored",
        "workspace_allowed_paths": [str(t / "rw")],
        "workspace_ignored_paths": str(t / "refused"),
    }
    run = RunState.new(workflow_id="wf", entry_node="n", session_id="s", vars=run_vars)
    code = f"print(open({str(t / 'refused/secret.txt')!r}).read())"
    effect = Effect(type=EffectType.TOOL_CALLS, payload={"tool_calls": [{"call_id": "c1", "name": "execute_python", "arguments": {"code": code, "_sandbox": {"private_workspace": "/", "posture": "any_except_denied"}}}]})
    outcome = handler(run, effect, None)
    res = outcome.result["results"][0]
    out = res["output"]
    assert MARK not in str(out)
    assert out["sandbox"]["kind"] == "macos-sandbox-exec" and out["sandbox"]["refused"] == [str(t / "refused")]
