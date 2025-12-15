from __future__ import annotations

import json
import os
from pathlib import Path

from abstractagent.tools.filesystem import update_file, write_file
from abstractagent.tools.self_improve import self_improve


def test_write_file_creates_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"

    out1 = write_file(str(target), "hi\n", overwrite=False)
    assert "Wrote" in out1
    assert target.read_text(encoding="utf-8") == "hi\n"

    out2 = write_file(str(target), "bye\n", overwrite=False)
    assert "already exists" in out2
    assert target.read_text(encoding="utf-8") == "hi\n"

    out3 = write_file(str(target), "bye\n", overwrite=True)
    assert "Wrote" in out3
    assert target.read_text(encoding="utf-8") == "bye\n"


def test_update_file_applies_unified_diff(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    patch = """--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 hello
-world
+there
"""
    out = update_file(str(target), patch)
    assert out.startswith("Updated ")
    assert target.read_text(encoding="utf-8") == "hello\nthere\n"


def test_update_file_rejects_header_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    patch = """--- a/other.txt
+++ b/other.txt
@@ -1,2 +1,2 @@
 hello
-world
+there
"""
    out = update_file(str(target), patch)
    assert "header does not match" in out.lower()


def test_self_improve_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    out_file = tmp_path / "improvements.jsonl"
    monkeypatch.setenv("ABSTRACTFRAMEWORK_IMPROVEMENTS_PATH", str(out_file))

    msg = self_improve("Add X", target="y", category="tooling", tags={"k": "v"})
    assert "Logged improvement suggestion" in msg
    assert out_file.exists()

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines
    data = json.loads(lines[-1])
    assert data["category"] == "tooling"
    assert data["target"] == "y"
    assert data["suggestion"] == "Add X"
    assert data["tags"] == {"k": "v"}

