from __future__ import annotations

import json
import os
from pathlib import Path

from abstractcore.tools.common_tools import edit_file, write_file
from abstractagent.tools.self_improve import self_improve


def test_write_file_creates_and_overwrites(tmp_path: Path) -> None:
    """Test write_file creates files and can overwrite.

    Note: abstractcore's write_file uses mode='w' (default) to overwrite,
    not an 'overwrite' parameter.
    """
    target = tmp_path / "hello.txt"

    # Create file
    out1 = write_file(str(target), "hi\n")
    assert "Successfully" in out1 or "wrote" in out1.lower()
    assert target.read_text(encoding="utf-8") == "hi\n"

    # Overwrite file (mode='w' is default)
    out2 = write_file(str(target), "bye\n")
    assert "Successfully" in out2 or "wrote" in out2.lower()
    assert target.read_text(encoding="utf-8") == "bye\n"

    # Append to file
    out3 = write_file(str(target), " more\n", mode="a")
    assert "appended" in out3.lower() or "successfully" in out3.lower()
    assert target.read_text(encoding="utf-8") == "bye\n more\n"


def test_edit_file_applies_unified_diff(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    patch = """--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 hello
-world
+there
"""
    out = edit_file(str(target), patch)
    assert out.startswith("Edited ")
    assert target.read_text(encoding="utf-8") == "hello\nthere\n"


def test_edit_file_rejects_header_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    patch = """--- a/other.txt
+++ b/other.txt
@@ -1,2 +1,2 @@
 hello
-world
+there
"""
    out = edit_file(str(target), patch)
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
