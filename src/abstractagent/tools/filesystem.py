"""Filesystem tools for the ReAct agent.

Simple tools for listing, searching, reading, and editing files.
Uses AbstractCore's @tool decorator for registration.
"""

import difflib
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from abstractcore.tools import tool


@tool(
    name="list_files",
    description="List files and directories in a path. Returns names only, not full paths.",
    when_to_use="When you need to see what files exist in a directory",
)
def list_files(path: str = ".") -> str:
    """List files and directories in the given path."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        if not p.is_dir():
            return f"Error: '{path}' is not a directory"
        
        items = sorted(p.iterdir())
        dirs = [f"{item.name}/" for item in items if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in items if item.is_file() and not item.name.startswith('.')]
        
        result = []
        if dirs:
            result.append("Directories: " + ", ".join(dirs[:20]))
        if files:
            result.append("Files: " + ", ".join(files[:20]))
        
        if not result:
            return "Directory is empty"
        return "\n".join(result)
    except PermissionError:
        return f"Error: Permission denied for '{path}'"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="read_file",
    description="Read the contents of a file. Returns the first 2000 characters.",
    when_to_use="When you need to see the contents of a specific file",
)
def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist"
        if not p.is_file():
            return f"Error: '{path}' is not a file"
        
        content = p.read_text(encoding='utf-8', errors='replace')
        if len(content) > 2000:
            return content[:2000] + f"\n... (truncated, {len(content)} total chars)"
        return content
    except PermissionError:
        return f"Error: Permission denied for '{path}'"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="search_files",
    description="Search for files matching a pattern in a directory. Uses glob patterns like '*.py' or '**/*.md'.",
    when_to_use="When you need to find files matching a pattern",
)
def search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching a glob pattern."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: Path '{path}' does not exist"
        
        matches = list(p.glob(pattern))[:20]  # Limit results
        
        if not matches:
            return f"No files found matching '{pattern}'"
        
        results = [str(m.relative_to(p)) for m in matches]
        return "Found:\n" + "\n".join(results)
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="execute_command",
    description="Execute a shell command and return the output. Use with caution.",
    when_to_use="When you need to run a command like 'pwd', 'echo', 'cat', etc.",
)
def execute_command(command: str) -> str:
    """Execute a shell command."""
    # Safety: block dangerous commands
    dangerous = ['rm -rf', 'sudo', 'chmod', 'chown', '>', '>>', '|', ';', '&&', 'curl', 'wget']
    for d in dangerous:
        if d in command.lower():
            return f"Error: Command contains blocked pattern '{d}'"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.getcwd(),
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nStderr: {result.stderr}"
        
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        
        if not output.strip():
            return "(no output)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="write_file",
    description="Write UTF-8 text to a file (creates parent directories). By default refuses to overwrite unless overwrite=true.",
    when_to_use="When you need to create a new file or fully overwrite a file with known content",
)
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write text content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        if p.exists() and not overwrite:
            return f"Error: File '{path}' already exists (set overwrite=true to replace it)"
        if p.exists() and not p.is_file():
            return f"Error: '{path}' exists but is not a file"

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

        lines = content.count("\n") + 1 if content else 0
        return f"Wrote {len(content)} chars ({lines} lines) to {str(p)}"
    except PermissionError:
        return f"Error: Permission denied for '{path}'"
    except Exception as e:
        return f"Error: {e}"


_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _normalize_diff_path(raw: str) -> str:
    raw = raw.strip()
    # Remove timestamps and extra markers (common in diff headers)
    raw = raw.split("\t", 1)[0].strip()
    raw = raw.split(" ", 1)[0].strip()
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    return raw


def _path_parts(path_str: str) -> Tuple[str, ...]:
    normalized = path_str.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    return tuple(parts)


def _is_suffix_path(candidate: str, target: Path) -> bool:
    candidate_parts = _path_parts(candidate)
    if not candidate_parts:
        return False
    target_parts = tuple(target.as_posix().split("/"))
    return len(candidate_parts) <= len(target_parts) and target_parts[-len(candidate_parts) :] == candidate_parts


def _parse_unified_diff(patch: str) -> Tuple[Optional[str], List[Tuple[int, int, int, int, List[str]]], Optional[str]]:
    """
    Parse a unified diff for a single file.

    Returns:
      (header_path, hunks, error)
    """
    lines = patch.splitlines()
    header_path: Optional[str] = None
    hunk_list: List[Tuple[int, int, int, int, List[str]]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("--- "):
            old_path = _normalize_diff_path(line[4:])
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                return None, [], "Invalid unified diff: missing '+++ ' header after '--- '"
            new_path = _normalize_diff_path(lines[i][4:])
            # Accept /dev/null only for create/delete diffs (not supported here)
            if old_path != "/dev/null" and new_path != "/dev/null":
                if header_path is None:
                    header_path = new_path
                elif header_path != new_path:
                    return None, [], "Unified diff appears to reference multiple files"
            i += 1
            continue

        if line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(line)
            if not m:
                return header_path, [], f"Invalid hunk header: {line}"

            old_start = int(m.group(1))
            old_len = int(m.group(2) or 1)
            new_start = int(m.group(3))
            new_len = int(m.group(4) or 1)

            i += 1
            hunk_lines: List[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("@@") or nxt.startswith("--- ") or nxt.startswith("diff --git "):
                    break
                hunk_lines.append(nxt)
                i += 1

            hunk_list.append((old_start, old_len, new_start, new_len, hunk_lines))
            continue

        i += 1

    if not hunk_list:
        return header_path, [], "No hunks found in diff (missing '@@ ... @@' sections)"

    return header_path, hunk_list, None


def _apply_unified_diff(original_text: str, hunks: List[Tuple[int, int, int, int, List[str]]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply unified diff hunks to text.

    Returns:
      (new_text, error)
    """
    ends_with_newline = original_text.endswith("\n")
    original_lines = original_text.splitlines()

    out: List[str] = []
    cursor = 0  # index into original_lines

    for old_start, _old_len, _new_start, _new_len, hunk_lines in hunks:
        hunk_start = max(old_start - 1, 0)
        if hunk_start > len(original_lines):
            return None, f"Hunk starts beyond end of file (start={old_start}, lines={len(original_lines)})"

        # Copy unchanged prefix
        out.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start

        for hl in hunk_lines:
            if hl == r"\ No newline at end of file":
                continue
            if not hl:
                # Empty lines are represented as a prefix char with empty remainder (" ", "-", "+")
                return None, "Invalid diff line: empty line without prefix"

            prefix = hl[0]
            text = hl[1:]

            if prefix == " ":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    got = original_lines[cursor] if cursor < len(original_lines) else "<EOF>"
                    return None, f"Context mismatch applying patch. Expected {text!r}, got {got!r}"
                out.append(text)
                cursor += 1
            elif prefix == "-":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    got = original_lines[cursor] if cursor < len(original_lines) else "<EOF>"
                    return None, f"Remove mismatch applying patch. Expected {text!r}, got {got!r}"
                cursor += 1
            elif prefix == "+":
                out.append(text)
            else:
                return None, f"Invalid diff line prefix {prefix!r} (expected one of ' ', '+', '-')"

    # Copy any remaining tail
    out.extend(original_lines[cursor:])

    new_text = "\n".join(out)
    if ends_with_newline and not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, None


@tool(
    name="update_file",
    description="Update an existing UTF-8 text file by applying a unified diff patch (single file). Returns a small diff preview of what changed.",
    when_to_use="When you need to make precise edits to an existing file without rewriting the entire file",
)
def update_file(path: str, patch: str) -> str:
    """Apply a unified diff patch to a file."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist"
        if not p.is_file():
            return f"Error: '{path}' is not a file"

        header_path, hunks, err = _parse_unified_diff(patch)
        if err:
            return f"Error: {err}"
        if header_path and not _is_suffix_path(header_path, p):
            return (
                "Error: Patch file header does not match the provided path.\n"
                f"Patch header: {header_path}\n"
                f"Target path:  {p}\n"
                "Generate a unified diff targeting the exact file you want to update."
            )

        original = p.read_text(encoding="utf-8", errors="replace")
        updated, apply_err = _apply_unified_diff(original, hunks)
        if apply_err:
            return f"Error: Patch did not apply cleanly: {apply_err}"

        assert updated is not None
        if updated == original:
            return "No changes applied (patch resulted in identical content)."

        p.write_text(updated, encoding="utf-8")

        old_lines = original.splitlines()
        new_lines = updated.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=str(p),
                tofile=str(p),
                lineterm="",
                n=3,
            )
        )
        preview = "\n".join(diff_lines[:120])
        if len(diff_lines) > 120:
            preview += f"\n... (diff truncated, {len(diff_lines)} lines total)"

        return f"Updated {str(p)}\n{preview}"
    except PermissionError:
        return f"Error: Permission denied for '{path}'"
    except Exception as e:
        return f"Error: {e}"
