"""Filesystem tools for the ReAct agent.

Simple tools for listing, searching, and reading files.
Uses AbstractCore's @tool decorator for registration.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

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


# Export all tools
ALL_TOOLS = [list_files, read_file, search_files, execute_command]
