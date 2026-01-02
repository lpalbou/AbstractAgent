"""AbstractAgent tools.

Common tools are imported from AbstractCore (canonical source).
Agent-specific tools (execute_python, self_improve) are defined locally.
"""

# Import common tools from AbstractCore (canonical source)
from abstractcore.tools.common_tools import (
    list_files,
    analyze_code,
    read_file,
    search_files,
    write_file,
    edit_file,
    web_search,
    fetch_url,
    execute_command,
)

# Agent-specific tools
from .code_execution import execute_python
from .self_improve import self_improve

# Default toolset for agents
ALL_TOOLS = [
    # File operations (from abstractcore)
    list_files,
    analyze_code,
    read_file,
    search_files,
    write_file,
    edit_file,
    # Web tools (from abstractcore)
    web_search,
    fetch_url,
    # System tools (from abstractcore)
    execute_command,
    # Agent-specific tools
    execute_python,
    self_improve,
]

__all__ = [
    # File operations
    "list_files",
    "analyze_code",
    "read_file",
    "search_files",
    "write_file",
    "edit_file",
    # Web tools
    "web_search",
    "fetch_url",
    # System tools
    "execute_command",
    # Agent-specific tools
    "execute_python",
    "self_improve",
    # Collections
    "ALL_TOOLS",
]
