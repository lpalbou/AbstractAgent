"""AbstractAgent - Agent implementations using AbstractRuntime and AbstractCore."""

from .adapters.loop_hooks import DEFAULT_EVENT_MAP, HookEvent, LoopHooks
from .adapters.native_loop_registry import (
    audit_native_loop_manifest,
    build_native_loop_manifest,
    materialize_native_loop_spec,
    pack_native_loop_bundle,
)
from .agents import (
    BaseAgent,
    ReactAgent,
    create_react_workflow,
    create_react_agent,
    CodeActAgent,
    create_codeact_workflow,
    create_codeact_agent,
)
from .tools import (
    ALL_TOOLS,
    list_files,
    read_file,
    search_files,
    execute_command,
    write_file,
    edit_file,
    web_search,
    fetch_url,
    execute_python,
    self_improve,
)

__all__ = [
    # Base class for custom agents
    "BaseAgent",
    # First-class loop hooks (listen + steer + capture)
    "DEFAULT_EVENT_MAP",
    "HookEvent",
    "LoopHooks",
    "audit_native_loop_manifest",
    "materialize_native_loop_spec",
    "build_native_loop_manifest",
    "pack_native_loop_bundle",
    # ReAct agent
    "ReactAgent",
    "create_react_workflow",
    "create_react_agent",
    # CodeAct agent
    "CodeActAgent",
    "create_codeact_workflow",
    "create_codeact_agent",
    # Tools
    "ALL_TOOLS",
    "list_files",
    "read_file",
    "search_files",
    "execute_command",
    "write_file",
    "edit_file",
    "web_search",
    "fetch_url",
    "execute_python",
    "self_improve",
]
