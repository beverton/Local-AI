"""
Tool Registry - single source of truth for tool runtime + validation/policy.

This is used by the streaming ToolAgentLoop (and can be reused by ChatAgent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import json
import re


@dataclass(frozen=True)
class ToolSpec:
    name: str
    execute: Callable[..., Any]
    # if True, execution must be confirmed by user
    requires_confirmation: bool = False


def _sanitize_path(p: str) -> str:
    if not p:
        return ""
    p = str(p).strip().strip('"\''"`")
    p = re.sub(r"[\s\)\]\}\.,;:!?]+$", "", p).strip()
    return p.strip('"\''"`")


def _sanitize_query(q: str) -> str:
    if not q:
        return ""
    q = str(q).strip()
    # remove surrounding quotes
    q = q.strip('"\''"`")
    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    # avoid extremely long prompts sent to search APIs
    if len(q) > 200:
        q = q[:200].rstrip()
    return q


def get_tool_registry(enable_web_search: bool = True) -> Dict[str, ToolSpec]:
    """
    Returns runtime tool registry. This assumes agent_tools.initialize_tools()
    was called during app startup.
    """
    from agent_tools import (
        read_file,
        write_file,
        list_directory,
        delete_file,
        file_exists,
        web_search,
    )

    reg: Dict[str, ToolSpec] = {
        "read_file": ToolSpec(name="read_file", execute=read_file, requires_confirmation=False),
        "write_file": ToolSpec(name="write_file", execute=write_file, requires_confirmation=False),
        "list_directory": ToolSpec(name="list_directory", execute=list_directory, requires_confirmation=False),
        # delete_file must always be confirmed (per user decision)
        "delete_file": ToolSpec(name="delete_file", execute=delete_file, requires_confirmation=True),
        "file_exists": ToolSpec(name="file_exists", execute=file_exists, requires_confirmation=False),
    }
    if enable_web_search:
        reg["web_search"] = ToolSpec(name="web_search", execute=web_search, requires_confirmation=False)
    return reg


def normalize_tool_call(name: str, arguments: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Normalizes tool arguments and applies lightweight sanitizers.
    """
    tool_name = (name or "").strip()
    args: Dict[str, Any] = arguments if isinstance(arguments, dict) else {}

    if tool_name in ("read_file", "write_file", "delete_file", "file_exists"):
        if "file_path" in args:
            args["file_path"] = _sanitize_path(args.get("file_path"))
        if tool_name == "write_file" and "content" in args and isinstance(args.get("content"), str):
            # preserve whitespace, but avoid accidental None/"null"
            if args["content"].strip().lower() in ("none", "null"):
                args["content"] = ""
    if tool_name == "list_directory":
        if "directory_path" in args:
            args["directory_path"] = _sanitize_path(args.get("directory_path"))
    if tool_name == "web_search":
        if "query" in args:
            args["query"] = _sanitize_query(args.get("query"))

    return tool_name, args


def tool_result_preview(result: Any, limit: int = 400) -> str:
    """
    Small preview suitable for SSE tool_result event.
    """
    try:
        if isinstance(result, (dict, list)):
            s = json.dumps(result, ensure_ascii=False)
        else:
            s = str(result)
    except Exception:
        s = "<unserializable>"
    if len(s) > limit:
        return s[:limit] + "…"
    return s

