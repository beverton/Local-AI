"""
Tool Definitions - Zentrale Tool-Schemas (OpenAI-Format)

Diese Definitionen werden für LLM-basiertes Tool-/Function-Calling genutzt.
Parameter-Namen sind kompatibel mit den vorhandenen Agent-Tools in `backend/agent_tools.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_openai_tool_definitions(enable_web_search: bool = True) -> List[Dict[str, Any]]:
    """
    Returns tool definitions in OpenAI tool-calling format.

    Note: The runtime tool execution is implemented via AgentManager/agent_tools.
    This function only provides schemas for the model.
    """
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Liest den Inhalt einer Datei (nur innerhalb des Workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Pfad zur Datei (relativ zum Workspace)."}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Erstellt oder überschreibt eine Datei (nur innerhalb des Workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Pfad zur Datei (relativ zum Workspace)."},
                        "content": {"type": "string", "description": "Dateiinhalt (kompletter Inhalt)."},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "Listet den Inhalt eines Verzeichnisses (nur innerhalb des Workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Verzeichnis-Pfad (relativ zum Workspace).",
                            "default": ".",
                        }
                    },
                    "required": ["directory_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Löscht eine Datei (nur innerhalb des Workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Pfad zur Datei (relativ zum Workspace)."}
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_exists",
                "description": "Prüft, ob eine Datei existiert (nur innerhalb des Workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Pfad zur Datei (relativ zum Workspace)."}
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]

    if enable_web_search:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Führt eine Websuche durch für aktuelle Informationen, Fakten oder News.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Suchanfrage"}
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    return tools

