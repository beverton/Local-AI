"""
Tests: Function Calling (unit-level / deterministic)

Wichtig: Diese Tests laufen ohne echtes Modell und prüfen:
- Tool-Call Parser
- Tool Definitions Toggle
- ChatAgent Integration mit Stub-Managern (Function Calling vor Pattern-Matching)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Füge backend zum Python-Pfad hinzu (wie andere Tests)
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))


def test_parse_tool_calls_hermes_style():
    from model_manager import ModelManager

    text = '<<tool_call>>{"name":"web_search","arguments":{"query":"Qwen2.5 tools"}}<</tool_call>>'
    calls = ModelManager._parse_tool_calls_from_text(text)
    assert calls and calls[0]["name"] == "web_search"
    assert calls[0]["arguments"]["query"] == "Qwen2.5 tools"


def test_parse_tool_calls_plain_json():
    from model_manager import ModelManager

    text = '{"name":"read_file","arguments":{"file_path":"README.md"}}'
    calls = ModelManager._parse_tool_calls_from_text(text)
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["file_path"] == "README.md"


def test_tool_definitions_web_search_toggle():
    from tool_definitions import get_openai_tool_definitions

    tools_on = get_openai_tool_definitions(enable_web_search=True)
    tools_off = get_openai_tool_definitions(enable_web_search=False)

    assert any(t.get("function", {}).get("name") == "web_search" for t in tools_on)
    assert not any(t.get("function", {}).get("name") == "web_search" for t in tools_off)


def test_tool_detection_web_search_query_extraction():
    from tool_detection import detect_web_search_query

    assert detect_web_search_query("Wer ist Ada Lovelace?") in ("ada lovelace", "Ada Lovelace", "ada lovelace?")
    assert detect_web_search_query("suche nach informationen über NVIDIA RTX 5090") is not None
    assert detect_web_search_query("2+2") is None


def test_chat_agent_prefers_function_calling_then_generates_answer():
    from agents.chat_agent import ChatAgent

    class DummyAgentManager:
        def execute_tool(self, tool_name: str, **kwargs):
            if tool_name == "read_file":
                return f"FILE({kwargs.get('file_path')}): ok"
            raise ValueError("unknown tool")

    class DummyModelManager:
        def generate_with_tools(self, messages, tools, max_length: int = 256, temperature: float = 0.0):
            return {
                "content": "",
                "raw": "",
                "tool_calls": [{"name": "read_file", "arguments": {"file_path": "test.txt"}}],
            }

        def generate(self, messages, max_length: int = 256, temperature: float = 0.3, is_coding: bool = False):
            # Erwartung: Tool-Ergebnis wurde in user_message eingebettet
            assert messages and messages[-1]["role"] == "user"
            assert "TOOL-ERGEBNIS [read_file]:" in messages[-1]["content"]
            return "final answer"

    agent = ChatAgent(agent_id="a1", conversation_id="c1", config={"max_length": 128, "temperature": 0.2})
    agent.set_model_manager(DummyModelManager())
    agent.set_agent_manager(DummyAgentManager())

    # BaseAgent.process_message füllt message_history und ruft _generate_response auf
    result = agent.process_message("Bitte prüfe die Datei test.txt")
    assert "final answer" in result


def test_chat_agent_fallbacks_to_pattern_matching_when_no_tool_calls():
    from agents.chat_agent import ChatAgent

    class DummyAgentManager:
        def execute_tool(self, tool_name: str, **kwargs):
            if tool_name == "read_file":
                return "dummy file content"
            raise ValueError("unknown tool")

    class DummyModelManager:
        def generate_with_tools(self, messages, tools, max_length: int = 256, temperature: float = 0.0):
            return {"content": "no tool", "raw": "no tool", "tool_calls": []}

        def generate(self, messages, max_length: int = 256, temperature: float = 0.3, is_coding: bool = False):
            assert "TOOL-ERGEBNIS [read_file]:" in messages[-1]["content"]
            return "ok"

    agent = ChatAgent(agent_id="a2", conversation_id="c2", config={"max_length": 128, "temperature": 0.2})
    agent.set_model_manager(DummyModelManager())
    agent.set_agent_manager(DummyAgentManager())

    result = agent.process_message("Lies die Datei test.txt")
    assert result.strip() == "ok"

