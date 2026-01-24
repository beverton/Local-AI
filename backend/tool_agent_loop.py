"""
Tool Agent Loop (streaming-capable)

Implements a professional function-calling loop:
- decide tool calls via ModelManager.generate_with_tools()
- execute tools via ToolRegistry (with policies)
- repeat until no tool calls
- stream final answer via ModelManager.generate_stream()

SSE events are yielded as dicts; caller is responsible to wrap as `data: <json>\n\n`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import json
import uuid

from tool_definitions import get_openai_tool_definitions
from tool_registry import get_tool_registry, normalize_tool_call, tool_result_preview
from file_content_extractor import extract_write_file_content
from confirm_policy import parse_confirm_delete


@dataclass
class ToolAgentConfig:
    max_tool_calls: int = 3
    decide_max_tokens: int = 256
    final_max_tokens: int = 2048
    decide_temperature: float = 0.0
    final_temperature: float = 0.3


def _format_tool_results_block(tool_results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for tr in tool_results:
        name = tr.get("name")
        res = tr.get("result")
        if name == "web_search" and isinstance(res, dict) and isinstance(res.get("results"), list):
            src_lines: List[str] = []
            for i, r in enumerate(res.get("results") or [], start=1):
                if not isinstance(r, dict):
                    continue
                url = (r.get("url") or "").strip()
                title = (r.get("title") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                if not url:
                    continue
                src_lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}".strip())
            blocks.append(
                "WEB-SUCHE ERGEBNISSE (QUELLEN):\n"
                + ("\n\n".join(src_lines) if src_lines else "(keine Ergebnisse)")
                + "\n\nREGELN:\n- Verwende NUR diese URLs als Quellen.\n- Zitiere im Text als [1], [2], ...\n- Erfinde keine weiteren Links/Quellen."
            )
            continue
        if isinstance(res, (dict, list)):
            res_str = json.dumps(res, ensure_ascii=False, indent=2)
        else:
            res_str = str(res)
        blocks.append(f"TOOL-ERGEBNIS [{name}]:\n{res_str}")
    return "\n\n".join(blocks)


class ToolAgentLoop:
    def __init__(self, model_manager, *, enable_web_search: bool):
        self.model_manager = model_manager
        self.enable_web_search = enable_web_search
        self.tools_schema = get_openai_tool_definitions(enable_web_search=enable_web_search)
        self.registry = get_tool_registry(enable_web_search=enable_web_search)

    def run_sse(
        self,
        *,
        messages: List[Dict[str, str]],
        user_message: str,
        conversation_id: Optional[str],
        model_id: Optional[str],
        language: str = "de",
        cfg: Optional[ToolAgentConfig] = None,
        emit_meta: bool = True,
    ) -> Generator[Dict[str, Any], None, str]:
        """
        Returns final full response string as generator return value.
        """
        cfg = cfg or ToolAgentConfig()
        trace_id = str(uuid.uuid4())
        if emit_meta:
            yield {"meta": {"trace_id": trace_id, "conversation_id": conversation_id, "model_id": model_id, "language": language}}

        # Confirmation fast-path: execute confirmed delete deterministically
        confirmed_delete = parse_confirm_delete(user_message or "")
        if confirmed_delete:
            call_id = str(uuid.uuid4())
            yield {"tool_call": {"call_id": call_id, "name": "delete_file", "arguments": {"file_path": confirmed_delete}}}
            spec = self.registry.get("delete_file")
            if not spec:
                msg = "delete_file ist nicht verfügbar."
                yield {"tool_result": {"call_id": call_id, "ok": False, "name": "delete_file", "error": msg}}
                yield {"chunk": msg}
                return msg
            try:
                result = spec.execute(file_path=confirmed_delete)
                yield {"tool_result": {"call_id": call_id, "ok": True, "name": "delete_file", "result_preview": tool_result_preview(result)}}
                msg = f"Datei gelöscht: `{confirmed_delete}`"
                yield {"chunk": msg}
                yield {"done": True}
                return msg
            except Exception as e:
                msg = f"Fehler beim Löschen: {e}"
                yield {"tool_result": {"call_id": call_id, "ok": False, "name": "delete_file", "error": msg}}
                yield {"chunk": msg}
                return msg

        tool_calls_used = 0
        tool_results: List[Dict[str, Any]] = []

        # loop: decide -> tool_exec -> decide ...
        while tool_calls_used < cfg.max_tool_calls:
            fc = self.model_manager.generate_with_tools(
                messages,
                tools=self.tools_schema,
                max_length=cfg.decide_max_tokens,
                temperature=cfg.decide_temperature,
            )
            calls = fc.get("tool_calls") or []
            if not calls:
                break

            for call in calls:
                if tool_calls_used >= cfg.max_tool_calls:
                    break

                raw_name = (call or {}).get("name")
                raw_args = (call or {}).get("arguments", {})
                name, args = normalize_tool_call(raw_name, raw_args)
                # deterministic behavior if model calls disabled web_search
                if name == "web_search" and not self.enable_web_search:
                    msg = "Web-Suche ist deaktiviert (auto_web_search=false). Bitte aktiviere sie in den Quality-Settings."
                    yield {"tool_result": {"call_id": str(uuid.uuid4()), "ok": False, "name": "web_search", "error": msg}}
                    yield {"chunk": msg}
                    return msg
                if not name or name not in self.registry:
                    yield {"tool_result": {"call_id": str(tool_calls_used), "ok": False, "error": f"Unknown tool: {name}"}}
                    tool_calls_used += 1
                    continue

                spec = self.registry[name]
                call_id = str(uuid.uuid4())

                # delete confirmation policy: never execute directly
                if spec.requires_confirmation and name == "delete_file":
                    fp = (args or {}).get("file_path") or ""
                    yield {"tool_call": {"call_id": call_id, "name": name, "arguments": args, "requires_confirmation": True}}
                    msg = (
                        f"Zum Löschen brauche ich eine Bestätigung.\n\n"
                        f"Bitte antworte mit: **CONFIRM DELETE {fp}**"
                    )
                    yield {"chunk": msg}
                    return msg

                # write_file safety: extract content if missing/empty
                if name == "write_file":
                    content = args.get("content") if isinstance(args, dict) else None
                    if not isinstance(content, str) or not content.strip():
                        extracted = extract_write_file_content(user_message)
                        if extracted:
                            args["content"] = extracted
                        else:
                            msg = "Für `write_file` fehlt der Inhalt. Bitte sende den gewünschten Inhalt (am besten in einem ```code``` Block)."
                            yield {"tool_result": {"call_id": call_id, "ok": False, "name": name, "error": msg}}
                            yield {"chunk": msg}
                            return msg

                yield {"tool_call": {"call_id": call_id, "name": name, "arguments": args}}

                ok = True
                result: Any = None
                err: Optional[str] = None
                try:
                    result = spec.execute(**args)
                except Exception as e:
                    ok = False
                    err = str(e)

                if ok:
                    tool_results.append({"name": name, "result": result})
                    yield {"tool_result": {"call_id": call_id, "ok": True, "name": name, "result_preview": tool_result_preview(result)}}
                else:
                    yield {"tool_result": {"call_id": call_id, "ok": False, "name": name, "error": err}}

                tool_calls_used += 1

            # Inject tool results back into messages (ChatAgent-style, robust across templates)
            if tool_results:
                tool_block = _format_tool_results_block(tool_results)
                messages = list(messages) + [
                    {
                        "role": "user",
                        "content": f"FRAGE: {user_message}\n\n{tool_block}\n\nWICHTIG:\n- Nutze NUR die Tool-Ergebnisse oben.\n- Wenn Web-Suche enthalten ist: zitiere Quellen als [1], [2], ... und erfinde keine Links.\nAntworte auf {language.upper()}:",
                    }
                ]
                tool_results = []

        # final streaming generation (pause happens naturally during tool execution; only chunk events here)
        full = ""
        for chunk in self.model_manager.generate_stream(
            messages,
            max_length=cfg.final_max_tokens,
            temperature=cfg.final_temperature,
        ):
            if chunk:
                full += chunk
                yield {"chunk": chunk}

        yield {"done": True}
        return full

