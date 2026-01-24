"""
Chat Agent - Agent für normale Gespräche mit Tool-Unterstützung
Kann automatisch Tools nutzen (WebSearch, Dateimanipulation) basierend auf Nachrichten
"""
from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent
import logging
import re
import json
import time
from tool_detection import detect_web_search_query
from file_content_extractor import extract_write_file_content
from confirm_policy import parse_confirm_delete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Agent für normale Gespräche mit automatischer Tool-Nutzung"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "chat_agent"
        self.name = "Chat Agent"
        self.system_prompt = """Du bist ein hilfreicher AI-Assistent ähnlich Perplexity AI. Antworte NUR auf Deutsch.

KRITISCH - Tool-Ergebnisse/Quellen nutzen:
Wenn dir Tool-Ergebnisse (Web-Suche, Dateien) gezeigt werden:
1. Nutze AUSSCHLIESSLICH diese Informationen (dein internes Wissen ist veraltet)
2. Kopiere URLs EXAKT wie gezeigt (keine Leerzeichen/Änderungen)
3. Tool-Ergebnisse haben IMMER Vorrang vor deinem internen Wissen
4. Wenn Quellen/URLs vorhanden sind: referenziere sie als [1], [2], ... und erfinde keine zusätzlichen Quellen

Für Antworten:
- Antworte präzise und direkt (keine Wiederholung von System/User-Text)
- Für Code: verwende Markdown Code-Blöcke mit Sprach-Tag, vollständig und ausführbar
"""
        
        # Verfügbare Tools
        self.available_tools = [
            "web_search",
            "read_file",
            "write_file",
            "list_directory",
            "delete_file",
            "file_exists"
        ]
    
    def _detect_tool_need(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Erkennt ob ein Tool benötigt wird basierend auf Pattern-Matching
        
        Returns:
            Dict mit "tool_name" und "params" oder None
        """
        message_lower = message.lower()
        message_original = message

        def _sanitize_user_path_fragment(p: str) -> str:
            """
            Normalisiert einen vom User/Regex extrahierten Pfad:
            - entfernt Quotes/Backticks
            - entfernt trailing Satzzeichen (z.B. '?', '.', ')', ']')
            """
            if not p:
                return ""
            p = p.strip().strip('"\''"`")
            # Entferne umschließende Klammern
            if (p.startswith("(") and p.endswith(")")) or (p.startswith("[") and p.endswith("]")) or (p.startswith("{") and p.endswith("}")):
                p = p[1:-1].strip()
            # Entferne trailing punctuation / whitespace
            p = re.sub(r'[\s\)\]\}\.,;:!?]+$', '', p).strip()
            # Nochmals Quotes/Backticks
            p = p.strip().strip('"\''"`")
            return p
        
        # WebSearch Erkennung (zentralisiert, kein doppelter Pattern-Code)
        query = detect_web_search_query(message_original)
        if query:
            return {"tool_name": "web_search", "params": {"query": query}}
        
        # Datei-Operation Patterns
        # read_file - Erweiterte Patterns für natürlichere Formulierungen
        read_patterns = [
            r"lies\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"lese\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"zeige\s+(?:mir\s+)?(?:den\s+)?(?:inhalt\s+)?(?:der\s+)?(?:datei\s+)?(.+)",
            r"öffne\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"was\s+steht\s+in\s+(?:der\s+)?(?:datei\s+)?(.+)",
            # Erweiterte natürliche Formulierungen
            r"(?:nachschauen|nachschaue|nachschau)\s+(?:in|bei|im|in der|in das)\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"(?:prüfe|prüf|prüfen)\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"(?:analysiere|analysier|analysieren)\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"(?:untersuche|untersuchen)\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"(?:schaue|schau|schauen)\s+(?:in|in die|in das|in der)\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"(?:zeige|zeig|zeigen)\s+(?:mir\s+)?(?:den\s+)?(?:code|inhalt|text)\s+(?:von|in|der|die|das)\s+(?:datei\s+)?(.+)",
            r"(?:kannst\s+du|kann\s+du|können\s+sie|können\s+du)\s+(?:in|bei|im|in der|in das)\s+(?:die\s+)?(?:datei\s+)?(.+?)(?:\s+nachschauen|\s+prüfen|\s+analysieren|\s+untersuchen)?",
            r"(?:kannst\s+du|kann\s+du)\s+(?:nachschauen|prüfen|analysieren|untersuchen)\s+(?:in|bei|im|in der|in das)\s+(?:die\s+)?(?:datei\s+)?(.+)",
        ]
        for pattern in read_patterns:
            match = re.search(pattern, message_lower)
            if match:
                file_path = _sanitize_user_path_fragment(match.group(1))
                if file_path:
                    return {"tool_name": "read_file", "params": {"file_path": file_path}}
        
        # write_file
        write_patterns = [
            r"schreibe\s+(?:in\s+)?(?:die\s+)?(?:datei\s+)?(.+?)\s+(?:den\s+)?(?:inhalt\s+)?(.+)",
            r"speichere\s+(?:in\s+)?(?:die\s+)?(?:datei\s+)?(.+?)\s+(?:den\s+)?(?:inhalt\s+)?(.+)",
            # Multiline / "erstelle ... mit folgendem Inhalt:"
            r"(?s)(?:erstelle|erzeuge)\s+(?:eine\s+)?datei\s+(.+?)\s+(?:mit\s+(?:folgendem\s+)?inhalt|inhalt)\s*[:\n]\s*(.+)$",
        ]
        for pattern in write_patterns:
            match = re.search(pattern, message_original, flags=re.IGNORECASE)
            if match:
                file_path = _sanitize_user_path_fragment(match.group(1))
                content = match.group(2).strip()
                if file_path and content:
                    return {"tool_name": "write_file", "params": {"file_path": file_path, "content": content}}
        
        # list_directory
        list_patterns = [
            r"liste\s+(?:das\s+)?(?:verzeichnis\s+)?(.+)",
            r"zeige\s+(?:mir\s+)?(?:den\s+)?(?:inhalt\s+)?(?:des\s+)?(?:verzeichnisses\s+)?(.+)",
            r"was\s+ist\s+in\s+(?:dem\s+)?(?:verzeichnis\s+)?(.+)",
        ]
        for pattern in list_patterns:
            match = re.search(pattern, message_lower)
            if match:
                dir_path = _sanitize_user_path_fragment(match.group(1))
                if not dir_path or dir_path == ".":
                    dir_path = "."
                return {"tool_name": "list_directory", "params": {"directory_path": dir_path}}
        
        # delete_file
        delete_patterns = [
            r"lösche\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"entferne\s+(?:die\s+)?(?:datei\s+)?(.+)",
        ]
        for pattern in delete_patterns:
            match = re.search(pattern, message_lower)
            if match:
                file_path = _sanitize_user_path_fragment(match.group(1))
                if file_path:
                    return {"tool_name": "delete_file", "params": {"file_path": file_path}}
        
        # file_exists
        exists_patterns = [
            r"existiert\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"gibt\s+es\s+(?:die\s+)?(?:datei\s+)?(.+)",
        ]
        for pattern in exists_patterns:
            match = re.search(pattern, message_lower)
            if match:
                file_path = _sanitize_user_path_fragment(match.group(1))
                if file_path:
                    return {"tool_name": "file_exists", "params": {"file_path": file_path}}
        
        return None
    
    def _is_tool_enabled(self, tool_name: str) -> bool:
        """
        Prüft ob ein Tool durch UI-Toggle aktiviert ist
        
        Args:
            tool_name: Name des Tools (z.B. "web_search", "read_file")
            
        Returns:
            True wenn Tool aktiviert ist, False sonst
        """
        try:
            # Lade Quality-Settings
            import os
            import json
            quality_settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "quality_settings.json")
            if os.path.exists(quality_settings_path):
                with open(quality_settings_path, 'r', encoding='utf-8') as f:
                    quality_settings = json.load(f)
                    
                    # Web-Search hat eigenen Toggle
                    if tool_name == "web_search":
                        return quality_settings.get("auto_web_search", False)
                    
                    # Datei-Tools: Prüfe ob allgemeiner Toggle existiert (aktuell: immer erlaubt)
                    # TODO: Wenn später ein allgemeiner Toggle für Datei-Tools hinzugefügt wird, hier prüfen
                    if tool_name in ["read_file", "write_file", "list_directory", "delete_file", "file_exists"]:
                        # Aktuell: Datei-Tools sind immer erlaubt (kein Toggle vorhanden)
                        return True
                    
                    # Unbekanntes Tool: nicht erlauben
                    return False
        except Exception as e:
            logger.debug(f"Fehler beim Laden der Quality-Settings für Tool-Check: {e}")
            # Bei Fehler: konservativ - erlaube nur bekannte Tools
            return tool_name in ["read_file", "write_file", "list_directory", "delete_file", "file_exists"]
        
        # Fallback: keine Settings gefunden - konservativ
        return False
    
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """Generiert eine Antwort, nutzt automatisch Tools wenn nötig"""
        if not self.model_manager:
            raise RuntimeError("ModelManager nicht gesetzt")

        # Confirmation fast-path (destructive ops)
        confirmed_delete = parse_confirm_delete(message or "")
        if confirmed_delete:
            try:
                self.execute_tool("delete_file", file_path=confirmed_delete)
                return f"Datei gelöscht: `{confirmed_delete}`"
            except Exception as e:
                return f"Fehler beim Löschen: {e}"
        
        # HINWEIS: Modell-Laden wird von main.py/Model-Service gehandhabt
        # Keine redundante Prüfung hier, da bei Model-Service das Modell
        # nicht im lokalen model_manager geladen ist
        
        # Konfiguration für Generierung
        effective_max_length = self.config.get("max_length", 1024)
        effective_temperature = self.config.get("temperature", 0.7)

        # 1) Function Calling (wenn möglich) -> danach Tool-Ausführung -> dann Antwort generieren
        #    Falls Function Calling nicht klappt/keine Tool-Calls: Fallback auf Pattern-Matching.
        used_function_calling = False
        tool_results: List[Dict[str, Any]] = []  # [{"name":..., "result":...}, ...]

        use_model_service = bool(self.model_service_client and self.model_service_client.is_available())
        if not use_model_service and hasattr(self.model_manager, "generate_with_tools"):
            try:
                from tool_definitions import get_openai_tool_definitions

                enable_web_search = self._is_tool_enabled("web_search")
                tools = get_openai_tool_definitions(enable_web_search=enable_web_search)

                # Messages für Tool-Entscheidung (ohne Tool-Ergebnisse)
                fc_messages: List[Dict[str, str]] = []
                if self.system_prompt:
                    fc_messages.append({"role": "system", "content": self.system_prompt})

                history_to_use = self.message_history[:-1] if len(self.message_history) > 1 else []
                recent_history = history_to_use[-10:] if len(history_to_use) > 10 else history_to_use
                for msg in recent_history:
                    if msg["role"] == "user" or msg["role"].startswith("agent_"):
                        fc_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        fc_messages.append({"role": "assistant", "content": msg["content"]})
                fc_messages.append({"role": "user", "content": message})

                # Tool-Entscheidung: deterministisch, kurze Ausgabe
                fc = self.model_manager.generate_with_tools(
                    fc_messages,
                    tools=tools,
                    max_length=min(256, int(effective_max_length)),
                    temperature=0.0,
                )
                calls = fc.get("tool_calls") or []
                if calls:
                    for call in calls:
                        tool_name = (call or {}).get("name")
                        tool_args = (call or {}).get("arguments", {})
                        if not tool_name:
                            continue
                        if not self._is_tool_enabled(tool_name):
                            logger.debug(f"Tool '{tool_name}' ist durch UI-Toggle deaktiviert - überspringe Function-Calling Tool")
                            continue
                        try:
                            logger.info(f"ChatAgent (Function Calling) nutzt Tool: {tool_name} mit Parametern: {tool_args}")
                            res = self.execute_tool(tool_name, **(tool_args if isinstance(tool_args, dict) else {}))
                        except Exception as e:
                            logger.error(f"Fehler bei Tool-Ausführung {tool_name}: {e}")
                            res = f"Fehler bei Tool-Ausführung: {str(e)}"
                        tool_results.append({"name": tool_name, "result": res})

                    used_function_calling = len(tool_results) > 0
            except Exception as e:
                logger.debug(f"Function Calling fehlgeschlagen, fallback auf Pattern-Matching: {e}")

        # 2) Fallback: Pattern-Matching (nur wenn Function Calling nicht genutzt wurde)
        if not used_function_calling:
            tool_info = self._detect_tool_need(message)
            if tool_info:
                tool_name = tool_info["tool_name"]
                tool_params = tool_info["params"]

                if not self._is_tool_enabled(tool_name):
                    logger.debug(f"Tool '{tool_name}' ist durch UI-Toggle deaktiviert - überspringe Tool-Nutzung")
                    # Web-Suche explizit angefragt, aber deaktiviert -> deterministisch antworten (kein Halluzinieren von URLs)
                    if tool_name == "web_search":
                        return "Web-Suche ist deaktiviert (auto_web_search=false). Bitte aktiviere sie in den Quality-Settings, oder stelle die Frage ohne Web-Recherche."
                else:
                    # Confirm-Policy: delete_file immer bestätigen
                    if tool_name == "delete_file":
                        msg_l = (message or "").strip()
                        fp = (tool_params or {}).get("file_path", "")
                        # akzeptiere einfache Bestätigung: "CONFIRM DELETE <path>"
                        if not (msg_l.upper().startswith("CONFIRM DELETE") and fp and fp in msg_l):
                            return f"Zum Löschen brauche ich eine Bestätigung. Bitte antworte mit: **CONFIRM DELETE {fp}**"
                    # write_file safety: content extraction (centralized)
                    if tool_name == "write_file":
                        c = (tool_params or {}).get("content")
                        if not isinstance(c, str) or not c.strip():
                            extracted = extract_write_file_content(message or "")
                            if extracted:
                                tool_params["content"] = extracted
                            else:
                                return "Für `write_file` fehlt der Inhalt. Bitte sende den Inhalt (am besten in einem ```code``` Block)."
                    try:
                        logger.info(f"ChatAgent nutzt Tool: {tool_name} mit Parametern: {tool_params}")
                        res = self.execute_tool(tool_name, **tool_params)
                    except Exception as e:
                        logger.error(f"Fehler bei Tool-Ausführung {tool_name}: {e}")
                        res = f"Fehler bei Tool-Ausführung: {str(e)}"
                    tool_results.append({"name": tool_name, "result": res})

                    # Wenn es eine direkte File-Tool-Anfrage ist, antworte deterministisch aus Tool-Result
                    try:
                        msg_l = (message or "").lower()
                        # "und gib mir X" ist bei Tool-Befehlen normal, gilt nicht als Analyse
                        wants_analysis = any(k in msg_l for k in [" danach", " erkl", " analys", " zusammenfass", " bewert", " vergleic", " finde fehler", " warum"])

                        allow_direct = (not wants_analysis) and tool_name in ["read_file", "write_file", "list_directory", "delete_file", "file_exists"]
                        # Web-Suche: direct response fast immer besser (korrekte URLs), außer wenn explizit Erklärung/Analyse verlangt wird
                        if tool_name == "web_search" and not wants_analysis:
                            allow_direct = True

                        if allow_direct:
                            if tool_name == "read_file":
                                return f"Inhalt von `{tool_params.get('file_path','')}`:\n\n{res}"
                            if tool_name == "write_file":
                                return f"Datei geschrieben: `{tool_params.get('file_path','')}`"
                            if tool_name == "delete_file":
                                return f"Datei gelöscht: `{tool_params.get('file_path','')}`"
                            if tool_name == "file_exists":
                                exists = bool(res)
                                return f"Datei `{tool_params.get('file_path','')}` existiert: {exists}"
                            if tool_name == "list_directory":
                                return f"Inhalt von `{tool_params.get('directory_path','')}`:\n\n{json.dumps(res, ensure_ascii=False, indent=2) if isinstance(res,(dict,list)) else str(res)}"
                            if tool_name == "web_search":
                                if isinstance(res, dict) and isinstance(res.get("results"), list):
                                    lines = []
                                    for i, item in enumerate(res.get("results", [])[:5], 1):
                                        title = (item or {}).get("title", f"Quelle {i}")
                                        url = (item or {}).get("url", "")
                                        snippet = (item or {}).get("snippet", "")
                                        lines.append(f"[{i}] {title}\n{snippet}\nURL: {url}".strip())
                                    return "Web-Suche Ergebnisse:\n\n" + "\n\n".join(lines)
                                return f"Web-Suche Ergebnis:\n\n{json.dumps(res, ensure_ascii=False, indent=2) if isinstance(res,(dict,list)) else str(res)}"
                    except Exception:
                        pass
        
        # Erstelle Messages für Modell
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Füge Kontext aus History hinzu (letzte 10 Nachrichten, OHNE die aktuelle)
        # Die aktuelle Nachricht wurde bereits von process_message zur History hinzugefügt
        # Also nehmen wir alle außer der letzten
        history_to_use = self.message_history[:-1] if len(self.message_history) > 1 else []
        recent_history = history_to_use[-10:] if len(history_to_use) > 10 else history_to_use
        
        for msg in recent_history:
            if msg["role"] == "user" or msg["role"].startswith("agent_"):
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Aktuelle Nachricht mit Tool-Ergebnissen (0..n Tools)
        user_message = message
        tool_used = None
        web_search_result_for_url_fix = None
        if tool_results:
            blocks: List[str] = []
            for tr in tool_results:
                tname = tr.get("name")
                tres = tr.get("result")
                if not tool_used:
                    tool_used = tname
                # Formatiere Tool-Ergebnis
                if isinstance(tres, dict) and tname == "web_search" and "results" in tres:
                    web_search_result_for_url_fix = tres
                    tool_result_str = "Web-Suche Ergebnisse:\n"
                    for i, result in enumerate(tres.get("results", [])[:5], 1):
                        title = result.get("title", "Kein Titel")
                        snippet = result.get("snippet", "Keine Beschreibung")
                        url = result.get("url", "")
                        tool_result_str += f"\n{i}. {title}\n   {snippet}\n"
                        if url:
                            tool_result_str += f"   URL: {url}\n"
                    if tres.get("summary"):
                        tool_result_str += f"\n{tres['summary']}\n"
                elif isinstance(tres, (dict, list)):
                    tool_result_str = json.dumps(tres, ensure_ascii=False, indent=2)
                else:
                    tool_result_str = str(tres)

                blocks.append(f"TOOL-ERGEBNIS [{tname}]:\n{tool_result_str}")

            user_message = f"""FRAGE: {message}

{'\n\n'.join(blocks)}

WICHTIG: Nutze NUR die Tool-Ergebnisse oben, nicht dein internes Wissen. Kopiere URLs exakt wie gezeigt.

Antworte auf Deutsch:"""
        
        messages.append({"role": "user", "content": user_message})# Stelle sicher, dass Messages korrekt alternieren (user/assistant/user/assistant)
        # Entferne doppelte aufeinanderfolgende user oder assistant Messages
        cleaned_messages = []
        last_role = None
        for msg in messages:
            current_role = msg.get("role")
            # System-Prompt immer hinzufügen
            if current_role == "system":
                cleaned_messages.append(msg)
                continue
            # Überspringe wenn gleiche Rolle wie vorher (außer beim ersten)
            if current_role == last_role and last_role is not None:
                continue
            cleaned_messages.append(msg)
            last_role = current_role
        
        messages = cleaned_messages
        
        # Generiere Antwort
        try:# 🔥 MODEL SERVICE SUPPORT - Verwende Model Service Client wenn verfügbar
            if self.model_service_client and self.model_service_client.is_available():
                # Verwende Model Service
                result = self.model_service_client.chat(
                    message=message,
                    messages=messages,
                    conversation_id=self.conversation_id,
                    max_length=effective_max_length,
                    temperature=effective_temperature
                )
                response = result.get("response", "") if result else ""
            else:
                # Fallback auf lokalen model_manager
                response = self.model_manager.generate(
                    messages,
                    max_length=effective_max_length,  # Längere Antworten für Tool-Ergebnisse
                    temperature=effective_temperature
                )# Bereinige Response (entferne System-Prompt-Phrasen)
            cleaned_response = response.strip()
            
            # Entferne führende "assistant" falls vorhanden
            if cleaned_response.lower().startswith('assistant'):
                cleaned_response = cleaned_response.split(':', 1)[-1].strip() if ':' in cleaned_response else cleaned_response[9:].strip()
            
            # Entferne Tool-Ergebnis-Referenzen falls noch vorhanden
            if tool_used and f"[Tool-Ergebnis von {tool_used}]" in cleaned_response:
                # Entferne nur die Referenz, behalte den Inhalt
                lines = cleaned_response.split('\n')
                cleaned_lines = []
                skip_next = False
                for line in lines:
                    if f"[Tool-Ergebnis von {tool_used}]" in line:
                        skip_next = True
                        continue
                    if skip_next and line.strip() == "":
                        skip_next = False
                        continue
                    if not skip_next:
                        cleaned_lines.append(line)
                cleaned_response = '\n'.join(cleaned_lines).strip()
            
            # POST-PROCESSING: Korrigiere URLs aus Tool-Ergebnissen
            # Verhindert Tippfehler beim Abschreiben von URLs durch das Modell
            if tool_used == "web_search" and web_search_result_for_url_fix and isinstance(web_search_result_for_url_fix, dict):
                cleaned_response = self._fix_urls_in_response(cleaned_response, web_search_result_for_url_fix)

            # Post-Processing: Normalisiere gemischte Alphabete (z.B. kyrillische Homoglyphen)
            try:
                if hasattr(self.model_manager, "_normalize_mixed_script_homoglyphs"):
                    cleaned_response = self.model_manager._normalize_mixed_script_homoglyphs(cleaned_response)
            except Exception:
                pass
            
            # POST-PROCESSING: Entferne nicht-deutsche Sprachen (z.B. Chinesisch)
            # 🔧 DEAKTIVIERT: Diese Funktion ist zu aggressiv und entfernt wichtige Zeichen aus URLs
            # cleaned_response = self._remove_non_german_text(cleaned_response)
            
            # Entferne ggf. versehentlich fortgesetzte Chat-Marker
            marker_match = re.search(r'\b(?:User|Assistant):', cleaned_response, flags=re.IGNORECASE)
            if marker_match:
                cleaned_response = cleaned_response[:marker_match.start()].strip()
            
            logger.info(f"ChatAgent Antwort generiert: {len(cleaned_response)} Zeichen, Tool verwendet: {tool_used}")
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Fehler bei Antwort-Generierung: {e}")
            raise
    
    def _fix_urls_in_response(self, response: str, web_search_result: Dict[str, Any]) -> str:
        """
        Post-Processing: Ersetzt fehlerhafte URLs in der Antwort durch korrekte URLs aus den Tool-Ergebnissen.
        Verhindert Tippfehler des Modells beim Abschreiben von URLs.
        
        Args:
            response: Die Antwort des Modells
            web_search_result: Die Web-Search Tool-Ergebnisse mit korrekten URLs
            
        Returns:
            Response mit korrigierten URLs
        """
        if not web_search_result.get("results"):
            return response
        
        # Extrahiere alle korrekten URLs aus den Web-Search-Ergebnissen
        correct_urls = []
        for result in web_search_result.get("results", []):
            url = result.get("url", "").strip()
            if url and url.startswith("http"):
                correct_urls.append(url)
        
        if not correct_urls:
            return response
        
        # Suche nach URLs oder URL-ähnlichen Mustern in der Response
        # und ersetze sie durch die korrekten URLs
        import re
        
        # Pattern für URLs (auch mit Tippfehlern/Leerzeichen/falschen Zeichen)
        # Erweitert um häufige Fehler: ; statt ., Leerzeichen, etc.
        url_pattern = r'<?\s*https?[:/;]+[^\s<>]+\s*>?'
        
        # Finde alle URL-ähnlichen Strings in der Response
        found_urls = re.findall(url_pattern, response)
        
        if not found_urls:
            # Keine URLs in Response gefunden - füge die korrekte URL am Ende hinzu
            if correct_urls:
                response += f"\n\nLink: {correct_urls[0]}"
                logger.info(f"URL hinzugefügt zur Response: {correct_urls[0]}")
            return response
        
        # Ersetze gefundene URLs durch die korrekten URLs
        for found_url in found_urls:
            found_url_clean = found_url.strip('<> ')
            
            # Finde die beste Übereinstimmung aus den korrekten URLs
            best_match = None
            best_score = 0
            
            for correct_url in correct_urls:
                # Berechne Ähnlichkeit (einfache Heuristik)
                # Normalisiere beide URLs für Vergleich (entferne Sonderzeichen, Leerzeichen)
                score = 0
                
                try:
                    correct_domain = correct_url.split('/')[2] if correct_url.count('/') >= 2 else correct_url.replace('https://', '').replace('http://', '')
                    found_domain = found_url_clean.split('/')[2] if found_url_clean.count('/') >= 2 else found_url_clean.replace('https://', '').replace('http://', '').replace('https;//', '').replace('http;//', '')
                except:
                    correct_domain = correct_url
                    found_domain = found_url_clean
                
                # Normalisiere für Vergleich: entferne alle nicht-alphanumerischen Zeichen
                correct_normalized = ''.join(c.lower() for c in correct_domain if c.isalnum())
                found_normalized = ''.join(c.lower() for c in found_domain if c.isalnum())
                
                # Exakte Übereinstimmung nach Normalisierung
                if correct_normalized == found_normalized:
                    score = 100
                # Teilübereinstimmung (mind. 70%)
                elif correct_normalized in found_normalized or found_normalized in correct_normalized:
                    score = 80
                # Domain-Hauptteil stimmt überein
                elif any(part in found_normalized for part in correct_domain.split('.') if len(part) > 4):
                    score = 60
                
                if score > best_score:
                    best_score = score
                    best_match = correct_url
            
            # Ersetze wenn gute Übereinstimmung gefunden
            if best_match and best_score >= 50:
                response = response.replace(found_url, best_match)
                logger.info(f"URL korrigiert: {found_url} -> {best_match}")
        
        return response
    
    def _remove_non_german_text(self, text: str) -> str:
        """
        Entfernt nicht-deutsche Zeichen (z.B. Chinesisch, Arabisch) aus der Antwort.
        Behält: Deutsch, Englisch (für URLs/Code), Zahlen, Satzzeichen.
        
        Args:
            text: Die zu bereinigende Antwort
            
        Returns:
            Bereinigte Antwort ohne nicht-deutsche Zeichen
        """
        import re
        
        # Pattern für nicht-westliche Schriftzeichen (CJK, Arabisch, etc.)
        # Erlaubt: Lateinische Buchstaben, Zahlen, deutsche Umlaute, gängige Satzzeichen
        # Entfernt: Chinesisch, Japanisch, Koreanisch, Arabisch, etc.
        
        # Zeichen-Bereiche für CJK (Chinesisch-Japanisch-Koreanisch)
        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]+'
        
        # Entferne CJK-Zeichen
        text = re.sub(cjk_pattern, '', text)
        
        # Entferne auch isolierte nicht-lateinische Zeichen (außer deutsche Umlaute)
        # Erlaube: a-z, A-Z, 0-9, äöüßÄÖÜ, Leerzeichen, gängige Satzzeichen
        # Entferne: alles andere
        text = re.sub(r'[^\x00-\x7F\u00C0-\u017F\s\[\]\(\)\{\}\<\>\-_.,;:!?\'"\n\r\t/\\@#$%^&*+=]', '', text)
        
        # Bereinige mehrfache Leerzeichen
        text = re.sub(r'\s+', ' ', text)
        
        # Bereinige mehrfache Zeilenumbrüche
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        return text.strip()


