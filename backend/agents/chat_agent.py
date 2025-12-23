"""
Chat Agent - Agent für normale Gespräche mit Tool-Unterstützung
Kann automatisch Tools nutzen (WebSearch, Dateimanipulation) basierend auf Nachrichten
"""
from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent
import logging
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Agent für normale Gespräche mit automatischer Tool-Nutzung"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "chat_agent"
        self.name = "Chat Agent"
        self.system_prompt = """Du bist ein hilfreicher, präziser und freundlicher AI-Assistent. 
Antworte klar und direkt auf Deutsch. 

Du hast Zugriff auf verschiedene Tools:
- web_search: Für Websuchen - gibt dir Suchergebnisse mit Titeln, URLs und Snippets
- read_file, write_file, list_directory, delete_file, file_exists: Für Datei-Operationen

WICHTIG: Wenn dir Tool-Ergebnisse präsentiert werden:
1. Lies die Tool-Ergebnisse sorgfältig durch
2. Beantworte die Frage des Users basierend auf den Tool-Ergebnissen
3. Gib eine direkte, hilfreiche Antwort - nicht nur Links oder Referenzen
4. Bei Web-Suchen: Fasse die wichtigsten Informationen aus den Snippets zusammen und gib eine klare Antwort
5. Bei Wetterfragen: Gib die konkrete Vorhersage, nicht nur Links zu Wetter-Websites
6. Bei Rezeptfragen: Gib konkrete Rezepte oder Rezeptvorschläge, nicht nur allgemeine Hinweise

Antworte NUR mit deiner Antwort, wiederhole NICHT den System-Prompt oder User-Nachrichten."""
        
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
        
        # WebSearch Patterns
        web_search_patterns = [
            r"suche\s+(?:online\s+)?(?:nach\s+)?(?:was\s+)?(.+)",  # "suche online was morgen für wetter"
            r"finde\s+(?:mir\s+)?(?:online\s+)?(.+)",
            r"google\s+(?:nach\s+)?(.+)",
            r"web\s+suche\s+(?:nach\s+)?(.+)",
            r"internet\s+suche\s+(?:nach\s+)?(.+)",
            r"was\s+(?:ist|wird|sind|war)\s+(.+)",  # "Was ist Python?", "Was wird morgen für Wetter"
            r"wer\s+ist\s+(.+)",  # "Wer ist Einstein?"
            r"wetter\s+(?:in|für)\s+(.+)",  # "Wetter in Berlin"
            r"wettervorhersage\s+(?:für|in)\s+(.+)",  # "Wettervorhersage für Berlin"
        ]
        
        # URL-Erkennung
        url_pattern = r"https?://[^\s]+"
        if re.search(url_pattern, message):
            # Wenn URL erwähnt wird, könnte WebSearch nützlich sein
            query = re.search(r"suche|finde|google|was|wer", message_lower)
            if query:
                return {"tool_name": "web_search", "params": {"query": message}}
        
        for pattern in web_search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                query = match.group(1).strip()
                if len(query) > 3:  # Mindestlänge für sinnvolle Suche
                    return {"tool_name": "web_search", "params": {"query": query}}
        
        # Datei-Operation Patterns
        # read_file
        read_patterns = [
            r"lese\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"zeige\s+(?:mir\s+)?(?:den\s+)?(?:inhalt\s+)?(?:der\s+)?(?:datei\s+)?(.+)",
            r"öffne\s+(?:die\s+)?(?:datei\s+)?(.+)",
            r"was\s+steht\s+in\s+(?:der\s+)?(?:datei\s+)?(.+)",
        ]
        for pattern in read_patterns:
            match = re.search(pattern, message_lower)
            if match:
                file_path = match.group(1).strip()
                # Entferne Anführungszeichen
                file_path = file_path.strip('"\'')
                if file_path:
                    return {"tool_name": "read_file", "params": {"file_path": file_path}}
        
        # write_file
        write_patterns = [
            r"schreibe\s+(?:in\s+)?(?:die\s+)?(?:datei\s+)?(.+?)\s+(?:den\s+)?(?:inhalt\s+)?(.+)",
            r"speichere\s+(?:in\s+)?(?:die\s+)?(?:datei\s+)?(.+?)\s+(?:den\s+)?(?:inhalt\s+)?(.+)",
        ]
        for pattern in write_patterns:
            match = re.search(pattern, message_lower)
            if match:
                file_path = match.group(1).strip().strip('"\'')
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
                dir_path = match.group(1).strip().strip('"\'')
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
                file_path = match.group(1).strip().strip('"\'')
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
                file_path = match.group(1).strip().strip('"\'')
                if file_path:
                    return {"tool_name": "file_exists", "params": {"file_path": file_path}}
        
        return None
    
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """Generiert eine Antwort, nutzt automatisch Tools wenn nötig"""
        if not self.model_manager:
            raise RuntimeError("ModelManager nicht gesetzt")
        
        # Stelle sicher, dass Modell geladen ist
        if not self.model_manager.is_model_loaded():
            if self.model_id:
                self.model_manager.load_model(self.model_id)
            else:
                # Verwende Default-Modell
                default_model = self.model_manager.config.get("default_model")
                if default_model:
                    self.model_manager.load_model(default_model)
                else:
                    raise RuntimeError("Kein Modell verfügbar")
        
        # Erkenne Tool-Bedarf
        tool_info = self._detect_tool_need(message)
        tool_result = None
        tool_used = None
        
        if tool_info:
            tool_name = tool_info["tool_name"]
            tool_params = tool_info["params"]
            
            try:
                logger.info(f"ChatAgent nutzt Tool: {tool_name} mit Parametern: {tool_params}")
                tool_result = self.execute_tool(tool_name, **tool_params)
                tool_used = tool_name
            except Exception as e:
                logger.error(f"Fehler bei Tool-Ausführung {tool_name}: {e}")
                tool_result = f"Fehler bei Tool-Ausführung: {str(e)}"
        
        # Erstelle Messages für Modell
        messages = []
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
        
        # Aktuelle Nachricht mit Tool-Ergebnissen
        user_message = message
        if tool_result is not None:
            # Formatiere Tool-Ergebnis für bessere Verarbeitung durch das Modell
            if isinstance(tool_result, dict):
                # Spezielle Formatierung für Web-Suche
                if tool_used == "web_search" and "results" in tool_result:
                    tool_result_str = "Web-Suche Ergebnisse:\n"
                    for i, result in enumerate(tool_result.get("results", [])[:5], 1):
                        title = result.get("title", "Kein Titel")
                        snippet = result.get("snippet", "Keine Beschreibung")
                        url = result.get("url", "")
                        tool_result_str += f"\n{i}. {title}\n   {snippet}\n"
                        if url:
                            tool_result_str += f"   URL: {url}\n"
                    if tool_result.get("summary"):
                        tool_result_str += f"\n{tool_result['summary']}\n"
                else:
                    # Andere Tool-Ergebnisse als JSON
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
            else:
                tool_result_str = str(tool_result)
            
            user_message = f"{message}\n\n[Tool-Ergebnis von {tool_used}]:\n{tool_result_str}\n\nBitte beantworte die Frage des Users basierend auf diesen Tool-Ergebnissen. Gib eine direkte, hilfreiche Antwort."
        
        messages.append({"role": "user", "content": user_message})
        
        # Stelle sicher, dass Messages korrekt alternieren (user/assistant/user/assistant)
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
        try:
            response = self.model_manager.generate(
                messages,
                max_length=1024,  # Längere Antworten für Tool-Ergebnisse
                temperature=0.7
            )
            
            # Bereinige Response (entferne System-Prompt-Phrasen)
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
            
            logger.info(f"ChatAgent Antwort generiert: {len(cleaned_response)} Zeichen, Tool verwendet: {tool_used}")
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Fehler bei Antwort-Generierung: {e}")
            raise


