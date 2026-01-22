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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Agent für normale Gespräche mit automatischer Tool-Nutzung"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "chat_agent"
        self.name = "Chat Agent"
        self.system_prompt = """Du bist ein hilfreicher AI-Assistent. Antworte NUR auf Deutsch.

KRITISCH - Tool-Ergebnisse nutzen:
Wenn dir Tool-Ergebnisse (Web-Suche, Dateien) gezeigt werden:
1. Nutze AUSSCHLIESSLICH diese Informationen - dein Wissen ist veraltet
2. Kopiere URLs EXAKT wie gezeigt - keine Leerzeichen, keine Änderungen
3. Tool-Ergebnisse haben IMMER Vorrang vor deinem internen Wissen

Beispiel: Wenn Tool zeigt "X ist Y seit 2025" aber du glaubst "X ist Z" → Tool hat Recht, antworte mit "X ist Y seit 2025".

Antworte präzise, klar und ausschließlich auf Deutsch."""
        
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
        
        # WebSearch Patterns - mit verbesserter Erkennung
        # WICHTIG: Patterns die mit Fragen beginnen haben Priorität (wer/was/wo/wann/wie)
        web_search_patterns = [
            # Fragewörter - HOHE PRIORITÄT (extrahieren das Thema direkt)
            r"^(.+?)\?.*?(?:suche|finde|google|web|website)",  # "Wer ist X? Suche..." -> "Wer ist X"
            r"wer\s+(?:ist|sind|war|waren)\s+(.+?)(?:\?|$)",  # "Wer ist...?"
            r"was\s+(?:ist|wird|sind|war|bedeutet)\s+(.+?)(?:\?|$)",  # "Was ist...?"
            r"wo\s+(?:ist|sind|finde\s+ich|liegt|befindet\s+sich)\s+(.+?)(?:\?|$)",  # "Wo ist...?"
            r"wann\s+(?:ist|war|findet|beginnt)\s+(.+?)(?:\?|$)",  # "Wann ist...?"
            r"wie\s+(?:viel|viele|teuer|hoch)\s+(.+?)(?:\?|$)",  # "Wie viel...?"
            # Such-Keywords mit spezifischem Kontext
            r"(?:suche|finde|google)\s+(?:nach\s+)?(?:informationen\s+)?(?:über|zu)\s+(.+)",  # "suche nach Infos über X"
            r"website\s+(?:von|für|über|zu)\s+(.+)",  # "Website von X"
            r"webseite\s+(?:von|für|über|zu)\s+(.+)",  # "Webseite von X"
            # Wetter-spezifisch
            r"wetter\s+(?:in|für|heute|morgen)\s+(.+)",  # "Wetter in..."
            r"wettervorhersage\s+(?:für|in)\s+(.+)",  # "Wettervorhersage für..."
            # News/Aktualität
            r"(?:aktuelle|neueste)\s+(?:informationen|infos|news|nachrichten)\s+(?:über|zu|von)\s+(.+)",
            # Generische Such-Keywords - NIEDRIGE PRIORITÄT (am Ende)
            r"(?:suche|finde|google)\s+(.+)",  # Generisch: "suche X"
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
                
                # Bereinige Query: Entferne trailing Punkte/Kommas und Such-Keywords am Ende
                query = re.sub(r'[,\.;]+$', '', query)  # Entferne trailing Satzzeichen
                query = re.sub(r'\s+(?:suche|finde|google|web|website|webseite|link|url).*$', '', query)  # Entferne Such-Keywords am Ende
                query = query.strip()
                
                if len(query) > 3:  # Mindestlänge für sinnvolle Suche
                    return {"tool_name": "web_search", "params": {"query": query}}
        
        # Datei-Operation Patterns
        # read_file - Erweiterte Patterns für natürlichere Formulierungen
        read_patterns = [
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
            match = re.search(pattern, message_original, flags=re.IGNORECASE)
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
        
        # HINWEIS: Modell-Laden wird von main.py/Model-Service gehandhabt
        # Keine redundante Prüfung hier, da bei Model-Service das Modell
        # nicht im lokalen model_manager geladen ist
        
        # Erkenne Tool-Bedarf
        tool_info = self._detect_tool_need(message)
        tool_result = None
        tool_used = None
        
        if tool_info:
            tool_name = tool_info["tool_name"]
            tool_params = tool_info["params"]
            
            # WICHTIG: Prüfe ob Tool durch UI-Toggle aktiviert ist
            if not self._is_tool_enabled(tool_name):
                logger.debug(f"Tool '{tool_name}' ist durch UI-Toggle deaktiviert - überspringe Tool-Nutzung")
                tool_info = None  # Überspringe Tool-Nutzung
            else:
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
            
            user_message = f"""FRAGE: {message}

TOOL-ERGEBNIS [{tool_used}]:
{tool_result_str}

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
            effective_max_length = self.config.get("max_length", 1024)
            effective_temperature = self.config.get("temperature", 0.7)
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
            if tool_used == "web_search" and tool_result and isinstance(tool_result, dict):
                cleaned_response = self._fix_urls_in_response(cleaned_response, tool_result)
            
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


