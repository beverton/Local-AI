"""
Preference Learner - Optionales System zum Lernen aus Interaktionen
"""
import json
import os
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceLearner:
    """Lernt Präferenzen aus Gesprächen und passt System-Prompts an"""
    
    def __init__(self, preferences_file: str = "data/preferences.json"):
        self.preferences_file = preferences_file
        self.preferences = self._load_preferences()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Stellt sicher, dass das data-Verzeichnis existiert"""
        os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Lädt gespeicherte Präferenzen"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Präferenzen: {e}")
        
        # Default-Präferenzen
        return {
            "enabled": False,
            "style_preferences": {},
            "topic_preferences": {},
            "format_preferences": {},
            "system_prompt_additions": ""
        }
    
    def is_enabled(self) -> bool:
        """Prüft ob Preference Learning aktiviert ist"""
        return self.preferences.get("enabled", False)
    
    def set_enabled(self, enabled: bool):
        """Aktiviert oder deaktiviert Preference Learning"""
        self.preferences["enabled"] = enabled
        self._save_preferences()
    
    def get_system_prompt(self, base_prompt: str = "") -> str:
        """
        Gibt den System-Prompt zurück (mit oder ohne Präferenzen)
        
        Args:
            base_prompt: Der Basis-System-Prompt
            
        Returns:
            Der vollständige System-Prompt
        """
        if not self.is_enabled():
            return base_prompt
        
        additions = self.preferences.get("system_prompt_additions", "")
        if additions:
            return f"{base_prompt}\n\n{additions}"
        
        return base_prompt
    
    def learn_from_conversation(self, messages: List[Dict[str, str]]):
        """
        Analysiert eine Conversation und extrahiert Präferenzen
        
        Args:
            messages: Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
        """
        if not self.is_enabled():
            return
        
        # Einfache Präferenz-Extraktion (kann später erweitert werden)
        # Hier analysieren wir die User-Messages nach Mustern
        
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        
        # Beispiel: Erkenne Format-Präferenzen
        for msg in user_messages:
            if "code" in msg.lower() or "programmieren" in msg.lower():
                self.preferences["topic_preferences"]["coding"] = \
                    self.preferences["topic_preferences"].get("coding", 0) + 1
        
        # Beispiel: Erkenne Stil-Präferenzen
        if any("kurz" in msg.lower() or "knapp" in msg.lower() for msg in user_messages):
            self.preferences["style_preferences"]["brief"] = True
        
        # Aktualisiere System-Prompt basierend auf Präferenzen
        self._update_system_prompt()
        self._save_preferences()
    
    def _update_system_prompt(self):
        """Aktualisiert den System-Prompt basierend auf gelernten Präferenzen"""
        additions = []
        
        # Style-Präferenzen
        if self.preferences.get("style_preferences", {}).get("brief"):
            additions.append("Antworte kurz und prägnant.")
        
        # Topic-Präferenzen
        if self.preferences.get("topic_preferences", {}).get("coding", 0) > 3:
            additions.append("Der Nutzer interessiert sich besonders für Programmierung.")
        
        self.preferences["system_prompt_additions"] = " ".join(additions)
    
    def reset_preferences(self):
        """Setzt alle Präferenzen auf Default zurück"""
        self.preferences = {
            "enabled": self.preferences.get("enabled", False),  # Enabled-Status behalten
            "style_preferences": {},
            "topic_preferences": {},
            "format_preferences": {},
            "system_prompt_additions": ""
        }
        self._save_preferences()
        logger.info("Präferenzen zurückgesetzt")
    
    def get_preferences(self) -> Dict[str, Any]:
        """Gibt die aktuellen Präferenzen zurück"""
        return self.preferences.copy()
    
    def _save_preferences(self):
        """Speichert Präferenzen in JSON-Datei"""
        try:
            with open(self.preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Präferenzen: {e}")











