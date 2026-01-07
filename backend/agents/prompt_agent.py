"""
Prompt Agent - Erstellt detaillierte Bildbeschreibungen/Prompts aus Text
"""
from typing import Optional, Dict
from .base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptAgent(BaseAgent):
    """Agent der Text in detaillierte Bild-Prompts umwandelt"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "prompt_agent"
        self.name = "Prompt Agent"
        self.system_prompt = """Du bist ein spezialisierter Prompt-Generator für Bildgenerierung.
Deine Aufgabe ist es, aus Benutzeranfragen detaillierte, präzise Prompts für Bildgenerierungsmodelle zu erstellen.

WICHTIG:
- Erstelle präzise, detaillierte Bildbeschreibungen
- Verwende klare, beschreibende Sprache
- Füge relevante Details hinzu (Stil, Beleuchtung, Komposition, etc.)
- Antworte NUR mit dem Prompt, keine zusätzlichen Erklärungen
- Der Prompt sollte direkt für Bildgenerierungsmodelle verwendbar sein"""
        
        self.available_tools = []
    
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """Generiert einen detaillierten Bild-Prompt aus Text"""
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
        
        # Erstelle Messages für Modell
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Füge Kontext aus History hinzu (letzte 5 Nachrichten)
        recent_history = self.message_history[-10:] if len(self.message_history) > 10 else self.message_history
        for msg in recent_history:
            if msg["role"] == "user" or msg["role"].startswith("agent_"):
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Aktuelle Nachricht
        messages.append({"role": "user", "content": message})
        
        # Generiere Prompt
        try:
            response = self.model_manager.generate(
                messages,
                max_length=256,
                temperature=0.7
            )
            logger.info(f"Prompt generiert: {len(response)} Zeichen")
            return response.strip()
        except Exception as e:
            logger.error(f"Fehler bei Prompt-Generierung: {e}")
            raise












