"""
Vision Agent - Beschreibt generierte Bilder
"""
from typing import Optional, Dict
from .base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """Agent der Bilder beschreibt"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "vision_agent"
        self.name = "Vision Agent"
        self.system_prompt = """Du bist ein Bildbeschreibungs-Agent.
Deine Aufgabe ist es, Bilder detailliert und präzise zu beschreiben.

WICHTIG:
- Beschreibe das Bild detailliert und präzise
- Erwähne wichtige visuelle Elemente (Objekte, Farben, Komposition, Stil, etc.)
- Antworte auf Deutsch
- Sei objektiv und beschreibend"""
        
        self.available_tools = ["describe_image"]
    
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """Beschreibt ein Bild"""
        if not self.model_manager:
            raise RuntimeError("ModelManager nicht gesetzt")
        
        # Die message sollte ein Base64-kodiertes Bild sein oder ein Hinweis darauf
        # Prüfe ob es Base64 ist
        image_base64 = None
        
        if message.startswith("data:image") or len(message) > 1000:
            # Wahrscheinlich Base64-Bild
            if message.startswith("data:image"):
                # Entferne Data-URL-Präfix
                image_base64 = message.split(",", 1)[1] if "," in message else message
            else:
                image_base64 = message
        else:
            # Suche in History nach Bildern
            for msg in reversed(self.message_history):
                if msg["role"] == "assistant" and ("Base64" in msg["content"] or len(msg["content"]) > 1000):
                    # Versuche Base64 zu extrahieren
                    content = msg["content"]
                    if "Base64:" in content:
                        # Extrahiere Base64-String
                        parts = content.split("Base64:")
                        if len(parts) > 1:
                            image_base64 = parts[1].strip().split()[0]
                            break
        
        if not image_base64:
            return "Fehler: Kein Bild gefunden. Bitte sende ein Base64-kodiertes Bild."
        
        try:
            # Verwende describe_image Tool
            description = self.execute_tool(
                "describe_image",
                image_base64=image_base64,
                model_id=self.model_id
            )
            
            logger.info("Bild beschrieben")
            return description
        except Exception as e:
            logger.error(f"Fehler bei Bildbeschreibung: {e}")
            return f"Fehler bei Bildbeschreibung: {str(e)}"












