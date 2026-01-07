"""
Image Agent - Generiert Bilder basierend auf Prompts
"""
from typing import Optional, Dict
from .base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAgent(BaseAgent):
    """Agent der Bilder aus Prompts generiert"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(agent_id, conversation_id, model_id, config)
        self.agent_type = "image_agent"
        self.name = "Image Agent"
        self.system_prompt = """Du bist ein Bildgenerierungs-Agent.
Du erhältst Prompts und generierst daraus Bilder."""
        
        self.available_tools = ["generate_image"]
    
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """Generiert ein Bild aus einem Prompt"""
        if not self.image_manager:
            raise RuntimeError("ImageManager nicht gesetzt")
        
        # Der message sollte ein Prompt sein
        prompt = message.strip()
        
        if not prompt:
            return "Fehler: Kein Prompt erhalten"
        
        try:
            # Verwende generate_image Tool
            image_base64 = self.execute_tool(
                "generate_image",
                prompt=prompt,
                model_id=self.model_id,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=1024,
                height=1024
            )
            
            logger.info(f"Bild generiert: {len(prompt)} Zeichen Prompt")
            return f"Bild erfolgreich generiert. (Base64: {len(image_base64)} Zeichen)"
        except Exception as e:
            logger.error(f"Fehler bei Bildgenerierung: {e}")
            return f"Fehler bei Bildgenerierung: {str(e)}"










