"""
MCP Adapter - Adapter zwischen MCP-Format und Model Service Format
Konvertiert MCP-Messages in Model Service Requests und umgekehrt
"""
import logging
from typing import Dict, Any, List, Optional
from model_service_client import ModelServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPAdapter:
    """Adapter zwischen MCP und Model Service"""
    
    def __init__(self, model_service: ModelServiceClient):
        self.model_service = model_service
    
    def mcp_messages_to_model_service(self, mcp_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Konvertiert MCP-Messages in Model Service Format
        
        Args:
            mcp_messages: Liste von MCP-Messages mit "role" und "content"
            
        Returns:
            Liste von Model Service Messages
        """
        # MCP und Model Service verwenden das gleiche Format
        # Aber wir können hier Anpassungen vornehmen falls nötig
        return mcp_messages.copy()
    
    def model_service_to_mcp(self, model_service_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Konvertiert Model Service Response in MCP-Format
        
        Args:
            model_service_response: Response vom Model Service
            
        Returns:
            MCP-formatierte Response
        """
        response_text = model_service_response.get("response", "")
        
        return {
            "content": [{
                "type": "text",
                "text": response_text
            }]
        }
    
    def check_model_service_status(self) -> Dict[str, Any]:
        """Prüft Status des Model Service"""
        if not self.model_service.is_available():
            return {
                "available": False,
                "error": "Model Service nicht erreichbar"
            }
        
        status = self.model_service.get_status()
        if not status:
            return {
                "available": False,
                "error": "Fehler beim Abrufen des Status"
            }
        
        return {
            "available": True,
            "status": status
        }
    
    def ensure_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """
        Stellt sicher, dass ein Modell geladen ist
        
        Args:
            model_id: Optional: Spezifisches Modell (sonst aktuelles)
            
        Returns:
            True wenn Modell geladen ist oder geladen wurde
        """
        if not self.model_service.is_available():
            return False
        
        status = self.model_service.get_text_model_status()
        if not status:
            return False
        
        if status.get("loaded"):
            if model_id is None or status.get("model_id") == model_id:
                return True
        
        # Versuche Modell zu laden
        if model_id:
            return self.model_service.load_text_model(model_id)
        else:
            # Verwende aktuelles Modell
            current_model = status.get("model_id")
            if current_model:
                return self.model_service.load_text_model(current_model)
        
        return False







