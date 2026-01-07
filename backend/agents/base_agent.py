"""
Base Agent - Basis-Klasse für alle Agenten
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Basis-Klasse für alle Agenten"""
    
    def __init__(self, agent_id: str, conversation_id: str, 
                 model_id: Optional[str] = None, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        self.model_id = model_id
        self.config = config or {}
        self.created_at = datetime.now().isoformat()
        self.available_tools: List[str] = []
        self.message_history: List[Dict[str, str]] = []
        
        # Wird von Subklassen gesetzt
        self.agent_type: str = ""
        self.name: str = ""
        self.system_prompt: str = ""
        self.model_manager = None  # Wird später gesetzt
        self.agent_manager = None  # Wird später gesetzt
        self.image_manager = None  # Wird später gesetzt
        self.model_service_client = None  # Wird später gesetzt (für Model Service)
    
    def set_model_manager(self, model_manager):
        """Setzt den ModelManager"""
        self.model_manager = model_manager
    
    def set_agent_manager(self, agent_manager):
        """Setzt den AgentManager"""
        self.agent_manager = agent_manager
    
    def set_image_manager(self, image_manager):
        """Setzt den ImageManager"""
        self.image_manager = image_manager
    
    def set_model_service_client(self, model_service_client):
        """Setzt den Model Service Client"""
        self.model_service_client = model_service_client
    
    def process_message(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """
        Verarbeitet eine Nachricht
        
        Args:
            message: Die Nachricht
            from_agent_id: Optional: ID des sendenden Agenten
            
        Returns:
            Die Antwort
        """
        # Füge Nachricht zur History hinzu
        self.message_history.append({
            "role": "user" if from_agent_id is None else f"agent_{from_agent_id}",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Generiere Antwort
            response = self._generate_response(message, from_agent_id)
        except RuntimeError as e:
            # RuntimeError (z.B. ImageManager nicht verfügbar, Modell nicht geladen)
            error_msg = f"Runtime-Fehler: {str(e)}"
            logger.error(f"Agent {self.agent_id} ({self.agent_type}): {error_msg}")
            response = f"Fehler: {error_msg}"
        except Exception as e:
            # Alle anderen Fehler abfangen, damit der Agent nicht abstürzt
            error_msg = f"Unerwarteter Fehler: {str(e)}"
            logger.error(f"Agent {self.agent_id} ({self.agent_type}) ist abgestürzt: {e}", exc_info=True)
            response = f"Fehler: {error_msg}. Bitte versuchen Sie es erneut oder kontaktieren Sie den Support."
        
        # Füge Antwort zur History hinzu
        self.message_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    @abstractmethod
    def _generate_response(self, message: str, from_agent_id: Optional[str] = None) -> str:
        """
        Generiert eine Antwort (muss von Subklassen implementiert werden)
        
        Args:
            message: Die Nachricht
            from_agent_id: Optional: ID des sendenden Agenten
            
        Returns:
            Die Antwort
        """
        pass
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Führt ein Tool aus
        
        Args:
            tool_name: Der Name des Tools
            **kwargs: Tool-Parameter
            
        Returns:
            Das Tool-Ergebnis
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' nicht verfügbar für diesen Agenten")
        
        if not self.agent_manager:
            raise RuntimeError("AgentManager nicht gesetzt")
        
        return self.agent_manager.execute_tool(tool_name, **kwargs)
    
    def call_agent(self, target_agent_id: str, message: str) -> str:
        """
        Ruft einen anderen Agenten auf
        
        Args:
            target_agent_id: Die ID des Ziel-Agenten
            message: Die Nachricht
            
        Returns:
            Die Antwort des Ziel-Agenten
        """
        if not self.agent_manager:
            raise RuntimeError("AgentManager nicht gesetzt")
        
        return self.agent_manager.call_agent(
            self.conversation_id,
            self.agent_id,
            target_agent_id,
            message
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Gibt Informationen über den Agenten zurück"""
        return {
            "id": self.agent_id,
            "type": self.agent_type,
            "name": self.name,
            "conversation_id": self.conversation_id,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "available_tools": self.available_tools,
            "message_count": len(self.message_history)
        }

