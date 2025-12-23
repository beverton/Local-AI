"""
Agent Manager - Verwaltet Agenten pro Conversation
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    """Verwaltet Agenten - pro Conversation getrennt"""
    
    def __init__(self):
        # Agent-Registry: Verfügbare Agent-Typen
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        
        # Agent-Instanzen: conversation_id -> {agent_id -> agent_instance}
        self.agent_instances: Dict[str, Dict[str, Any]] = {}
        
        # Tool-Registry: Verfügbare Tools
        self.tool_registry: Dict[str, callable] = {}
        
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Registriert die Standard-Agent-Typen"""
        self.agent_registry = {
            "prompt_agent": {
                "name": "Prompt Agent",
                "description": "Erstellt detaillierte Bildbeschreibungen/Prompts aus Text",
                "model_type": "text",
                "class": None  # Wird später gesetzt
            },
            "image_agent": {
                "name": "Image Agent",
                "description": "Generiert Bilder basierend auf Prompts",
                "model_type": "image",
                "class": None
            },
            "vision_agent": {
                "name": "Vision Agent",
                "description": "Beschreibt generierte Bilder",
                "model_type": "text",
                "class": None
            }
        }
    
    def register_agent_type(self, agent_type: str, name: str, description: str, 
                           model_type: str, agent_class):
        """Registriert einen neuen Agent-Typ"""
        self.agent_registry[agent_type] = {
            "name": name,
            "description": description,
            "model_type": model_type,
            "class": agent_class
        }
    
    def get_available_agent_types(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Agent-Typen zurück"""
        return {
            agent_type: {
                "name": info["name"],
                "description": info["description"],
                "model_type": info["model_type"]
            }
            for agent_type, info in self.agent_registry.items()
        }
    
    def create_agent(self, conversation_id: str, agent_type: str, 
                    model_id: Optional[str] = None, config: Optional[Dict] = None,
                    set_managers_func: Optional[callable] = None) -> str:
        """
        Erstellt einen neuen Agent für eine Conversation
        
        Args:
            conversation_id: Die ID der Conversation
            agent_type: Der Typ des Agenten (z.B. "prompt_agent")
            model_id: Optionales Modell für den Agenten
            config: Optionale Konfiguration
            set_managers_func: Optional: Funktion zum Setzen der Manager
            
        Returns:
            Die ID des erstellten Agenten
        """
        if agent_type not in self.agent_registry:
            raise ValueError(f"Unbekannter Agent-Typ: {agent_type}")
        
        agent_info = self.agent_registry[agent_type]
        agent_class = agent_info.get("class")
        
        if agent_class is None:
            raise ValueError(f"Agent-Klasse für {agent_type} nicht registriert")
        
        # Stelle sicher, dass Conversation-Instanzen existieren
        if conversation_id not in self.agent_instances:
            self.agent_instances[conversation_id] = {}
        
        # Erstelle Agent-Instanz
        agent_id = str(uuid.uuid4())
        agent_instance = agent_class(
            agent_id=agent_id,
            conversation_id=conversation_id,
            model_id=model_id,
            config=config or {}
        )
        
        # Setze Manager falls Funktion bereitgestellt
        if set_managers_func:
            set_managers_func(agent_instance)
        
        self.agent_instances[conversation_id][agent_id] = agent_instance
        
        logger.info(f"Agent {agent_id} ({agent_type}) für Conversation {conversation_id} erstellt")
        return agent_id
    
    def get_agent(self, conversation_id: str, agent_id: str) -> Optional[Any]:
        """Gibt einen Agent zurück"""
        if conversation_id not in self.agent_instances:
            return None
        return self.agent_instances[conversation_id].get(agent_id)
    
    def get_conversation_agents(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Gibt alle Agenten einer Conversation zurück"""
        if conversation_id not in self.agent_instances:
            return []
        
        agents = []
        for agent_id, agent_instance in self.agent_instances[conversation_id].items():
            agents.append({
                "id": agent_id,
                "type": agent_instance.agent_type,
                "name": agent_instance.name,
                "model_id": agent_instance.model_id,
                "created_at": agent_instance.created_at
            })
        return agents
    
    def delete_agent(self, conversation_id: str, agent_id: str) -> bool:
        """Löscht einen Agent"""
        if conversation_id not in self.agent_instances:
            return False
        
        if agent_id in self.agent_instances[conversation_id]:
            del self.agent_instances[conversation_id][agent_id]
            logger.info(f"Agent {agent_id} aus Conversation {conversation_id} gelöscht")
            return True
        return False
    
    def delete_conversation_agents(self, conversation_id: str):
        """Löscht alle Agenten einer Conversation"""
        if conversation_id in self.agent_instances:
            del self.agent_instances[conversation_id]
            logger.info(f"Alle Agenten für Conversation {conversation_id} gelöscht")
    
    def register_tool(self, tool_name: str, tool_function: callable):
        """Registriert ein Tool"""
        self.tool_registry[tool_name] = tool_function
        logger.info(f"Tool '{tool_name}' registriert")
    
    def get_available_tools(self) -> List[str]:
        """Gibt alle verfügbaren Tools zurück"""
        return list(self.tool_registry.keys())
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Führt ein Tool aus"""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool '{tool_name}' nicht gefunden")
        
        tool_function = self.tool_registry[tool_name]
        return tool_function(**kwargs)
    
    def call_agent(self, conversation_id: str, from_agent_id: str, 
                  to_agent_id: str, message: str) -> str:
        """
        Ermöglicht Kommunikation zwischen Agenten
        
        Args:
            conversation_id: Die ID der Conversation
            from_agent_id: Die ID des sendenden Agenten
            to_agent_id: Die ID des empfangenden Agenten
            message: Die Nachricht
            
        Returns:
            Die Antwort des empfangenden Agenten
        """
        to_agent = self.get_agent(conversation_id, to_agent_id)
        if not to_agent:
            raise ValueError(f"Agent {to_agent_id} nicht gefunden")
        
        logger.info(f"Agent {from_agent_id} ruft Agent {to_agent_id} auf")
        response = to_agent.process_message(message, from_agent_id=from_agent_id)
        return response

