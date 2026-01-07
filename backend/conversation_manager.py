"""
Conversation Manager - Verwaltet Gesprächsverläufe
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationManager:
    """Verwaltet Gesprächsverläufe - speichert, lädt und verwaltet Conversations"""
    
    def __init__(self, conversations_dir: str = "data/conversations"):
        self.conversations_dir = conversations_dir
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Stellt sicher, dass das Conversations-Verzeichnis existiert"""
        os.makedirs(self.conversations_dir, exist_ok=True)
    
    def create_conversation(self, title: Optional[str] = None, model_id: Optional[str] = None, conversation_type: str = "chat") -> str:
        """
        Erstellt eine neue Conversation
        
        Args:
            title: Optionaler Titel für die Conversation
            model_id: Optionales Modell für die Conversation
            conversation_type: Typ der Conversation ("chat" oder "image")
            
        Returns:
            Die ID der neuen Conversation
        """
        conversation_id = str(uuid.uuid4())
        if not title:
            if conversation_type == "image":
                title = f"Bild {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            else:
                title = f"Gespräch {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conversation = {
            "id": conversation_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "model_id": model_id,  # Modell pro Conversation
            "conversation_type": conversation_type,  # Typ der Conversation
            "file_mode": False  # File-Mode (nur Datei-Operationen, Standard: deaktiviert)
        }
        
        self._save_conversation(conversation)
        logger.info(f"Neue Conversation erstellt: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Fügt eine Nachricht zu einer Conversation hinzu mit optionalen Metadaten (Quellen, Quality-Info, etc.)
        
        Args:
            conversation_id: Die ID der Conversation
            role: "user" oder "assistant"
            content: Der Nachrichteninhalt
            metadata: Optionale Metadaten (z.B. sources, quality, response_id)
            
        Returns:
            True wenn erfolgreich
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata  # Enthält: sources, quality, response_id, etc.
        
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Aktualisiere Titel basierend auf erster User-Nachricht
        if len(conversation["messages"]) == 1 and role == "user":
            conversation["title"] = content[:50] + ("..." if len(content) > 50 else "")
        
        self._save_conversation(conversation)
        return True
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Lädt eine Conversation
        
        Args:
            conversation_id: Die ID der Conversation
            
        Returns:
            Die Conversation oder None wenn nicht gefunden
        """
        file_path = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Conversation: {e}")
            return None
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Gibt alle Conversations zurück (nur Metadaten, keine Messages)
        
        Returns:
            Liste von Conversations mit Metadaten
        """
        conversations = []
        
        if not os.path.exists(self.conversations_dir):
            return conversations
        
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith('.json'):
                conversation_id = filename[:-5]  # Entferne .json
                conversation = self.get_conversation(conversation_id)
                if conversation:
                    # Nur Metadaten zurückgeben
                    conversations.append({
                        "id": conversation["id"],
                        "title": conversation["title"],
                        "created_at": conversation["created_at"],
                        "updated_at": conversation["updated_at"],
                        "message_count": len(conversation.get("messages", [])),
                        "model_id": conversation.get("model_id"),  # Modell-ID hinzufügen
                        "conversation_type": conversation.get("conversation_type", "chat")  # Default zu "chat" für alte Conversations
                    })
        
        # Sortiere nach updated_at (neueste zuerst)
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Löscht eine Conversation
        
        Args:
            conversation_id: Die ID der Conversation
            
        Returns:
            True wenn erfolgreich
        """
        file_path = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                return False
        
        #return False
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Gibt die Message-History einer Conversation zurück (für Model-Input)
        
        Args:
            conversation_id: Die ID der Conversation
            
        Returns:
            Liste von Messages im Format [{"role": "user", "content": "..."}, ...]
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation.get("messages", [])
        ]
    
    def set_conversation_model(self, conversation_id: str, model_id: Optional[str]) -> bool:
        """
        Setzt das Modell für eine Conversation
        
        Args:
            conversation_id: Die ID der Conversation
            model_id: Die ID des Modells (None zum Entfernen)
            
        Returns:
            True wenn erfolgreich
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation["model_id"] = model_id
        conversation["updated_at"] = datetime.now().isoformat()
        self._save_conversation(conversation)
        logger.info(f"Modell für Conversation {conversation_id} gesetzt: {model_id}")
        return True
    
    def get_conversation_model(self, conversation_id: str) -> Optional[str]:
        """
        Gibt das Modell einer Conversation zurück
        
        Args:
            conversation_id: Die ID der Conversation
            
        Returns:
            Die Modell-ID oder None
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        return conversation.get("model_id")
    
    def set_file_mode(self, conversation_id: str, enabled: bool) -> bool:
        """
        Aktiviert oder deaktiviert den File-Mode für eine Conversation (nur Datei-Operationen)
        Web-Search ist immer aktiv (über Quality Management)
        
        Args:
            conversation_id: Die ID der Conversation
            enabled: True um File-Mode zu aktivieren, False um zu deaktivieren
            
        Returns:
            True wenn erfolgreich
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation["file_mode"] = enabled
        conversation["updated_at"] = datetime.now().isoformat()
        self._save_conversation(conversation)
        logger.info(f"File-Mode für Conversation {conversation_id} gesetzt: {enabled}")
        return True
    
    def get_file_mode(self, conversation_id: str) -> bool:
        """
        Gibt den File-Mode-Status einer Conversation zurück
        
        Args:
            conversation_id: Die ID der Conversation
            
        Returns:
            True wenn File-Mode aktiviert ist, False sonst (auch wenn Conversation nicht existiert)
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        return conversation.get("file_mode", False)
    
    # Alias für Rückwärtskompatibilität (wird später entfernt)
    def set_agent_mode(self, conversation_id: str, enabled: bool) -> bool:
        """Alias für set_file_mode (Rückwärtskompatibilität)"""
        return self.set_file_mode(conversation_id, enabled)
    
    def get_agent_mode(self, conversation_id: str) -> bool:
        """Alias für get_file_mode (Rückwärtskompatibilität)"""
        return self.get_file_mode(conversation_id)
    
    def _save_conversation(self, conversation: Dict[str, Any]):
        """Speichert eine Conversation in eine JSON-Datei"""
        file_path = os.path.join(self.conversations_dir, f"{conversation['id']}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Conversation: {e}")


