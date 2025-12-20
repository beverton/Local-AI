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
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Erstellt eine neue Conversation
        
        Args:
            title: Optionaler Titel für die Conversation
            
        Returns:
            Die ID der neuen Conversation
        """
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "title": title or f"Gespräch {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        
        self._save_conversation(conversation)
        logger.info(f"Neue Conversation erstellt: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """
        Fügt eine Nachricht zu einer Conversation hinzu
        
        Args:
            conversation_id: Die ID der Conversation
            role: "user" oder "assistant"
            content: Der Nachrichteninhalt
            
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
                        "message_count": len(conversation.get("messages", []))
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
                logger.info(f"Conversation gelöscht: {conversation_id}")
                return True
            except Exception as e:
                logger.error(f"Fehler beim Löschen der Conversation: {e}")
                return False
        
        return False
    
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
    
    def _save_conversation(self, conversation: Dict[str, Any]):
        """Speichert eine Conversation in eine JSON-Datei"""
        file_path = os.path.join(self.conversations_dir, f"{conversation['id']}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Conversation: {e}")


