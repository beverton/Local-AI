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
    
    def __init__(self, conversations_dir: Optional[str] = None):
        if conversations_dir is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
            conversations_dir = os.path.join(project_root, "data", "conversations")
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
            conversation_id: Die ID der Conversation (UUID)
            
        Returns:
            Die Conversation oder None wenn nicht gefunden
        """
        # Suche zuerst nach Datei mit UUID-Dateinamen (Rückwärtskompatibilität)
        old_file_path = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        if os.path.exists(old_file_path):
            try:
                with open(old_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Conversation: {e}")
                return None
        
        # Suche nach Datei mit Datum/Uhrzeit-Format
        if not os.path.exists(self.conversations_dir):
            return None
        
        for filename in os.listdir(self.conversations_dir):
            if not filename.endswith('.json'):
                continue
            try:
                file_path = os.path.join(self.conversations_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                    if conv.get('id') == conversation_id:
                        return conv
            except Exception:
                continue
        
        return None
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Gibt alle Conversations zurück (nur Metadaten, keine Messages)
        Liest alle JSON-Dateien im Conversations-Verzeichnis und gibt nur die zurück, die tatsächlich existieren.
        
        Returns:
            Liste von Conversations mit Metadaten
        """
        conversations = []
        
        if not os.path.exists(self.conversations_dir):
            return conversations
        
        # Durchsuche alle JSON-Dateien im Ordner
        for filename in os.listdir(self.conversations_dir):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(self.conversations_dir, filename)
            
            # Prüfe ob Datei existiert (könnte zwischenzeitlich gelöscht worden sein)
            if not os.path.exists(file_path):
                continue
            
            try:
                # Lade Conversation direkt aus Datei (nicht über get_conversation, da Dateiname != ID)
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                
                # Validiere, dass es eine gültige Conversation ist
                if not isinstance(conversation, dict) or 'id' not in conversation:
                    continue
                
                # Nur Metadaten zurückgeben
                conversations.append({
                    "id": conversation["id"],
                    "title": conversation.get("title", "Unbenannt"),
                    "created_at": conversation.get("created_at", datetime.now().isoformat()),
                    "updated_at": conversation.get("updated_at", datetime.now().isoformat()),
                    "message_count": len(conversation.get("messages", [])),
                    "model_id": conversation.get("model_id"),
                    "conversation_type": conversation.get("conversation_type", "chat")
                })
            except (json.JSONDecodeError, IOError, OSError) as e:
                # Ignoriere defekte oder nicht lesbare Dateien
                logger.warning(f"Konnte Conversation-Datei {filename} nicht laden: {e}")
                continue
        
        # Sortiere nach updated_at (neueste zuerst)
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Löscht eine Conversation
        
        Args:
            conversation_id: Die ID der Conversation (UUID)
            
        Returns:
            True wenn erfolgreich
        """
        # Suche nach Datei (entweder UUID-Dateiname oder Datum/Uhrzeit-Format)
        old_file_path = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        if os.path.exists(old_file_path):
            try:
                os.remove(old_file_path)
                logger.info(f"Conversation {conversation_id} gelöscht (UUID-Dateiname)")
                return True
            except Exception as e:
                logger.error(f"Fehler beim Löschen von {conversation_id} (UUID-Dateiname): {e}")
                return False
        
        # Suche nach Datei mit Datum/Uhrzeit-Format
        if not os.path.exists(self.conversations_dir):
            logger.warning(f"Conversations-Verzeichnis existiert nicht: {self.conversations_dir}")
            return False
        
        for filename in os.listdir(self.conversations_dir):
            if not filename.endswith('.json'):
                continue
            try:
                file_path = os.path.join(self.conversations_dir, filename)
                # Prüfe ob Datei existiert (könnte zwischenzeitlich gelöscht worden sein)
                if not os.path.exists(file_path):
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                    if conv.get('id') == conversation_id:
                        os.remove(file_path)
                        logger.info(f"Conversation {conversation_id} gelöscht (Dateiname: {filename})")
                        return True
            except (json.JSONDecodeError, IOError, OSError) as e:
                # Ignoriere defekte Dateien und fahre mit nächster fort
                logger.debug(f"Konnte Datei {filename} nicht lesen: {e}")
                continue
            except Exception as e:
                logger.error(f"Unerwarteter Fehler beim Löschen von {conversation_id} (Datei: {filename}): {e}")
                continue
        
        logger.warning(f"Conversation {conversation_id} nicht gefunden zum Löschen")
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
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Aktualisiert den Titel einer Conversation
        
        Args:
            conversation_id: Die ID der Conversation
            title: Der neue Titel
            
        Returns:
            True wenn erfolgreich
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation["title"] = title
        conversation["updated_at"] = datetime.now().isoformat()
        self._save_conversation(conversation)
        logger.info(f"Titel für Conversation {conversation_id} aktualisiert: {title}")
        return True
    
    def delete_multiple_conversations(self, conversation_ids: List[str]) -> Dict[str, bool]:
        """
        Löscht mehrere Conversations
        
        Args:
            conversation_ids: Liste von Conversation-IDs
            
        Returns:
            Dictionary mit Conversation-ID als Key und Erfolg (True/False) als Value
        """
        results = {}
        for conv_id in conversation_ids:
            results[conv_id] = self.delete_conversation(conv_id)
        return results
    
    def _save_conversation(self, conversation: Dict[str, Any]):
        """Speichert eine Conversation in eine JSON-Datei mit Datum/Uhrzeit im Dateinamen"""
        # Generiere Dateinamen basierend auf created_at (chronologisch sortierbar)
        created_at = conversation.get('created_at', datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except Exception:
            dt = datetime.now()
        
        # Format: YYYY-MM-DD_HH-MM-SS_<uuid-prefix>.json
        # UUID-Präfix für Eindeutigkeit (erste 8 Zeichen)
        uuid_prefix = conversation['id'][:8]
        date_str = dt.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{date_str}_{uuid_prefix}.json"
        
        file_path = os.path.join(self.conversations_dir, filename)
        
        # Wenn eine alte Datei mit UUID-Dateinamen existiert, lösche sie
        old_file_path = os.path.join(self.conversations_dir, f"{conversation['id']}.json")
        if os.path.exists(old_file_path) and old_file_path != file_path:
            try:
                os.remove(old_file_path)
            except Exception:
                pass
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Conversation: {e}")


