"""
Output Manager - Verwaltet Ausgabe-Pfade und Dateinamen für generierte Inhalte
"""
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class OutputManager:
    """Verwaltet Ausgabe-Pfade und Dateinamen"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_config = self.config.get("output_paths", {})
        
        # Defaults
        self.base_directory = self.output_config.get(
            "base_directory", 
            "G:\\KI Modelle\\Outputs"
        )
        self.images_subdir = self.output_config.get("images", "generated_images")
        self.conversations_subdir = self.output_config.get("conversations", "conversations")
        self.audio_subdir = self.output_config.get("audio", "audio")
        self.use_date_folders = self.output_config.get("use_date_folders", True)
        self.filename_format = self.output_config.get("filename_format", "{date}_{title}")
        
        # Stelle sicher dass Basis-Verzeichnis existiert
        self._ensure_base_directory()
    
    def _load_config(self) -> Dict:
        """Lädt die Konfiguration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Konnte Config nicht laden: {e}, verwende Defaults")
            return {}
    
    def _ensure_base_directory(self):
        """Stellt sicher dass das Basis-Verzeichnis existiert"""
        try:
            Path(self.base_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Konnte Basis-Verzeichnis nicht erstellen: {e}")
    
    def _sanitize_title(self, text: str, max_length: int = 50) -> str:
        """
        Bereinigt Text für Dateinamen
        
        Args:
            text: Der zu bereinigende Text
            max_length: Maximale Länge des Titels
            
        Returns:
            Bereinigter Text für Dateinamen
        """
        # Entferne/Ersetze ungültige Zeichen
        sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', text)
        
        # Ersetze mehrfache Leerzeichen/Unterstriche
        sanitized = re.sub(r'[\s_]+', '_', sanitized)
        
        # Entferne führende/trailing Unterstriche
        sanitized = sanitized.strip('_')
        
        # Kürze auf max_length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('_')
        
        return sanitized if sanitized else "output"
    
    def _generate_title_from_prompt(self, prompt: str, max_words: int = 5) -> str:
        """
        Generiert einen Titel aus einem Prompt
        
        Args:
            prompt: Der Original-Prompt
            max_words: Maximale Anzahl Wörter im Titel
            
        Returns:
            Generierter Titel
        """
        # Bereinige Prompt
        prompt = prompt.strip()
        
        # Entferne Negativ-Prompt-Marker
        if "negative_prompt:" in prompt.lower():
            prompt = prompt.split("negative_prompt:")[0].strip()
        
        # Teile in Wörter
        words = prompt.split()
        
        # Nehme erste max_words Wörter
        title_words = words[:max_words]
        
        # Füge zusammen
        title = "_".join(title_words)
        
        # Bereinige
        return self._sanitize_title(title)
    
    def get_image_output_path(
        self, 
        prompt: str, 
        extension: str = "png",
        custom_title: Optional[str] = None
    ) -> Path:
        """
        Generiert einen Output-Pfad für ein Bild
        
        Args:
            prompt: Der Prompt (für Titel-Generierung)
            extension: Datei-Endung (default: png)
            custom_title: Optional - Custom Titel statt automatisch generiert
            
        Returns:
            Vollständiger Pfad für die Ausgabedatei
        """
        # Basis-Pfad für Bilder
        images_path = Path(self.base_directory) / self.images_subdir
        
        # Datum-Ordner wenn aktiviert
        if self.use_date_folders:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            images_path = images_path / date_folder
        
        # Erstelle Ordner
        images_path.mkdir(parents=True, exist_ok=True)
        
        # Generiere Titel
        if custom_title:
            title = self._sanitize_title(custom_title)
        else:
            title = self._generate_title_from_prompt(prompt)
        
        # Generiere Dateinamen nach Format
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = self.filename_format.format(
            date=date_str,
            title=title
        )
        
        # Stelle sicher dass Dateiname einzigartig ist
        base_filename = filename
        counter = 1
        while (images_path / f"{filename}.{extension}").exists():
            filename = f"{base_filename}_{counter}"
            counter += 1
        
        return images_path / f"{filename}.{extension}"
    
    def get_conversation_export_path(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        extension: str = "json"
    ) -> Path:
        """
        Generiert einen Export-Pfad für eine Conversation
        
        Args:
            conversation_id: ID der Conversation
            title: Optional - Titel der Conversation
            extension: Datei-Endung (default: json)
            
        Returns:
            Vollständiger Pfad für die Exportdatei
        """
        # Basis-Pfad für Conversations
        conv_path = Path(self.base_directory) / self.conversations_subdir
        
        # Datum-Ordner wenn aktiviert
        if self.use_date_folders:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            conv_path = conv_path / date_folder
        
        # Erstelle Ordner
        conv_path.mkdir(parents=True, exist_ok=True)
        
        # Generiere Dateinamen
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if title:
            sanitized_title = self._sanitize_title(title)
            filename = self.filename_format.format(
                date=date_str,
                title=sanitized_title
            )
        else:
            filename = f"{date_str}_{conversation_id[:8]}"
        
        # Stelle sicher dass Dateiname einzigartig ist
        base_filename = filename
        counter = 1
        while (conv_path / f"{filename}.{extension}").exists():
            filename = f"{base_filename}_{counter}"
            counter += 1
        
        return conv_path / f"{filename}.{extension}"
    
    def get_audio_output_path(
        self,
        title: Optional[str] = None,
        extension: str = "wav"
    ) -> Path:
        """
        Generiert einen Output-Pfad für Audio
        
        Args:
            title: Optional - Titel der Audio-Datei
            extension: Datei-Endung (default: wav)
            
        Returns:
            Vollständiger Pfad für die Ausgabedatei
        """
        # Basis-Pfad für Audio
        audio_path = Path(self.base_directory) / self.audio_subdir
        
        # Datum-Ordner wenn aktiviert
        if self.use_date_folders:
            date_folder = datetime.now().strftime("%Y-%m-%d")
            audio_path = audio_path / date_folder
        
        # Erstelle Ordner
        audio_path.mkdir(parents=True, exist_ok=True)
        
        # Generiere Dateinamen
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if title:
            sanitized_title = self._sanitize_title(title)
            filename = self.filename_format.format(
                date=date_str,
                title=sanitized_title
            )
        else:
            filename = f"{date_str}_audio"
        
        # Stelle sicher dass Dateiname einzigartig ist
        base_filename = filename
        counter = 1
        while (audio_path / f"{filename}.{extension}").exists():
            filename = f"{base_filename}_{counter}"
            counter += 1
        
        return audio_path / f"{filename}.{extension}"
    
    def update_config(
        self,
        base_directory: Optional[str] = None,
        use_date_folders: Optional[bool] = None,
        filename_format: Optional[str] = None
    ) -> bool:
        """
        Aktualisiert die Output-Konfiguration
        
        Args:
            base_directory: Neues Basis-Verzeichnis
            use_date_folders: Ob Datum-Ordner verwendet werden sollen
            filename_format: Neues Dateinamen-Format
            
        Returns:
            True wenn erfolgreich gespeichert
        """
        try:
            # Aktualisiere interne Werte
            if base_directory is not None:
                self.base_directory = base_directory
                self.output_config["base_directory"] = base_directory
            
            if use_date_folders is not None:
                self.use_date_folders = use_date_folders
                self.output_config["use_date_folders"] = use_date_folders
            
            if filename_format is not None:
                self.filename_format = filename_format
                self.output_config["filename_format"] = filename_format
            
            # Aktualisiere Config
            self.config["output_paths"] = self.output_config
            
            # Speichere Config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            # Stelle sicher dass neues Verzeichnis existiert
            self._ensure_base_directory()
            
            logger.info(f"Output-Konfiguration aktualisiert: {self.base_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Output-Konfiguration: {e}")
            return False


# Globale Instanz
_output_manager = None

def get_output_manager(config_path: Optional[str] = None) -> OutputManager:
    """Gibt die globale OutputManager-Instanz zurück"""
    global _output_manager
    if _output_manager is None:
        # Wenn kein Pfad angegeben, verwende Projekt-Root
        if config_path is None:
            # Bestimme Projekt-Root (ein Verzeichnis über backend/)
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(backend_dir)
            config_path = os.path.join(project_root, "config.json")
        _output_manager = OutputManager(config_path)
    return _output_manager
