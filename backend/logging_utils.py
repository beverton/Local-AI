"""
Zentrales Logging-System für Local AI
Strukturierte Logs mit Tags für einfaches Filtern und Debugging
"""
import logging
import sys
import os
from typing import Optional
from datetime import datetime

class StructuredLogger:
    """
    Strukturierter Logger mit Tags für verschiedene System-Bereiche
    """
    
    # Definiere Log-Tags für verschiedene Bereiche
    TAGS = {
        "MODEL_LOAD": "[MODEL_LOAD]",
        "MODEL_GEN": "[MODEL_GEN]",
        "CHAT": "[CHAT]",
        "QUALITY": "[QUALITY]",
        "WEB_SEARCH": "[WEB_SEARCH]",
        "VALIDATION": "[VALIDATION]",
        "ERROR": "[ERROR]",
        "API": "[API]",
        "MCP": "[MCP]",
        "CONFIG": "[CONFIG]",
        "PERF": "[PERF]",
    }
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialisiere Logger
        
        Args:
            name: Logger-Name (normalerweise __name__)
            log_file: Optional: Pfad zur Log-Datei
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Verhindere doppelte Handler
        if self.logger.handlers:
            return
        
        # Format: TIMESTAMP LEVEL TAG MESSAGE
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File Handler (wenn angegeben)
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Konnte Log-Datei nicht erstellen: {e}")
    
    def _format_message(self, tag: str, message: str) -> str:
        """Formatiere Log-Nachricht mit Tag"""
        tag_str = self.TAGS.get(tag, f"[{tag}]")
        return f"{tag_str} {message}"
    
    def debug(self, message: str, tag: Optional[str] = None):
        """Debug-Log"""
        msg = self._format_message(tag, message) if tag else message
        self.logger.debug(msg)
    
    def info(self, message: str, tag: Optional[str] = None):
        """Info-Log"""
        msg = self._format_message(tag, message) if tag else message
        self.logger.info(msg)
    
    def warning(self, message: str, tag: Optional[str] = None):
        """Warning-Log"""
        msg = self._format_message(tag, message) if tag else message
        self.logger.warning(msg)
    
    def error(self, message: str, tag: Optional[str] = None, exc_info: bool = False):
        """Error-Log mit optionalem Exception-Info"""
        msg = self._format_message(tag, message) if tag else message
        self.logger.error(msg, exc_info=exc_info)
    
    def exception(self, message: str, tag: Optional[str] = None):
        """Exception-Log mit vollständigem Traceback"""
        msg = self._format_message(tag, message) if tag else message
        self.logger.exception(msg)
    
    def model_load(self, message: str, level: str = "info"):
        """Spezielle Methode für Model-Load-Logs"""
        getattr(self, level)(message, tag="MODEL_LOAD")
    
    def model_gen(self, message: str, level: str = "info"):
        """Spezielle Methode für Model-Generation-Logs"""
        getattr(self, level)(message, tag="MODEL_GEN")
    
    def chat(self, message: str, level: str = "info"):
        """Spezielle Methode für Chat-Logs"""
        getattr(self, level)(message, tag="CHAT")
    
    def quality(self, message: str, level: str = "info"):
        """Spezielle Methode für Quality-Logs"""
        getattr(self, level)(message, tag="QUALITY")
    
    def web_search(self, message: str, level: str = "info"):
        """Spezielle Methode für Web-Search-Logs"""
        getattr(self, level)(message, tag="WEB_SEARCH")
    
    def validation(self, message: str, level: str = "info"):
        """Spezielle Methode für Validation-Logs"""
        getattr(self, level)(message, tag="VALIDATION")
    
    def api(self, message: str, level: str = "info"):
        """Spezielle Methode für API-Logs"""
        getattr(self, level)(message, tag="API")
    
    def error_log(self, message: str, exc_info: bool = True):
        """Spezielle Methode für Error-Logs"""
        self.error(message, tag="ERROR", exc_info=exc_info)


def get_logger(name: str, log_file: Optional[str] = None) -> StructuredLogger:
    """
    Factory-Funktion für StructuredLogger
    
    Args:
        name: Logger-Name (normalerweise __name__)
        log_file: Optional: Pfad zur Log-Datei
    
    Returns:
        StructuredLogger-Instanz
    """
    return StructuredLogger(name, log_file)


# Globale Log-Datei
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "local_ai.log")
