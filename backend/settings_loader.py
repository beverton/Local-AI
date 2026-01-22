"""
Zentrales Settings-Loader-Modul mit Caching
Konsolidiert alle Performance-Settings-Lade-Implementierungen
"""
import json
import os
import threading
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Cache für Performance-Settings
_perf_settings_cache: Optional[Dict[str, Any]] = None
_perf_settings_cache_timestamp: float = 0
_perf_settings_cache_lock = threading.Lock()
_perf_settings_file_path: Optional[str] = None
_cache_ttl: float = 5.0  # Cache-TTL in Sekunden (5s für schnelle Updates)

# Default Settings
DEFAULT_PERFORMANCE_SETTINGS = {
    "cpu_threads": None,  # None = Auto
    "gpu_optimization": "balanced",
    "disable_cpu_offload": False,
    "use_torch_compile": False,
    "use_quantization": False,
    "quantization_bits": 8,
    "use_flash_attention": True,
    "enable_tf32": True,
    "enable_cudnn_benchmark": True,
    "gpu_max_percent": 90.0,
    "primary_budget_percent": None,
    "max_length": 512,
    "temperature": 0.6
}


def _find_performance_settings_file() -> Optional[str]:
    """
    Findet die Performance-Settings-Datei durch Versuch mehrerer Pfade
    
    Returns:
        Pfad zur Datei oder None wenn nicht gefunden
    """
    # Versuche mehrere mögliche Pfade
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "performance_settings.json"),
        os.path.join(os.path.dirname(__file__), "..", "data", "performance_settings.json"),
        "data/performance_settings.json",
        os.path.join(os.getcwd(), "data", "performance_settings.json")
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    return None


def load_performance_settings(force_reload: bool = False) -> Dict[str, Any]:
    """
    Lädt Performance-Settings mit Caching
    
    Args:
        force_reload: Wenn True, Cache ignorieren und neu laden
        
    Returns:
        Dict mit Performance-Settings (merged mit Defaults)
    """
    global _perf_settings_cache, _perf_settings_cache_timestamp, _perf_settings_file_path
    
    with _perf_settings_cache_lock:
        current_time = time.time()
        
        # Prüfe ob Cache gültig ist
        if not force_reload and _perf_settings_cache is not None:
            if current_time - _perf_settings_cache_timestamp < _cache_ttl:
                return _perf_settings_cache.copy()
        
        # Finde Settings-Datei wenn noch nicht gefunden
        if _perf_settings_file_path is None:
            _perf_settings_file_path = _find_performance_settings_file()
        
        # Lade Settings
        settings = DEFAULT_PERFORMANCE_SETTINGS.copy()
        
        if _perf_settings_file_path and os.path.exists(_perf_settings_file_path):
            try:
                with open(_perf_settings_file_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Merge mit Defaults (für neue Optionen)
                    settings.update(loaded_settings)
                    logger.debug(f"[SETTINGS] Performance-Settings geladen von {_perf_settings_file_path}")
            except Exception as e:
                logger.warning(f"[SETTINGS] Fehler beim Laden der Performance-Settings von {_perf_settings_file_path}: {e}")
        else:
            if _perf_settings_file_path:
                logger.debug(f"[SETTINGS] Performance-Settings-Datei nicht gefunden: {_perf_settings_file_path}, verwende Defaults")
            else:
                logger.debug(f"[SETTINGS] Performance-Settings-Datei nicht gefunden, verwende Defaults")
        
        # Update Cache
        _perf_settings_cache = settings.copy()
        _perf_settings_cache_timestamp = current_time
        
        return settings


def invalidate_cache():
    """Invalidiert den Cache (z.B. nach Änderungen)"""
    global _perf_settings_cache, _perf_settings_cache_timestamp
    with _perf_settings_cache_lock:
        _perf_settings_cache = None
        _perf_settings_cache_timestamp = 0
        logger.debug("[SETTINGS] Cache invalidiert")


def get_performance_settings_file_path() -> Optional[str]:
    """Gibt den Pfad zur Performance-Settings-Datei zurück"""
    global _perf_settings_file_path
    if _perf_settings_file_path is None:
        _perf_settings_file_path = _find_performance_settings_file()
    return _perf_settings_file_path
