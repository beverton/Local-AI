# Strukturiertes Logging-System

## Übersicht

Ein zentrales, strukturiertes Logging-System wurde implementiert, um Fehler schneller zu finden und zu beheben.

## Features

### 1. Strukturierte Logs mit Tags
- `[MODEL_LOAD]` - Modell-Laden
- `[MODEL_GEN]` - Modell-Generierung
- `[CHAT]` - Chat-Requests
- `[QUALITY]` - Quality Management
- `[WEB_SEARCH]` - Web-Suche
- `[VALIDATION]` - Response-Validierung
- `[ERROR]` - Fehler
- `[API]` - API-Calls
- `[MCP]` - MCP Server
- `[CONFIG]` - Konfiguration
- `[PERF]` - Performance

### 2. Log-Dateien
- `logs/model_manager.log` - Model Manager Logs
- `logs/model_service.log` - Model Service Logs
- `logs/main_server.log` - Main Server Logs
- `logs/quality_manager.log` - Quality Manager Logs
- `logs/local_ai.log` - Globale Logs

### 3. Log-Format
```
TIMESTAMP | LEVEL | [TAG] MESSAGE
```

Beispiel:
```
2026-01-09 19:48:47 | INFO | [MODEL_LOAD] Starte Laden: model_id=qwen-2.5-7b-instruct
```

## Verwendung

### In Python-Code:
```python
from logging_utils import get_logger

logger = get_logger(__name__)

# Spezielle Methoden für verschiedene Bereiche
logger.model_load("Modell wird geladen...")
logger.model_gen("Generierung startet...")
logger.chat("Chat-Request erhalten")
logger.error_log("Fehler aufgetreten", exc_info=True)
```

## Vorteile

1. **Schnelle Fehlerlokalisierung**: Tags erlauben einfaches Filtern
2. **Vollständige Tracebacks**: Alle Exceptions werden mit Stack-Traces geloggt
3. **Strukturierte Ausgabe**: Konsistentes Format für alle Logs
4. **Datei-Logging**: Logs werden in Dateien gespeichert für spätere Analyse
5. **Performance-Tracking**: Zeitstempel für alle wichtigen Operationen

## Nächste Schritte

1. ✅ Logging-System implementiert
2. ✅ Alle kritischen Bereiche ausgestattet
3. ⏳ CUDA OOM Problem beheben (device_map="auto" mit max_memory)
4. ⏳ Weitere Bereiche mit Logging ausstatten (Web Search, Validation, etc.)
