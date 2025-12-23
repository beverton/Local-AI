# Test-Suite für Local AI

Diese Test-Suite ermöglicht automatisches Testen und Debugging aller Features und Modelle.

## Features

- **Strukturierte Logs**: Alle Tests schreiben strukturierte JSON-Logs
- **Automatisches Debugging**: Tests identifizieren Probleme automatisch
- **Log-Analyse**: Standalone-Tool zum Analysieren von Logs
- **Debug-Reports**: Automatische Generierung von Debug-Reports mit Lösungsvorschlägen

## Voraussetzungen

1. Server muss laufen auf `http://127.0.0.1:8000`
2. Python-Pakete: `requests` (für API-Tests)

```bash
pip install requests
```

## Test-Ausführung

### Alle Tests ausführen

```bash
python tests/run_tests.py
```

### Einzelne Test-Module

```bash
# Unit Tests für Modell-Laden
python -m pytest tests/test_model_loading.py -v

# Integration Tests für API-Endpunkte
python -m pytest tests/test_api_endpoints.py -v

# Workflow-Tests
python -m pytest tests/test_workflows.py -v
```

## Log-Analyse

### Test-Logs analysieren

```bash
# Alle Test-Logs
python tests/analyze_logs.py

# Spezifischer Test
python tests/analyze_logs.py --test test_model_manager_load

# Mit Details
python tests/analyze_logs.py --details
```

### Debug-Logs analysieren

```bash
# Fehler in .cursor/debug.log finden
python tests/analyze_logs.py --debug-logs

# Mit Warnungen
python tests/analyze_logs.py --debug-logs --warnings
```

### Debug-Reports anzeigen

```bash
# Alle Reports
python tests/analyze_logs.py --reports

# Spezifischer Test
python tests/analyze_logs.py --reports --test test_model_manager_load
```

## Test-Struktur

### Log-Dateien

- `.cursor/test_logs.jsonl`: Strukturierte Test-Logs (JSON Lines Format)
- `.cursor/debug.log`: Backend Debug-Logs (wird von Tests analysiert)
- `.cursor/debug_reports/`: Debug-Reports (JSON-Dateien)

### Test-Komponenten

1. **LogManager** (`tests/log_manager.py`): Verwaltet Test-Logs
2. **SystemChecker** (`tests/system_checker.py`): Prüft System-Status
3. **TestRunner** (`tests/test_runner.py`): Führt Tests aus mit Auto-Debugging

### Test-Module

1. **test_model_loading.py**: Unit Tests für Modell-Manager
2. **test_api_endpoints.py**: Integration Tests für API-Endpunkte
3. **test_workflows.py**: End-to-End Workflow-Tests

## Debug-Reports

Debug-Reports enthalten:
- Identifizierte Probleme
- System-State zum Zeitpunkt des Fehlers
- Relevante Log-Einträge
- Lösungsvorschläge

Beispiel:
```json
{
  "test_name": "test_model_manager_load",
  "status": "failed",
  "problems": [
    {
      "type": "model_not_loaded",
      "description": "Model was not loaded after load_model() returned True",
      "location": "state_check"
    }
  ],
  "suggested_fixes": [
    {
      "problem": "model_not_loaded",
      "fix": "Prüfe ob load_model() korrekt aufgerufen wurde und current_model_id gesetzt ist"
    }
  ]
}
```

## Selbstständiges Debugging

Die Tests können:
1. **Logs lesen**: Aus `.cursor/debug.log` und `.cursor/test_logs.jsonl`
2. **System-State prüfen**: API-Endpunkte, Modell-Status
3. **Probleme identifizieren**: Automatische Analyse
4. **Lösungsvorschläge generieren**: Basierend auf Problem-Typ
5. **Reports erstellen**: Strukturierte Debug-Reports

## Workflow-Tests

Die Workflow-Tests simulieren komplette Anwendungsfälle:

1. **Whisper → Chat**: Audio-Transkription gefolgt von Chat
2. **Chat → Image**: Chat gefolgt von Bildgenerierung
3. **Full Workflow**: Whisper → Chat → Image
4. **Simultanes Arbeiten**: Mehrere Modelle gleichzeitig

## Troubleshooting

**Tests schlagen fehl weil Server nicht läuft:**
- Starte Server: `python backend/main.py` oder `start_local_ai.bat`
- Prüfe ob Server auf Port 8000 läuft

**Keine Modelle verfügbar:**
- Prüfe `config.json` - Modelle müssen konfiguriert sein
- Tests werden übersprungen wenn keine Modelle verfügbar

**Logs werden nicht geschrieben:**
- Prüfe ob `.cursor/` Verzeichnis existiert und beschreibbar ist
- Prüfe Dateiberechtigungen








