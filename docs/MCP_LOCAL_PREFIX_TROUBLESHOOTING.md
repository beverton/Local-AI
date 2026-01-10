# Troubleshooting: "local:" Prefix funktioniert nicht

## Problem: Keine Reaktion vom lokalen Modell bei "local:"

Wenn Sie `local: was ist 2+5?` schreiben und keine Antwort vom lokalen Modell kommt, gibt es mehrere mögliche Ursachen:

## Schritt-für-Schritt Diagnose

### 1. Prüfen Sie, ob das Modell geladen ist

**Im Browser:**
```
http://127.0.0.1:8001/models/text/status
```

**Oder über Chat:**
```
"Zeige mir den Status des Text-Modells"
```

**Erwartetes Ergebnis:**
```json
{
  "loaded": true,
  "model_id": "qwen-2.5-7b-instruct"
}
```

### 2. Prüfen Sie, ob der MCP-Server läuft

**In Cursor:**
1. Öffnen Sie `View` → `Output`
2. Wählen Sie `MCP` oder `@anysphere.cursor-mcp.MCP`
3. Suchen Sie nach Fehlermeldungen

**Erwartetes Ergebnis:**
- Keine Fehlermeldungen
- MCP-Server sollte "initialized" sein

### 3. Prüfen Sie die MCP-Konfiguration

**Datei:** `c:\Users\schic\.cursor\mcp.json`

**Sollte enthalten:**
```json
{
  "mcpServers": {
    "local-ai": {
      "command": "python",
      "args": [
        "G:\\04-CODING\\Local Ai\\backend\\mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "G:\\04-CODING\\Local Ai\\backend"
      }
    }
  }
}
```

### 4. Cursor neu starten

**Wichtig:** Nach Änderungen an der MCP-Konfiguration muss Cursor **vollständig neu gestartet** werden.

1. Schließen Sie alle Cursor-Fenster
2. Starten Sie Cursor neu
3. Warten Sie, bis der MCP-Server verbunden ist

### 5. Explizit Tool aufrufen

Wenn "local:" nicht automatisch funktioniert, können Sie das Tool explizit aufrufen:

**Methode 1: Explizit Tool nennen**
```
"Verwende das chat Tool um zu fragen: was ist 2+5?"
```

**Methode 2: Direkt Tool ansprechen**
```
"Frage das lokale Modell: was ist 2+5?"
```

**Methode 3: Über MCP Tool direkt**
```
MCP: Call Tool → chat → {"message": "was ist 2+5?"}
```

## Häufige Probleme und Lösungen

### Problem: "MCP Server nicht verbunden"

**Lösung:**
1. Prüfen Sie, ob Python im PATH ist
2. Prüfen Sie, ob der Pfad zu `mcp_server.py` korrekt ist
3. Starten Sie Cursor neu

### Problem: "Tool wird nicht erkannt"

**Lösung:**
1. Prüfen Sie die MCP-Logs auf Fehler
2. Stellen Sie sicher, dass der MCP-Server initialisiert wurde
3. Versuchen Sie, das Tool explizit aufzurufen

### Problem: "Modell antwortet nicht"

**Lösung:**
1. Prüfen Sie, ob ein Modell geladen ist
2. Prüfen Sie die Model Service Logs: `logs/model_service.log`
3. Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`

### Problem: "Auto-Modell antwortet statt lokales Modell"

**Lösung:**
1. Prüfen Sie, ob `auto_model_silent_mode` aktiviert ist
2. Verwenden Sie explizit das Tool: "Verwende das chat Tool..."
3. Starten Sie Cursor neu

## Test-Script

Sie können auch direkt testen, ob das lokale Modell funktioniert:

```python
import requests

# Test Model Service
response = requests.post(
    'http://127.0.0.1:8001/chat',
    json={
        'message': 'was ist 2+5',
        'messages': [{'role': 'user', 'content': 'was ist 2+5'}],
        'max_length': 256,
        'temperature': 0.3
    },
    timeout=60
)

print('Status:', response.status_code)
print('Antwort:', response.json().get('response', 'KEINE ANTWORT'))
```

## Nächste Schritte

1. ✅ Prüfen Sie, ob Modell geladen ist
2. ✅ Prüfen Sie MCP-Logs auf Fehler
3. ✅ Starten Sie Cursor neu
4. ✅ Versuchen Sie explizit Tool aufzurufen
5. ✅ Prüfen Sie Model Service Logs

Wenn nichts funktioniert, prüfen Sie die Logs:
- `logs/model_service.log` - Model Service Logs
- Cursor Output → MCP - MCP Server Logs
