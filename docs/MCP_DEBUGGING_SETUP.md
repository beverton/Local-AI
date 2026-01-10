# MCP Server Debugging Setup - Vollständige Übersicht

## Was wurde eingerichtet

### 1. Detailliertes Logging

**MCP-Server (`backend/mcp_server.py`):**
- ✅ Logging in Datei: `logs/mcp_server.log`
- ✅ Logging nach stdout (für Cursor)
- ✅ Detaillierte Logs für:
  - Server-Start
  - Request-Empfang
  - Tool-Aufrufe
  - Model Service Kommunikation
  - Responses

**Model Service (`backend/model_service.py`):**
- ✅ Logging für Chat-Requests: `logs/model_service.log`
- ✅ Detaillierte Logs für:
  - Request-Empfang
  - Profil-Verwendung
  - Generierung
  - Response-Versand

### 2. Log-Dateien

- `logs/mcp_server.log` - Alle MCP-Server Aktivitäten
- `logs/model_service.log` - Model Service Aktivitäten
- `logs/main_server.log` - Haupt-Server Logs

### 3. Test-Script

- `test_mcp_request.py` - Testet MCP-Server manuell

## Wie prüfen Sie, ob es funktioniert

### Schritt 1: Model Service Status prüfen

```powershell
cd "G:\04-CODING\Local Ai"
python -c "import requests; r = requests.get('http://127.0.0.1:8001/models/text/status'); print('Model:', r.json().get('model_id'), '- Loaded:', r.json().get('loaded'))"
```

**Erwartet:** `Model: qwen-2.5-7b-instruct - Loaded: True`

### Schritt 2: MCP-Server Logs prüfen

```powershell
Get-Content logs\mcp_server.log -Tail 20
```

**Erwartet:** Logs zeigen Server-Start und Requests

### Schritt 3: Model Service Logs prüfen

```powershell
Get-Content logs\model_service.log -Tail 20
```

**Erwartet:** Logs zeigen Chat-Requests mit `[CHAT]` Tags

### Schritt 4: Cursor MCP-Konfiguration prüfen

**Pfad:** `%APPDATA%\Cursor\User\globalStorage\mcp.json`

**Korrekte Konfiguration:**
```json
{
  "mcpServers": {
    "local-ai": {
      "command": "C:\\Python313\\python.exe",
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

### Schritt 5: Cursor MCP-Logs prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie: `MCP` oder `anysphere.cursor-mcp.MCP user-local-ai`
3. Prüfen Sie die Logs

**Erfolgreiche Verbindung:**
```
[info] MCP Server "local-ai" gestartet
[info] MCP Server "local-ai" verbunden
[info] Server info: {"name": "local-ai-mcp-server", "version": "1.0.0"}
```

## Troubleshooting

### Problem: Keine Logs in `logs/mcp_server.log`

**Ursache:** Cursor startet den MCP-Server nicht

**Lösung:**
1. Prüfen Sie die MCP-Konfiguration in `%APPDATA%\Cursor\User\globalStorage\mcp.json`
2. Prüfen Sie, ob Python-Pfad korrekt ist
3. Starten Sie Cursor vollständig neu

### Problem: MCP-Server startet, aber keine Requests

**Ursache:** Cursor verwendet das Tool nicht automatisch

**Lösung:**
1. Prüfen Sie die Tool-Beschreibung (sollte "CRITICAL: When user writes 'local:'..." enthalten)
2. Versuchen Sie, das Tool manuell aufzurufen: `Ctrl+Shift+P` → "MCP: Show Servers"
3. Prüfen Sie die MCP-Logs in Cursor

### Problem: Requests kommen an, aber Model Service antwortet nicht

**Ursache:** Model Service ist nicht erreichbar oder Modell nicht geladen

**Lösung:**
1. Prüfen Sie Model Service Status: `http://127.0.0.1:8001/status`
2. Prüfen Sie Model Status: `http://127.0.0.1:8001/models/text/status`
3. Prüfen Sie die Logs: `logs/model_service.log`

### Problem: "No server info found" in Cursor

**Ursache:** MCP-Server gibt keine `serverInfo` zurück

**Lösung:**
1. Prüfen Sie `handle_initialize` in `backend/mcp_server.py`
2. Prüfen Sie die MCP-Logs in Cursor
3. Starten Sie Cursor neu

## Nächste Schritte

1. **Cursor vollständig neu starten** (alle Prozesse beenden)
2. **MCP-Logs in Cursor prüfen** (sollte Server-Start zeigen)
3. **Testen mit "local:" Prefix** (z.B. `local: Was ist 2+2?`)
4. **Logs prüfen:**
   - `logs/mcp_server.log` - Sollte Request zeigen
   - `logs/model_service.log` - Sollte Chat-Request zeigen
5. **Falls nichts ankommt:** MCP-Konfiguration in Cursor prüfen

## Wichtige Log-Tags

- `[MCP]` - MCP-Server Aktivitäten
- `[CHAT]` - Chat-Requests im Model Service
- `[MODEL_LOAD]` - Modell-Ladevorgänge

## Verifizierung

Nach allen Schritten sollten Sie sehen:

1. ✅ MCP-Server Logs zeigen Server-Start
2. ✅ Wenn Sie "local:" verwenden, kommt Request in `logs/mcp_server.log` an
3. ✅ Request wird an Model Service weitergeleitet (sichtbar in `logs/model_service.log`)
4. ✅ Model Service generiert Antwort (sichtbar in `logs/model_service.log`)
5. ✅ Antwort kommt zurück an MCP-Server (sichtbar in `logs/mcp_server.log`)
6. ✅ Antwort wird in Cursor angezeigt
