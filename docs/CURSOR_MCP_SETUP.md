# Cursor MCP Setup - Lokale AI-Modelle in Cursor nutzen

Diese Anleitung erklärt, wie Sie den Local AI Model Service als MCP-Server für Cursor konfigurieren.

## Voraussetzungen

1. **Local AI läuft**: Model Service muss auf Port 8001 laufen
2. **Python verfügbar**: Python muss im PATH verfügbar sein
3. **Cursor installiert**: Cursor Editor muss installiert sein

## Schritt 1: Model Service starten

Stellen Sie sicher, dass der Model Service läuft:

```bash
# Im Projekt-Verzeichnis
python backend/model_service.py
```

Oder verwenden Sie das Start-Script:
```bash
start_local_ai.bat
```

Der Model Service sollte auf `http://127.0.0.1:8001` erreichbar sein.

## Schritt 2: Cursor konfigurieren

### Option A: Über Cursor Settings UI

1. Öffnen Sie Cursor
2. Gehen Sie zu **Settings** → **Features** → **Model Context Protocol**
3. Klicken Sie auf **Add Server**
4. Geben Sie folgende Konfiguration ein:

**Name**: `local-ai`

**Command**: `python`

**Args**: 
```
G:\04-CODING\Local Ai\backend\mcp_server.py
```

**Environment Variables**:
```
PYTHONPATH=G:\04-CODING\Local Ai\backend
```

### Option B: Über Konfigurationsdatei

1. Öffnen Sie die Cursor-Konfigurationsdatei:
   - Windows: `%APPDATA%\Cursor\User\globalStorage\mcp.json`
   - Mac: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`
   - Linux: `~/.config/Cursor/User/globalStorage/mcp.json`

2. Fügen Sie die Konfiguration aus `config/cursor_mcp_config.json` hinzu

**Wichtig**: Passen Sie die Pfade in der Konfiguration an Ihr System an!

## Schritt 3: Pfade anpassen

In `config/cursor_mcp_config.json` müssen Sie die Pfade anpassen:

```json
{
  "mcpServers": {
    "local-ai": {
      "command": "python",
      "args": [
        "IHR_PROJEKT_PFAD\\backend\\mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "IHR_PROJEKT_PFAD\\backend"
      }
    }
  }
}
```

Ersetzen Sie `IHR_PROJEKT_PFAD` mit dem tatsächlichen Pfad zu Ihrem Local AI Projekt.

## Schritt 4: Cursor neu starten

Nach der Konfiguration starten Sie Cursor neu, damit die MCP-Server-Konfiguration geladen wird.

## Schritt 5: Testen

1. Öffnen Sie Cursor
2. Öffnen Sie die Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
3. Suchen Sie nach "MCP" oder "Model Context Protocol"
4. Sie sollten den "local-ai" Server sehen

### Verfügbare Tools

Der MCP-Server stellt folgende Tools zur Verfügung:

**Datei-Operationen:**
- **web_search**: Führt Websuchen durch
- **read_file**: Liest Dateien
- **write_file**: Schreibt Dateien
- **list_directory**: Listet Verzeichnisse auf
- **delete_file**: Löscht Dateien
- **file_exists**: Prüft Datei-Existenz

**Model Service Integration:**
- **list_models**: Listet alle verfügbaren Modelle (text, image, audio)
- **load_model**: Lädt ein Modell im Model Service
- **unload_model**: Entlädt ein Modell im Model Service
- **model_status**: Gibt Status des aktuell geladenen Modells zurück
- **chat**: Chat mit lokalem Modell über Model Service

### Chat mit lokalem Modell

Cursor kann jetzt mit Ihren lokalen Modellen kommunizieren. Die Modelle werden über den Model Service (Port 8001) abgerufen.

## Fehlerbehebung

### "Model Service nicht verfügbar"

- Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`
- Prüfen Sie die Firewall-Einstellungen
- Stellen Sie sicher, dass Port 8001 nicht blockiert ist

### "Python nicht gefunden"

- Stellen Sie sicher, dass Python im PATH ist
- Verwenden Sie den vollständigen Pfad zu Python in der Konfiguration

### "Module nicht gefunden"

- Stellen Sie sicher, dass alle Dependencies installiert sind: `pip install -r requirements-base.txt`
- Prüfen Sie, ob PYTHONPATH korrekt gesetzt ist

### "MCP Server startet nicht"

- Prüfen Sie die Logs in Cursor (View → Output → MCP)
- Testen Sie den Server manuell:
  ```bash
  cd backend
  python mcp_server.py
  ```
- Prüfen Sie, ob alle Imports funktionieren

## Manueller Test des MCP-Servers

Sie können den MCP-Server manuell testen:

```bash
cd backend
python mcp_server.py
```

Dann senden Sie eine JSON-RPC Request:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"workspaceRoot": "."}}
```

Der Server sollte eine Response zurückgeben.

## Erweiterte Konfiguration

### Andere Model Service Ports

Falls der Model Service auf einem anderen Port läuft, können Sie dies in `mcp_server.py` anpassen:

```python
server = MCPServer(model_service_host="127.0.0.1", model_service_port=8002)
```

### Workspace Root

Der Workspace Root wird beim `initialize` Request übergeben. Cursor sendet automatisch den aktuellen Workspace-Pfad.

## Support

Bei Problemen:
1. Prüfen Sie die Cursor-Logs (View → Output → MCP)
2. Prüfen Sie die Model Service-Logs
3. Testen Sie den MCP-Server manuell
4. Stellen Sie sicher, dass alle Dependencies installiert sind









