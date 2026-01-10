# MCP Server "No server info found" - Lösung

## Problem

Cursor zeigt den Fehler: `[error] No server info found` im MCP-Log, obwohl der Model Service läuft.

## Lösung

### Schritt 1: Cursor vollständig neu starten

1. **Schließen Sie Cursor vollständig**:
   - Schließen Sie alle Cursor-Fenster
   - Prüfen Sie im Task Manager, ob noch Cursor-Prozesse laufen
   - Beenden Sie alle Cursor-Prozesse falls nötig

2. **Starten Sie Cursor neu**

### Schritt 2: MCP-Konfiguration prüfen

Stellen Sie sicher, dass die MCP-Konfiguration korrekt ist:

**Pfad zur Konfiguration:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**Korrekte Konfiguration:**
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

**WICHTIG:**
- Verwenden Sie **doppelte Backslashes** (`\\`) in Windows-Pfaden
- Der Pfad muss **absolut** sein
- Prüfen Sie, ob Python im PATH ist, sonst verwenden Sie den vollständigen Pfad zu `python.exe`

### Schritt 3: MCP-Server manuell testen

Testen Sie, ob der MCP-Server korrekt startet:

```powershell
cd "G:\04-CODING\Local Ai\backend"
python mcp_server.py
```

Der Server sollte ohne Fehler starten und auf Input warten.

### Schritt 4: MCP-Logs in Cursor prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie im Dropdown: `MCP` oder `anysphere.cursor-mcp.MCP user-local-ai`
3. Prüfen Sie die Logs auf Fehlermeldungen

**Erfolgreiche Verbindung:**
```
[info] MCP Server "local-ai" verbunden
[info] Server info: {"name": "local-ai-mcp-server", "version": "1.0.0"}
```

**Bei Fehlern:**
- Prüfen Sie, ob Python korrekt installiert ist
- Prüfen Sie, ob alle Dependencies installiert sind
- Prüfen Sie, ob der Model Service läuft (`http://127.0.0.1:8001/status`)

### Schritt 5: MCP-Server neu registrieren

Falls das Problem weiterhin besteht:

1. Öffnen Sie Cursor Settings
2. Gehen Sie zu `Features` → `Model Context Protocol`
3. Entfernen Sie den `local-ai` Server
4. Fügen Sie ihn erneut hinzu mit der Konfiguration aus Schritt 2
5. Starten Sie Cursor neu

## Verifizierung

Nach dem Neustart sollten Sie in den MCP-Logs sehen:
- `[info] MCP Server "local-ai" verbunden`
- `[info] Server info: {"name": "local-ai-mcp-server", "version": "1.0.0"}`
- Keine `[error] No server info found` Fehler mehr

## Technische Details

Der MCP-Server gibt die `serverInfo` korrekt zurück:
```python
"serverInfo": {
    "name": "local-ai-mcp-server",
    "version": "1.0.0"
}
```

Das Problem ist meist, dass Cursor die alte Version gecacht hat oder der MCP-Server-Prozess nicht neu gestartet wurde.
