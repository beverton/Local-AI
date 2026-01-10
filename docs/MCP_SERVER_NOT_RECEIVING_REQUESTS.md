# MCP Server empfängt keine Requests - Lösung

## Problem

Wenn Sie "local:" verwenden, kommt nichts beim Server an. Das bedeutet, dass Cursor den MCP-Server-Prozess nicht startet oder nicht erreicht.

## Lösung

### Schritt 1: MCP-Konfiguration in Cursor prüfen

**WICHTIG:** Die MCP-Konfiguration muss in Cursor's globalStorage sein, nicht nur in `config/cursor_mcp_config.json`!

**Pfad zur Cursor-Konfiguration:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**Vollständiger Pfad (normalerweise):**
```
C:\Users\IHR_USERNAME\AppData\Roaming\Cursor\User\globalStorage\mcp.json
```

**So finden Sie den Pfad:**
1. Drücken Sie `Win + R`
2. Geben Sie ein: `%APPDATA%\Cursor\User\globalStorage`
3. Drücken Sie Enter
4. Öffnen Sie `mcp.json` (oder erstellen Sie sie, falls sie nicht existiert)

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

### Schritt 2: MCP-Server manuell testen

Testen Sie, ob der MCP-Server korrekt startet:

```powershell
cd "G:\04-CODING\Local Ai\backend"
python mcp_server.py
```

Der Server sollte ohne Fehler starten und auf Input warten (stdin). Sie können dann eine JSON-RPC Request senden:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"workspaceRoot": "."}}
```

Der Server sollte eine Response zurückgeben.

### Schritt 3: Cursor vollständig neu starten

**Nach Änderungen an der MCP-Konfiguration:**

1. **Schließen Sie Cursor vollständig:**
   - Schließen Sie alle Cursor-Fenster
   - Öffnen Sie den Task Manager (`Ctrl+Shift+Esc`)
   - Beenden Sie alle `Cursor.exe` Prozesse

2. **Starten Sie Cursor neu**

3. **Warten Sie 10-15 Sekunden**, damit Cursor den MCP-Server starten kann

### Schritt 4: MCP-Logs in Cursor prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie im Dropdown: `MCP` oder `anysphere.cursor-mcp.MCP user-local-ai`
3. Prüfen Sie die Logs

**Erfolgreiche Verbindung:**
```
[info] MCP Server "local-ai" gestartet
[info] MCP Server "local-ai" verbunden
[info] Server info: {"name": "local-ai-mcp-server", "version": "1.0.0"}
```

**Bei Fehlern:**
- `[error] Failed to start MCP server` → Prüfen Sie die Python-Pfade
- `[error] No server info found` → Server startet, aber gibt keine serverInfo zurück
- Keine Logs → Cursor startet den Server nicht (Konfiguration prüfen)

### Schritt 5: Python-Pfad prüfen

Falls Python nicht im PATH ist, verwenden Sie den vollständigen Pfad:

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

**So finden Sie den Python-Pfad:**
```powershell
where python
```

### Schritt 6: Dependencies prüfen

Stellen Sie sicher, dass alle Dependencies installiert sind:

```powershell
cd "G:\04-CODING\Local Ai"
pip install -r requirements.txt
```

### Schritt 7: Model Service Status prüfen

Der MCP-Server benötigt den Model Service. Prüfen Sie, ob er läuft:

```powershell
curl http://127.0.0.1:8001/status
```

Falls nicht, starten Sie ihn:
```powershell
python backend/model_service.py
```

## Troubleshooting

### Problem: Keine Logs in Cursor

**Mögliche Ursachen:**
1. MCP-Konfiguration nicht in `%APPDATA%\Cursor\User\globalStorage\mcp.json`
2. Cursor wurde nicht neu gestartet
3. Python-Pfad ist falsch

**Lösung:**
- Prüfen Sie die Konfiguration in `%APPDATA%\Cursor\User\globalStorage\mcp.json`
- Starten Sie Cursor vollständig neu
- Prüfen Sie den Python-Pfad

### Problem: Server startet, aber keine Requests

**Mögliche Ursachen:**
1. Cursor verwendet das Tool nicht automatisch
2. Tool-Beschreibung wird nicht korrekt interpretiert

**Lösung:**
- Prüfen Sie, ob das Tool in Cursor verfügbar ist: `Ctrl+Shift+P` → "MCP: Show Servers"
- Versuchen Sie, das Tool manuell aufzurufen

### Problem: "No server info found"

**Lösung:**
- Prüfen Sie, ob `handle_initialize` die `serverInfo` zurückgibt
- Prüfen Sie die MCP-Logs auf Fehler

## Verifizierung

Nach allen Schritten sollten Sie sehen:
1. ✅ MCP-Logs zeigen: `[info] MCP Server "local-ai" verbunden`
2. ✅ Server info wird angezeigt
3. ✅ Wenn Sie "local:" verwenden, kommt eine Request beim Server an
4. ✅ Das lokale Modell antwortet
