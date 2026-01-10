# Problem: MCP Tool wird nicht aufgerufen

## Symptome

- ✅ MCP-Server läuft (sichtbar in `logs/mcp_server.log`)
- ✅ Model Service läuft und Modell ist geladen
- ❌ Keine Requests in den Logs, wenn `local:` verwendet wird
- ❌ Cursor ruft das `chat` Tool nicht auf

## Mögliche Ursachen

### 1. Cursor sieht die Tools nicht

**Prüfung:**
- In Cursor: `Ctrl+Shift+P` → `MCP: Show Servers`
- Prüfen Sie, ob `local-ai` Server gelistet ist
- Prüfen Sie, ob das `chat` Tool sichtbar ist

**Lösung:**
- Cursor vollständig neu starten (alle Prozesse beenden)
- MCP-Konfiguration prüfen: `%APPDATA%\Cursor\User\globalStorage\mcp.json`

### 2. Cursor interpretiert Tool-Beschreibung nicht richtig

**Aktuelle Tool-Beschreibung:**
```
"MANDATORY TOOL USAGE: If user message starts with 'local:' or 'chat:' prefix, 
you MUST call this tool immediately. Do NOT generate a response yourself."
```

**Mögliche Lösung:**
- Tool-Beschreibung noch expliziter machen
- Instructions im `initialize` Response verbessern

### 3. MCP-Integration funktioniert nicht richtig

**Prüfung:**
- Cursor MCP-Logs: `View` → `Output` → `MCP` oder `anysphere.cursor-mcp.MCP user-local-ai`
- Sollte zeigen: `[info] MCP Server "local-ai" verbunden`

**Lösung:**
- MCP-Konfiguration prüfen
- Python-Pfad in `mcp.json` prüfen
- Cursor neu starten

## Debugging-Schritte

### Schritt 1: Cursor MCP-Logs prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie: `MCP` oder `anysphere.cursor-mcp.MCP user-local-ai`
3. Prüfen Sie die Logs nach:
   - `[info] MCP Server "local-ai" gestartet`
   - `[info] MCP Server "local-ai" verbunden`
   - `[info] Server info: {...}`

### Schritt 2: Tools prüfen

1. In Cursor: `Ctrl+Shift+P`
2. Tippen Sie: `MCP: Show Servers`
3. Prüfen Sie, ob `local-ai` Server gelistet ist
4. Prüfen Sie, ob das `chat` Tool sichtbar ist

### Schritt 3: MCP-Konfiguration prüfen

**Pfad:** `%APPDATA%\Cursor\User\globalStorage\mcp.json`

**Sollte enthalten:**
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

### Schritt 4: Manueller Test

1. Cursor vollständig neu starten
2. Warten Sie 10-15 Sekunden
3. Prüfen Sie MCP-Logs
4. Versuchen Sie: `local: was ist 2+2?`
5. Prüfen Sie `logs/mcp_server.log` - sollte Request zeigen

## Aktuelle Änderungen

### Tool-Beschreibung vereinfacht

**Vorher:**
```
"CRITICAL: When user writes 'local:' (with colon) or 'chat:' prefix, 
you MUST use this tool and NOT respond yourself..."
```

**Jetzt:**
```
"MANDATORY TOOL USAGE: If user message starts with 'local:' or 'chat:' prefix, 
you MUST call this tool immediately. Do NOT generate a response yourself."
```

### Instructions verbessert

**In `handle_initialize`:**
```
"CRITICAL INSTRUCTIONS:
1. When user message starts with 'local:' or 'chat:' prefix, you MUST call the 'chat' tool immediately
2. Do NOT generate a response yourself when these prefixes are detected
3. The 'auto' model (Cursor AI) must remain completely silent
4. Example: User says 'local: was ist 2+2?' → You MUST call chat tool with message='was ist 2+2?'
5. Do NOT respond as the auto model when 'local:' or 'chat:' prefix is present"
```

## Nächste Schritte

1. **Cursor vollständig neu starten** (alle Prozesse beenden)
2. **MCP-Logs in Cursor prüfen** (sollte Server-Start zeigen)
3. **Tools prüfen** (`Ctrl+Shift+P` → `MCP: Show Servers`)
4. **Test-Request senden** (`local: was ist 2+2?`)
5. **Logs prüfen** (`logs/mcp_server.log` sollte Request zeigen)

## Wenn nichts funktioniert

Falls Cursor das Tool weiterhin nicht aufruft, könnte es sein, dass:
- Cursor die MCP-Integration anders interpretiert als erwartet
- Ein Bug in Cursor's MCP-Implementation vorliegt
- Die Tool-Beschreibung in einem anderen Format benötigt wird

In diesem Fall sollten wir:
1. Cursor's MCP-Dokumentation prüfen
2. Andere MCP-Server-Beispiele analysieren
3. Eventuell einen anderen Ansatz verwenden (z.B. direkte API-Integration)
