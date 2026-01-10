# Cursor Lokales Modell - Fehlerbehebung

## Problem: Lokales Modell reagiert nicht

Wenn Sie "Local" in der Modellauswahl wählen oder "local:" schreiben, aber das lokale Modell nicht antwortet, folgen Sie diesen Schritten:

## Schritt 1: Prüfen Sie, ob der Model Service läuft

**Terminal/Command Prompt:**
```bash
curl http://127.0.0.1:8001/health
```

**Erwartete Antwort:**
```json
{"status":"healthy",...}
```

**Falls nicht erreichbar:**
- Starten Sie den Model Service: `scripts/start_model_service.bat`
- Oder: `python backend/model_service.py`

## Schritt 2: Prüfen Sie, ob ein Modell geladen ist

**Terminal/Command Prompt:**
```bash
curl http://127.0.0.1:8001/models/text/status
```

**Erwartete Antwort:**
```json
{"loaded":true,"model_id":"qwen-2.5-7b-instruct",...}
```

**Falls kein Modell geladen:**
- Laden Sie ein Modell über die Model Manager UI: `http://127.0.0.1:8001`
- Oder über API: `POST http://127.0.0.1:8001/models/text/load` mit Body `{"model_id": "qwen-2.5-7b-instruct"}`

## Schritt 3: Prüfen Sie die MCP-Integration

### 3.1 MCP-Server Status prüfen

1. **In Cursor:**
   - `View` → `Output` → Wählen Sie "MCP" aus der Dropdown-Liste
   - Prüfen Sie, ob es Fehler gibt

2. **MCP-Config prüfen:**
   - Datei: `config/cursor_mcp_config.json`
   - Sollte enthalten:
   ```json
   {
     "mcpServers": {
       "local-ai": {
         "command": "python",
         "args": ["G:\\04-CODING\\Local Ai\\backend\\mcp_server.py"],
         "env": {
           "PYTHONPATH": "G:\\04-CODING\\Local Ai\\backend"
         }
       }
     }
   }
   ```

### 3.2 Cursor neu starten

**Wichtig:** Nach Änderungen an der MCP-Config muss Cursor **vollständig neu gestartet** werden:

1. Schließen Sie **alle** Cursor-Fenster
2. Warten Sie 5 Sekunden
3. Starten Sie Cursor neu
4. Warten Sie, bis der MCP-Server verbunden ist (prüfen Sie die MCP-Logs)

## Schritt 4: Testen Sie die MCP-Integration

### Option A: "local:" Prefix verwenden

**Im Cursor Chat:**
```
local: was ist 2+5?
```

**Erwartetes Verhalten:**
- Das lokale Modell sollte antworten
- "Auto" sollte **nicht** antworten (wenn `auto_model_silent_mode=true`)

### Option B: MCP Tool direkt verwenden

**Im Cursor Chat:**
```
Verwende das chat Tool um zu fragen: was ist 2+5?
```

**Oder:**
```
Frage das lokale Modell: was ist 2+5?
```

## Schritt 5: Prüfen Sie die Modellauswahl

### 5.1 Lokales Modell in der Liste

1. Öffnen Sie den Cursor Chat (`Ctrl+L`)
2. Klicken Sie auf die Modellauswahl (oben im Chat)
3. Prüfen Sie, ob **"Local AI"** oder **"local-ai"** in der Liste erscheint

**Falls nicht:**
- Cursor neu starten (Schritt 3.2)
- MCP-Logs prüfen auf Fehler
- MCP-Config prüfen (Schritt 3.1)

### 5.2 Lokales Modell auswählen

1. Wählen Sie **"Local AI"** aus der Modellauswahl
2. Stellen Sie eine Frage (ohne "local:" Prefix)
3. Das lokale Modell sollte antworten

**Falls Fehler:**
- Prüfen Sie die MCP-Logs
- Prüfen Sie, ob der Model Service läuft (Schritt 1)
- Prüfen Sie, ob ein Modell geladen ist (Schritt 2)

## Schritt 6: SSRF-Block umgehen (falls HTTP-Integration gewünscht)

**Problem:** Cursor blockiert HTTP-Verbindungen zu `127.0.0.1` oder `localhost`

**Lösung 1: MCP-Integration nutzen (EMPFOHLEN)**
- MCP läuft über stdio, keine HTTP-Verbindung nötig
- Keine SSRF-Blockierung
- Funktioniert direkt nach Cursor-Neustart

**Lösung 2: localhost statt 127.0.0.1**
- In Cursor Settings:
  - "Override OpenAI Base URL": `http://localhost:8001/v1`
  - Toggle aktivieren
  - "OpenAI API Key": `local` (oder leer lassen)

**Lösung 3: ngrok Proxy (für Testing)**
```bash
ngrok http 8001
```
- Nutzen Sie die ngrok-URL in Cursor Settings

## Häufige Fehler

### Fehler: "doesn't work with your API key"

**Ursache:** Cursor erwartet einen API Key, auch wenn er nicht benötigt wird

**Lösung:**
1. In Cursor Settings:
   - "OpenAI API Key": `local` (oder einen beliebigen Wert)
   - "Override OpenAI Base URL": `http://localhost:8001/v1`
   - Toggle aktivieren

2. **ODER:** Nutzen Sie die MCP-Integration (kein API Key nötig)

### Fehler: "connection to private IP is blocked"

**Ursache:** Cursor blockiert private IPs (SSRF-Schutz)

**Lösung:**
- Nutzen Sie die **MCP-Integration** (läuft über stdio, keine HTTP-Verbindung)
- Oder: Versuchen Sie `localhost` statt `127.0.0.1`

### Fehler: "No model loaded"

**Ursache:** Kein Modell im Model Service geladen

**Lösung:**
- Laden Sie ein Modell: `http://127.0.0.1:8001/models/text/load` mit Body `{"model_id": "qwen-2.5-7b-instruct"}`
- Oder über Model Manager UI: `http://127.0.0.1:8001`

### Fehler: "Local AI" erscheint nicht in der Modellauswahl

**Ursache:** MCP-Server nicht verbunden oder nicht initialisiert

**Lösung:**
1. Cursor vollständig neu starten (Schritt 3.2)
2. MCP-Logs prüfen (`View` → `Output` → `MCP`)
3. MCP-Config prüfen (Schritt 3.1)

## Empfohlener Workflow

**Für beste Ergebnisse:**

1. **Nutzen Sie die MCP-Integration:**
   - Keine HTTP-Verbindung nötig
   - Keine SSRF-Blockierung
   - Funktioniert direkt nach Cursor-Neustart

2. **Verwenden Sie "local:" Prefix:**
   ```
   local: was ist 2+5?
   ```
   - Explizit, klar, zuverlässig
   - "Auto" antwortet nicht (wenn `auto_model_silent_mode=true`)

3. **Oder wählen Sie "Local AI" in der Modellauswahl:**
   - Funktioniert, wenn MCP-Server verbunden ist
   - Kein Prefix nötig

## Nächste Schritte

1. ✅ Model Service läuft (Schritt 1)
2. ✅ Modell geladen (Schritt 2)
3. ✅ MCP-Integration aktiv (Schritt 3)
4. ✅ Cursor neu gestartet (Schritt 3.2)
5. ✅ Test mit "local:" Prefix (Schritt 4)
6. ✅ Test mit Modellauswahl (Schritt 5)

Falls es immer noch nicht funktioniert, prüfen Sie die MCP-Logs auf detaillierte Fehlermeldungen.
