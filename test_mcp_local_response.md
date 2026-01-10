# Test: "local:" Prefix in Cursor

## Was sollte passieren?

Wenn Sie in Cursor schreiben:
```
local: was ist 2+5?
```

Sollte das **lokale Modell** antworten, nicht der externe AI-Assistent.

## Wie prüfen Sie, ob es funktioniert hat?

### 1. Antwort-Quelle prüfen

**Lokales Modell antwortet:**
- Antwort kommt vom lokalen Modell (qwen-2.5-7b-instruct)
- Antwort kann etwas anders formuliert sein als von externen Modellen
- Antwort kann etwas länger dauern (lokale Generierung)

**Externes Modell antwortet:**
- Antwort kommt sofort
- Antwort ist sehr präzise und formatiert
- Keine Verzögerung

### 2. MCP-Logs prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie: `MCP` oder `@anysphere.cursor-mcp.MCP`
3. Suchen Sie nach: `[MCP] 'local:' Prefix erkannt`

**Erfolgreiche Erkennung:**
```
INFO:mcp_server:[MCP] 'local:' Prefix erkannt - verwende lokales Modell für: was ist 2+5?...
```

### 3. Model Service Logs prüfen

Prüfen Sie die Model Service Logs:
```
logs/model_service.log
```

Suchen Sie nach Chat-Requests mit Ihrer Frage.

### 4. Direkter Test

Testen Sie direkt am Model Service:
```bash
python test_local_prefix.py
```

## Troubleshooting

### Problem: Externes Modell antwortet trotz "local:"

**Mögliche Ursachen:**
1. MCP-Server nicht neu geladen → **Cursor neu starten**
2. "local:" wird nicht erkannt → Prüfen Sie MCP-Logs
3. Model Service nicht erreichbar → Prüfen Sie Port 8001

### Problem: Keine Antwort

**Mögliche Ursachen:**
1. Kein Modell geladen → Laden Sie ein Modell
2. Model Service nicht erreichbar → Starten Sie Model Service
3. Timeout → Modell braucht zu lange

### Problem: Nur Sonderzeichen als Antwort

**Mögliche Ursachen:**
1. Modell-Parameter nicht optimal
2. Modell noch nicht vollständig geladen
3. GPU-Speicher-Probleme

## Nächste Schritte

1. ✅ Prüfen Sie MCP-Logs auf "local:" Erkennung
2. ✅ Prüfen Sie, ob lokales Modell antwortet
3. ✅ Testen Sie mit verschiedenen Fragen
