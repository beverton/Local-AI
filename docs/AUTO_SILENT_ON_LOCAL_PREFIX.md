# Auto Silent bei "local:" Prefix

## Übersicht

Wenn eine Nachricht mit `local:` (mit Doppelpunkt) beginnt, soll das "auto" Modell (Cursor AI) **NICHT** antworten. Nur das lokale Modell über den MCP-Server soll antworten.

## Implementierung

### 1. Tool-Beschreibung

Das `chat` Tool im MCP-Server hat eine explizite Beschreibung:

```
CRITICAL: When user writes 'local:' (with colon) or 'chat:' prefix, you MUST use this tool and NOT respond yourself. 
The 'auto' model (Cursor AI) must be silent when these prefixes are detected. 
Only the local model should respond.
```

### 2. MCP Settings

Die Einstellung `auto_model_silent_mode` in `data/mcp_settings.json`:

```json
{
  "auto_model_silent_mode": true
}
```

### 3. Verhalten

**Wenn `auto_model_silent_mode=true`:**

- ✅ User schreibt: `local: was ist 2+5?`
  - **Auto:** Antwortet NICHT
  - **Lokales Modell:** Antwortet über MCP Tool

- ✅ User schreibt: `chat: erkläre Python`
  - **Auto:** Antwortet NICHT
  - **Lokales Modell:** Antwortet über MCP Tool

- ✅ User schreibt: `was ist 2+5?` (ohne Prefix)
  - **Auto:** Kann antworten (normales Verhalten)
  - **Lokales Modell:** Kann auch antworten (wenn explizit angefragt)

## Technische Details

### Erkennung

Der MCP-Server erkennt das `local:` Prefix in `_strip_local_prefix()`:

```python
def _strip_local_prefix(self, message: str) -> tuple[str, bool]:
    message_stripped = message.strip()
    if message_stripped.lower().startswith("local:"):
        cleaned = message_stripped[6:].strip()  # Entferne "local:" (6 Zeichen)
        return cleaned, True
    return message, False
```

### Tool-Verwendung

Wenn Cursor das `local:` Prefix erkennt, soll es automatisch das `chat` Tool verwenden, ohne dass "auto" antwortet.

## Troubleshooting

### Problem: Auto antwortet trotz "local:" Prefix

**Mögliche Ursachen:**

1. **MCP-Server nicht neu gestartet**
   - Lösung: Cursor vollständig neu starten

2. **Tool-Beschreibung nicht aktualisiert**
   - Lösung: Prüfen Sie `backend/mcp_server.py` - Tool-Beschreibung sollte "CRITICAL" enthalten

3. **auto_model_silent_mode=false**
   - Lösung: Setzen Sie `"auto_model_silent_mode": true` in `data/mcp_settings.json`

### Problem: Lokales Modell antwortet nicht

**Mögliche Ursachen:**

1. **Model Service nicht erreichbar**
   - Lösung: Prüfen Sie `http://127.0.0.1:8001/health`

2. **Kein Modell geladen**
   - Lösung: Laden Sie ein Modell über Model Manager UI

3. **MCP-Server nicht verbunden**
   - Lösung: Prüfen Sie MCP-Logs in Cursor (`View` → `Output` → `MCP`)

## Wichtige Regel

⚠️ **Bei "local:" (mit Doppelpunkt) Prefix:**
- ❌ Auto antwortet NICHT
- ✅ Nur lokales Modell antwortet über MCP Tool

## Nächste Schritte

1. ✅ Cursor vollständig neu starten
2. ✅ Testen: `local: was ist 2+5?`
3. ✅ Prüfen: Antwortet nur lokales Modell, auto antwortet nicht
