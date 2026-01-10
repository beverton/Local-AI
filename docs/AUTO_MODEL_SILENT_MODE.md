# Auto Model Silent Mode

## Übersicht

Der "Auto Model Silent Mode" verhindert, dass das "auto" Modell (Cursor AI) antwortet, wenn Nachrichten mit `local:` oder `chat:` beginnen. In diesem Fall antwortet nur das lokale Modell über den MCP-Server.

## Konfiguration

Die Einstellung wird in `data/mcp_settings.json` gespeichert:

```json
{
  "auto_model_silent_mode": true
}
```

### Aktivieren/Deaktivieren

**Aktivieren (Standard):**
```json
{
  "auto_model_silent_mode": true
}
```

**Deaktivieren:**
```json
{
  "auto_model_silent_mode": false
}
```

## Funktionsweise

Wenn `auto_model_silent_mode` auf `true` gesetzt ist:

1. **Mit Präfix (`local:` oder `chat:`):**
   - Das "auto" Modell erkennt das Präfix und antwortet nicht
   - Nur das lokale Modell über den MCP-Server antwortet
   - Beispiel: `local: was ist 3+5?` → Nur lokales Modell antwortet

2. **Ohne Präfix:**
   - Normales Verhalten: Beide Modelle können antworten
   - Beispiel: `was ist 3+5?` → Normales Verhalten

## Verwendung

### Aktivieren
1. Öffne `data/mcp_settings.json`
2. Setze `"auto_model_silent_mode": true`
3. Speichere die Datei

### Deaktivieren
1. Öffne `data/mcp_settings.json`
2. Setze `"auto_model_silent_mode": false`
3. Speichere die Datei

## Hinweise

- Die Einstellung wird sofort wirksam (kein Neustart nötig)
- Wenn das lokale Modell nicht verfügbar ist, kann das "auto" Modell trotzdem antworten (Fallback)
- Die Präfixe sind case-insensitive: `local:`, `Local:`, `LOCAL:` werden alle erkannt
