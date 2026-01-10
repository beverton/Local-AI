# Lösung für SSRF-Block in Cursor

## Problem

Cursor blockiert Verbindungen zu privaten IPs (127.0.0.1) aus Sicherheitsgründen:
```
{"error":{"type":"client","reason":"ssrf_blocked","message":"connection to private IP is blocked"}}
```

## Lösungen

### Option 1: localhost statt 127.0.0.1 verwenden

Manchmal akzeptiert Cursor `localhost` statt `127.0.0.1`:

1. **In Cursor Settings:**
   - "Override OpenAI Base URL": `http://localhost:8001/v1`
   - Toggle aktivieren

2. **Testen Sie, ob das funktioniert**

### Option 2: Cursor Settings - SSRF Whitelist

Prüfen Sie, ob Cursor eine Whitelist für lokale Endpoints hat:

1. **In Cursor Settings suchen nach:**
   - "SSRF"
   - "Private IP"
   - "Localhost"
   - "Allowed IPs"

2. **Falls vorhanden:**
   - Fügen Sie `127.0.0.1` oder `localhost` zur Whitelist hinzu

### Option 3: Environment Variable

Manchmal können Sie SSRF-Schutz über Umgebungsvariablen umgehen:

```bash
CURSOR_ALLOW_LOCALHOST=true
```

### Option 4: Proxy verwenden

Erstellen Sie einen lokalen Proxy, der auf einer öffentlichen Domain läuft:

1. **ngrok verwenden** (für Testing):
   ```bash
   ngrok http 8001
   ```
   - Nutzen Sie die ngrok-URL in Cursor

2. **Oder lokaler Reverse Proxy** mit einer Domain

### Option 5: Cursor Rules / Config

Prüfen Sie, ob es eine `.cursorrules` oder Config-Datei gibt, wo Sie Ausnahmen definieren können.

## Empfohlene Lösung

**Versuchen Sie zuerst Option 1** (`localhost` statt `127.0.0.1`):

1. In Cursor Settings:
   - "Override OpenAI Base URL": `http://localhost:8001/v1`
   - Toggle aktivieren
   - "OpenAI API Key": `local` (oder einen beliebigen Wert)

2. Testen Sie mit einer Frage im Chat

Falls das nicht funktioniert, müssen wir eine andere Lösung finden.
