# Lokales Modell in Cursor Modellauswahl integrieren

## Übersicht

Diese Anleitung erklärt, wie Sie das lokale Modell direkt in die Modellauswahl im Cursor Chat-Fenster integrieren, sodass Sie es wie "auto" auswählen können.

## Implementierung

### 1. MCP-Server Capabilities erweitert

Der MCP-Server wurde erweitert, um Chat-Completion-Capabilities zu unterstützen:

```python
"capabilities": {
    "tools": {
        "listChanged": True
    },
    "completion": {
        "completionProvider": {
            "model": "local-ai",
            "name": "Local AI (qwen-2.5-7b-instruct)"
        }
    }
}
```

### 2. Completion-Methode implementiert

Der MCP-Server unterstützt jetzt die `completion/complete` Methode, die von Cursor verwendet wird, wenn ein Modell aus der Auswahl gewählt wird.

## Verwendung

### Schritt 1: Cursor neu starten

**Wichtig:** Nach Änderungen am MCP-Server muss Cursor **vollständig neu gestartet** werden:

1. Schließen Sie alle Cursor-Fenster
2. Starten Sie Cursor neu
3. Warten Sie, bis der MCP-Server verbunden ist

### Schritt 2: Modellauswahl prüfen

1. Öffnen Sie den Cursor Chat (`Ctrl+L` oder Chat-Panel)
2. Klicken Sie auf die Modellauswahl (normalerweise oben im Chat-Fenster)
3. Sie sollten jetzt **"Local AI"** oder **"local-ai"** in der Liste sehen

### Schritt 3: Lokales Modell auswählen

1. Wählen Sie **"Local AI"** aus der Modellauswahl
2. Stellen Sie eine Frage (ohne "local:" Prefix)
3. Das lokale Modell sollte antworten

## Troubleshooting

### Problem: "Local AI" erscheint nicht in der Modellauswahl

**Mögliche Ursachen:**

1. **Cursor nicht neu gestartet**
   - Lösung: Cursor vollständig schließen und neu starten

2. **MCP-Server nicht verbunden**
   - Lösung: Prüfen Sie die MCP-Logs (`View` → `Output` → `MCP`)
   - Stellen Sie sicher, dass der MCP-Server initialisiert wurde

3. **Capabilities nicht erkannt**
   - Lösung: Prüfen Sie, ob die Capabilities korrekt im `handle_initialize` zurückgegeben werden

### Problem: Modell antwortet nicht

**Mögliche Ursachen:**

1. **Kein Modell geladen**
   - Lösung: Laden Sie ein Modell: `http://127.0.0.1:8001/models/text/load`
   - Oder über Chat: "Lade das Modell qwen-2.5-7b-instruct als Text-Modell"

2. **Model Service nicht erreichbar**
   - Lösung: Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`

3. **Completion-Methode nicht unterstützt**
   - Lösung: Prüfen Sie die MCP-Logs auf Fehler bei `completion/complete` Requests

## Alternative: Falls Integration nicht funktioniert

Falls das lokale Modell nicht in der Modellauswahl erscheint, können Sie weiterhin:

1. **"local:" Prefix verwenden:**
   ```
   local: was ist 2+5?
   ```

2. **Explizit Tool aufrufen:**
   ```
   "Verwende das chat Tool um zu fragen: was ist 2+5?"
   ```

3. **Tool direkt ansprechen:**
   ```
   "Frage das lokale Modell: was ist 2+5?"
   ```

## Technische Details

### Completion-Request Format

Cursor sendet Completion-Requests im Format:
```json
{
  "method": "completion/complete",
  "params": {
    "prompt": "User-Frage",
    "messages": [{"role": "user", "content": "User-Frage"}]
  }
}
```

Der MCP-Server konvertiert diese automatisch zu Chat-Requests und delegiert an den Model Service.

### Modell-Name

Der Modell-Name wird dynamisch basierend auf dem geladenen Modell bestimmt:
- Standard: `"Local AI (qwen-2.5-7b-instruct)"`
- Kann angepasst werden, um den tatsächlich geladenen Modell-Namen anzuzeigen

## Nächste Schritte

1. ✅ Cursor neu starten
2. ✅ Modellauswahl prüfen
3. ✅ Lokales Modell auswählen
4. ✅ Testen mit einer Frage
5. ✅ Falls nicht funktioniert: MCP-Logs prüfen

## Hinweise

- Die Integration funktioniert nur, wenn der MCP-Server korrekt konfiguriert ist
- Der Model Service muss laufen und ein Modell muss geladen sein
- Cursor muss neu gestartet werden, damit Änderungen wirksam werden
