# "local:" Prefix - Automatische Erkennung für lokales Modell

## Übersicht

Wenn Sie in Cursor eine Nachricht mit `local:` beginnen, wird automatisch das lokale Modell verwendet. Das Prefix wird erkannt, entfernt und die Nachricht an Ihr lokales Modell weitergeleitet.

## Verwendung

### Im Cursor Chat

Einfach `local:` vor Ihre Frage schreiben:

```
local: Was ist 2+5?
```

```
local: Erkläre mir Python Decorators
```

```
local: Erstelle eine Funktion die Fibonacci-Zahlen berechnet
```

### Funktionsweise

1. **Erkennung**: Der MCP-Server erkennt automatisch, wenn eine Nachricht mit `local:` beginnt
2. **Bereinigung**: Das `local:` Prefix wird entfernt (6 Zeichen)
3. **Weiterleitung**: Die bereinigte Nachricht wird an das lokale Modell weitergeleitet
4. **Logging**: Die Erkennung wird in den Logs vermerkt

### Beispiel

**Eingabe:**
```
local: Was ist Machine Learning?
```

**Verarbeitung:**
1. `local:` wird erkannt
2. Nachricht wird zu `Was ist Machine Learning?` bereinigt
3. Wird an lokales Modell gesendet
4. Antwort kommt vom lokalen Modell

### Vorteile

- ✅ **Einfach**: Nur `local:` vor die Frage schreiben
- ✅ **Explizit**: Klar erkennbar, dass lokales Modell verwendet wird
- ✅ **Automatisch**: Keine manuelle Tool-Auswahl nötig
- ✅ **Flexibel**: Funktioniert im Chat und über MCP Tools

### Alternative Methoden

Falls Sie `local:` nicht verwenden möchten:

1. **Explizit Tool aufrufen:**
   ```
   "Frage das lokale Modell: Was ist 2+5?"
   ```

2. **MCP Tool direkt:**
   ```
   MCP: Call Tool → chat → {"message": "Was ist 2+5?"}
   ```

### Technische Details

- **Case-insensitive**: `local:`, `Local:`, `LOCAL:` funktionieren alle
- **Whitespace-tolerant**: `local: ` (mit Leerzeichen) funktioniert auch
- **Logging**: Erkennung wird in MCP-Logs vermerkt
- **Rückwärtskompatibel**: Funktioniert auch ohne Prefix

## Troubleshooting

### "local:" wird nicht erkannt

- Prüfen Sie, ob der MCP-Server läuft
- Prüfen Sie die MCP-Logs: `View` → `Output` → `MCP`
- Stellen Sie sicher, dass die Nachricht mit `local:` beginnt (keine Leerzeichen davor)

### Lokales Modell antwortet nicht

- Prüfen Sie, ob ein Modell geladen ist: `"Zeige mir den Status des Text-Modells"`
- Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`
- Laden Sie ein Modell: `"Lade das Modell qwen-2.5-7b-instruct als Text-Modell"`
