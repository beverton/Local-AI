# Smart Browser Tab Management

## Übersicht

Das Local AI Startskript verwendet ein intelligentes Browser-Tab-Management-System, das verhindert, dass bei jedem Start neue Tabs geöffnet werden.

## Funktionsweise

### Beim Start (`start_local_ai.bat`)

1. **Erster Start**: Wenn die URLs noch nie geöffnet wurden, werden neue Browser-Tabs geöffnet
2. **Wiederholter Start**: Wenn die URLs bereits in Tabs geöffnet sind, werden diese Tabs refreshed anstatt neue zu öffnen

### Beim Stop (`stop_server.bat`)

- Beendet alle Services (Local AI Server und Model Service)
- Löscht den Tab-Status, sodass beim nächsten Start neue Tabs geöffnet werden

## Technische Details

### Python-Skript: `scripts/open_or_refresh_browser.py`

Das Skript verwaltet eine Status-Datei unter `data/browser_tabs.state`, die alle geöffneten URLs trackt.

**Verwendung:**
```bash
# Einzelne URL öffnen/refreshen
python scripts\open_or_refresh_browser.py "http://127.0.0.1:8001/"

# Mehrere URLs öffnen/refreshen
python scripts\open_or_refresh_browser.py "http://127.0.0.1:8001/" "http://127.0.0.1:8000/static/index.html"

# Tab-Status löschen
python scripts\open_or_refresh_browser.py --clear
```

**Verhalten:**
- `new=0` Parameter: Bei bereits geöffnetem Tab wird versucht, diesen zu refreshen
- `new=2` Parameter: Bei neuem Tab wird ein neuer Tab im Browser geöffnet

### Status-Datei: `data/browser_tabs.state`

Einfache Textdatei mit einer URL pro Zeile:
```
http://127.0.0.1:8001/
http://127.0.0.1:8000/static/index.html
```

## Browser-Kompatibilität

Das System funktioniert mit allen Standard-Browsern:
- Chrome/Edge: Refresht meist den existierenden Tab
- Firefox: Refresht meist den existierenden Tab
- Safari: Öffnet möglicherweise einen neuen Tab (Browser-abhängig)

**Hinweis:** Das exakte Verhalten hängt vom Browser und dessen Einstellungen ab. In den meisten Fällen wird der existierende Tab fokussiert und refreshed.

## Vorteile

✅ Keine Tab-Flut mehr bei mehrmaligem Start  
✅ Automatischer Refresh der Seiten beim Neustart  
✅ Sauberes Cleanup beim Stop  
✅ Funktioniert mit allen Standard-Browsern  

## Manuelle Tab-Verwaltung

Falls Sie den automatischen Tab-Status zurücksetzen möchten:

```bash
# Option 1: Via Python-Skript
python scripts\open_or_refresh_browser.py --clear

# Option 2: Datei manuell löschen
del data\browser_tabs.state
```

## Fehlerbehebung

**Problem:** Browser öffnet trotzdem neue Tabs

**Lösung:** 
1. Prüfen Sie, ob die Status-Datei korrekt erstellt wird: `data\browser_tabs.state`
2. Löschen Sie die Status-Datei und starten Sie neu
3. Einige Browser-Konfigurationen erzwingen immer neue Tabs - das ist normales Browser-Verhalten

**Problem:** Tabs werden nicht refreshed

**Lösung:**
- Das ist browser-abhängiges Verhalten
- Die URLs werden korrekt geöffnet, aber der Browser entscheidet, ob er refresht oder fokussiert
- In den meisten Fällen wird zumindest der Tab fokussiert, was ausreicht

## Integration

Das Smart Browser Tab Management ist automatisch in folgende Skripte integriert:

- ✅ `start_local_ai.bat` - Verwendet Smart Tab Management
- ✅ `stop_server.bat` - Löscht Tab-Status beim Beenden
- ✅ Vollautomatisch, keine Konfiguration nötig




