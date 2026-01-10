# MCP-Tools in Cursor nutzen - Praktische Anleitung

## Wichtige Info: MCP-Tools sind direkt im Chat verfügbar!

In Cursor werden MCP-Tools **automatisch** verfügbar gemacht, sobald der MCP-Server verbunden ist. Sie müssen **nicht** über `Ctrl+Shift+P` aufgerufen werden.

---

## Wie Sie MCP-Tools in Cursor verwenden

### Methode 1: Direkt im Chat (Empfohlen)

1. **Öffnen Sie den Cursor Chat** (normalerweise `Ctrl+L` oder Chat-Panel)
2. **Stellen Sie eine Frage**, die ein Tool benötigt
3. **Cursor verwendet automatisch** die passenden MCP-Tools

**Beispiel:**
```
Sie: "Lies die Datei README.md"
→ Cursor verwendet automatisch das `read_file` Tool
```

### Methode 2: Explizit Tool aufrufen

Sie können Tools auch explizit ansprechen:

```
Sie: "Verwende das chat Tool um zu fragen: Was ist Python?"
→ Cursor verwendet das `chat` Tool mit Ihrem lokalen Modell
```

### Methode 3: Über Composer (Code-Generierung)

1. Öffnen Sie **Composer** (`Ctrl+I` oder `Cmd+I`)
2. Stellen Sie eine Anfrage
3. Cursor kann automatisch MCP-Tools verwenden

---

## Verfügbare MCP-Tools

### 📁 Datei-Operationen

**read_file**
```
"Lies die Datei backend/model_manager.py"
```

**write_file**
```
"Schreibe 'Hello World' in die Datei test.txt"
```

**list_directory**
```
"Zeige mir den Inhalt des Verzeichnisses backend/"
```

**delete_file**
```
"Lösche die Datei test.txt"
```

**file_exists**
```
"Prüfe ob die Datei config.json existiert"
```

### 🌐 Web-Suche

**web_search**
```
"Suche im Web nach 'Python async programming'"
```

### 🤖 Model Service Integration

**list_models**
```
"Zeige mir alle verfügbaren Modelle"
```

**load_model**
```
"Lade das Modell qwen-2.5-7b-instruct als Text-Modell"
```

**model_status**
```
"Zeige mir den Status des Text-Modells"
```

**chat**
```
"Frage das lokale Modell: Erkläre mir Python Decorators"
```

**Oder mit "local:" Prefix (automatische Erkennung):**
```
"local: Erkläre mir Python Decorators"
```
→ Das "local:" Prefix wird automatisch erkannt und entfernt, das lokale Modell wird verwendet

---

## Lokales Modell als Standard verwenden

### Schritt 1: Modell laden

**Option A: Über Chat**
```
"Lade das Modell qwen-2.5-7b-instruct als Text-Modell"
```

**Option B: Über Browser**
```
POST http://127.0.0.1:8001/models/text/load
Body: {"model_id": "qwen-2.5-7b-instruct"}
```

### Schritt 2: Mit lokalem Modell chatten

**Methode 1: Mit "local:" Prefix (Empfohlen)**
```
"local: Was ist Machine Learning?"
```
→ Automatische Erkennung: "local:" wird erkannt und entfernt, lokales Modell wird verwendet

**Methode 2: Explizit Tool aufrufen**
```
"Frage das lokale Modell: Was ist Machine Learning?"
```
→ Cursor erkennt automatisch, dass das `chat` Tool benötigt wird

**Methode 3: Direkt Tool verwenden**
```
"Verwende das chat Tool: Erkläre mir Python Generators"
```

---

## Troubleshooting

### Problem: "MCP-Tools werden nicht erkannt"

**Lösung:**
1. Prüfen Sie die MCP-Logs: `View` → `Output` → `MCP`
2. Stellen Sie sicher, dass der MCP-Server verbunden ist
3. Starten Sie Cursor neu

### Problem: "Chat Tool gibt keine Antwort"

**Lösung:**
1. Prüfen Sie, ob ein Modell geladen ist:
   ```
   "Zeige mir den Status des Text-Modells"
   ```
2. Falls nicht geladen, laden Sie es:
   ```
   "Lade das Modell qwen-2.5-7b-instruct als Text-Modell"
   ```
3. Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`

### Problem: "Ctrl+Shift+P funktioniert nicht"

**Das ist normal!** MCP-Tools werden nicht über die Command Palette aufgerufen. Sie sind direkt im Chat verfügbar.

---

## Praktische Beispiele

### Beispiel 1: Datei lesen und analysieren

```
Sie: "Lies die Datei backend/model_manager.py und erkläre mir die Hauptfunktionen"
→ Cursor verwendet read_file und chat Tools
```

### Beispiel 2: Code generieren mit lokalem Modell

```
Sie: "Frage das lokale Modell: Erstelle eine Python-Funktion die Fibonacci-Zahlen berechnet"
→ Cursor verwendet chat Tool
```

### Beispiel 3: Web-Suche und lokale Analyse

```
Sie: "Suche im Web nach 'Python async best practices' und frage dann das lokale Modell nach einer Zusammenfassung"
→ Cursor verwendet web_search und chat Tools
```

### Beispiel 4: Modell-Status prüfen

```
Sie: "Zeige mir den Status aller geladenen Modelle"
→ Cursor verwendet model_status Tool
```

---

## Tipps & Tricks

### 1. Kombinieren Sie Tools

Cursor kann mehrere Tools automatisch kombinieren:
```
"Lies config.json, analysiere die Einstellungen und frage das lokale Modell nach Verbesserungsvorschlägen"
```

### 2. Explizite Tool-Nutzung

Wenn Sie ein bestimmtes Tool verwenden möchten, nennen Sie es:
```
"Verwende das chat Tool um zu fragen: Was ist der Unterschied zwischen async und sync in Python?"
```

### 3. Modell wechseln

```
"Entlade das aktuelle Text-Modell und lade stattdessen qwen-2.5-3b"
```

---

## Zusammenfassung

✅ **MCP-Tools sind automatisch verfügbar** - kein `Ctrl+Shift+P nötig  
✅ **Verwenden Sie den Chat** - Cursor wählt automatisch die passenden Tools  
✅ **Lokales Modell nutzen** - einfach "Frage das lokale Modell..." sagen  
✅ **Tools kombinieren** - Cursor kann mehrere Tools automatisch nutzen  

**Ihr lokales Modell ist jetzt vollständig in Cursor integriert!**
