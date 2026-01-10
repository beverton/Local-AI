# Cursor MCP Integration - Schritt-für-Schritt Anleitung

## Übersicht

Diese Anleitung zeigt Ihnen, wie Sie Ihren lokalen AI-Model Service (Local AI) als MCP-Server in Cursor integrieren und als Standard-Modell verwenden.

## Voraussetzungen

✅ **Model Service läuft** auf Port 8001  
✅ **Python** ist installiert und im PATH  
✅ **Cursor Editor** ist installiert  
✅ **Lokales Modell** ist geladen (z.B. `qwen-2.5-7b-instruct`)

---

## Schritt 1: Model Service starten

### 1.1 Model Service prüfen

Öffnen Sie einen Browser und gehen Sie zu:
```
http://127.0.0.1:8001/status
```

Sie sollten eine JSON-Antwort mit Modell-Status sehen.

### 1.2 Falls nicht gestartet

**Option A: Über Start-Script**
```bash
# Im Projekt-Verzeichnis
start_local_ai.bat
```

**Option B: Manuell starten**
```bash
cd "G:\04-CODING\Local Ai"
python backend/model_service.py
```

Der Service sollte auf `http://127.0.0.1:8001` laufen.

---

## Schritt 2: Cursor MCP-Konfiguration erstellen

### 2.1 Cursor-Konfigurationsdatei finden

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**Vollständiger Pfad (normalerweise):**
```
C:\Users\IHR_USERNAME\AppData\Roaming\Cursor\User\globalStorage\mcp.json
```

**Tipp:** Drücken Sie `Win + R`, geben Sie `%APPDATA%\Cursor\User\globalStorage` ein und drücken Enter.

### 2.2 Konfigurationsdatei erstellen/bearbeiten

1. Öffnen Sie den Ordner `%APPDATA%\Cursor\User\globalStorage`
2. Falls `mcp.json` nicht existiert, erstellen Sie eine neue Datei mit diesem Namen
3. Öffnen Sie `mcp.json` mit einem Text-Editor (Notepad++, VS Code, etc.)

### 2.3 Konfiguration einfügen

**WICHTIG:** Passen Sie den Pfad `G:\04-CODING\Local Ai` an Ihren tatsächlichen Projekt-Pfad an!

```json
{
  "mcpServers": {
    "local-ai": {
      "command": "python",
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

**Hinweise:**
- Verwenden Sie **doppelte Backslashes** (`\\`) in Windows-Pfaden
- Der Pfad muss **absolut** sein (nicht relativ)
- Falls Sie Python nicht im PATH haben, verwenden Sie den vollständigen Pfad:
  ```json
  "command": "C:\\Python313\\python.exe"
  ```

### 2.4 Datei speichern

Speichern Sie die Datei als `mcp.json` (UTF-8 Encoding).

---

## Schritt 3: Cursor neu starten

1. **Schließen Sie Cursor vollständig** (nicht nur das Fenster, sondern den gesamten Prozess)
2. **Starten Sie Cursor neu**
3. Cursor lädt die MCP-Konfiguration beim Start

---

## Schritt 4: MCP-Server-Verbindung prüfen

### 4.1 MCP-Status in Cursor prüfen

1. Öffnen Sie Cursor
2. Drücken Sie `Ctrl+Shift+P` (Command Palette)
3. Suchen Sie nach: `MCP` oder `Model Context Protocol`
4. Sie sollten Optionen wie "MCP: Show Servers" sehen

### 4.2 MCP-Logs prüfen

1. In Cursor: `View` → `Output`
2. Wählen Sie im Dropdown: `MCP` oder `Model Context Protocol`
3. Sie sollten Logs sehen, die zeigen, ob der Server verbunden ist

**Erfolgreiche Verbindung sieht so aus:**
```
[INFO] MCP Server "local-ai" verbunden
[INFO] Tools geladen: 11 Tools verfügbar
```

**Bei Fehlern:**
- Prüfen Sie die Logs auf Fehlermeldungen
- Stellen Sie sicher, dass der Model Service läuft
- Prüfen Sie, ob die Pfade in `mcp.json` korrekt sind

---

## Schritt 5: Lokales Modell als Standard verwenden

### 5.1 Modell-Status prüfen

Stellen Sie sicher, dass Ihr lokales Modell geladen ist:

**Option A: Über Browser**
```
http://127.0.0.1:8001/models/text/status
```

**Option B: Über MCP Tool in Cursor**

In Cursor können Sie jetzt MCP-Tools verwenden:
1. Öffnen Sie die Command Palette (`Ctrl+Shift+P`)
2. Suchen Sie nach `MCP: Call Tool`
3. Wählen Sie `model_status`
4. Geben Sie `{"model_type": "text"}` ein

### 5.2 Modell laden (falls nicht geladen)

**Option A: Über Browser**
```
POST http://127.0.0.1:8001/models/text/load
Body: {"model_id": "qwen-2.5-7b-instruct"}
```

**Option B: Über MCP Tool in Cursor**
1. Command Palette → `MCP: Call Tool`
2. Wählen Sie `load_model`
3. Geben Sie ein:
   ```json
   {
     "model_id": "qwen-2.5-7b-instruct",
     "model_type": "text"
   }
   ```

### 5.3 Cursor so konfigurieren, dass es das lokale Modell verwendet

**WICHTIG:** Cursor verwendet standardmäßig seine eigenen Modelle. Um das lokale Modell zu verwenden:

1. **In Cursor Settings:**
   - `Ctrl+,` (Settings öffnen)
   - Suchen Sie nach "Model" oder "AI"
   - Prüfen Sie die Model-Einstellungen

2. **MCP Chat verwenden:**
   - Cursor kann über MCP-Tools mit Ihrem lokalen Modell kommunizieren
   - Verwenden Sie `MCP: Call Tool` → `chat` für direkte Kommunikation

3. **Für Code-Completion:**
   - Cursor verwendet normalerweise seine eigenen Modelle für Code-Completion
   - Das lokale Modell wird über MCP-Tools verfügbar gemacht
   - Sie können es für spezielle Anfragen verwenden

---

## Schritt 6: Testen

### 6.1 Verfügbare Tools testen

1. Command Palette (`Ctrl+Shift+P`)
2. `MCP: Call Tool`
3. Testen Sie verschiedene Tools:

**list_models:**
```json
{"model_type": "text"}
```

**model_status:**
```json
{"model_type": "text"}
```

**chat:**
```json
{
  "message": "Hallo, wie geht es dir?",
  "max_length": 512,
  "temperature": 0.7
}
```

### 6.2 Chat mit lokalem Modell testen

**Methode 1: Mit "local:" Prefix (Empfohlen)**
```
Im Cursor Chat: "local: Was ist 2+5?"
```
→ Das "local:" Prefix wird automatisch erkannt und entfernt, lokales Modell wird verwendet

**Methode 2: Über MCP Tool**
1. Command Palette → `MCP: Call Tool` → `chat`
2. Geben Sie eine Nachricht ein:
   ```json
   {
     "message": "Erkläre mir Python Decorators",
     "max_length": 1024,
     "temperature": 0.7
   }
   ```
   Oder mit "local:" Prefix:
   ```json
   {
     "message": "local: Erkläre mir Python Decorators",
     "max_length": 1024,
     "temperature": 0.7
   }
   ```
3. Sie sollten eine Antwort von Ihrem lokalen Modell erhalten

---

## Schritt 7: Troubleshooting

### Problem: "Model Service nicht verfügbar"

**Lösung:**
1. Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`
2. Prüfen Sie die Firewall-Einstellungen
3. Stellen Sie sicher, dass Port 8001 nicht blockiert ist

### Problem: "Python nicht gefunden"

**Lösung:**
1. Prüfen Sie, ob Python im PATH ist: `python --version`
2. Verwenden Sie den vollständigen Pfad in `mcp.json`:
   ```json
   "command": "C:\\Python313\\python.exe"
   ```

### Problem: "Module nicht gefunden"

**Lösung:**
1. Installieren Sie Dependencies:
   ```bash
   cd "G:\04-CODING\Local Ai"
   pip install -r requirements-base.txt
   ```
2. Prüfen Sie, ob `PYTHONPATH` korrekt gesetzt ist

### Problem: "MCP Server startet nicht"

**Lösung:**
1. Prüfen Sie die Cursor-Logs: `View` → `Output` → `MCP`
2. Testen Sie den Server manuell:
   ```bash
   cd "G:\04-CODING\Local Ai\backend"
   python mcp_server.py
   ```
3. Prüfen Sie, ob alle Imports funktionieren

### Problem: "Chat gibt keine Antwort"

**Lösung:**
1. Prüfen Sie, ob ein Modell geladen ist: `http://127.0.0.1:8001/models/text/status`
2. Verwenden Sie längeres `max_length` (z.B. 512 statt 100)
3. Prüfen Sie die Model Service-Logs

---

## Schritt 8: Verfügbare MCP-Tools

Ihr MCP-Server stellt folgende Tools zur Verfügung:

### Datei-Operationen:
- **web_search**: Websuche durchführen
- **read_file**: Datei lesen
- **write_file**: Datei schreiben
- **list_directory**: Verzeichnis auflisten
- **delete_file**: Datei löschen
- **file_exists**: Datei-Existenz prüfen

### Model Service Integration:
- **list_models**: Alle verfügbaren Modelle auflisten
- **load_model**: Modell laden
- **unload_model**: Modell entladen
- **model_status**: Status des geladenen Modells abrufen
- **chat**: Chat mit lokalem Modell

---

## Schritt 9: Automatisches Laden beim Start

### 9.1 Model Service konfigurieren

Der Model Service lädt standardmäßig das `default_model` aus `config.json` beim Start.

**Prüfen Sie `config.json`:**
```json
{
  "default_model": "qwen-2.5-7b-instruct",
  ...
}
```

### 9.2 Automatisches Starten

Sie können ein Script erstellen, das sowohl Model Service als auch Cursor startet:

**start_cursor_with_local_ai.bat:**
```batch
@echo off
echo Starte Model Service...
start "Model Service" python "G:\04-CODING\Local Ai\backend\model_service.py"
timeout /t 5 /nobreak
echo Starte Cursor...
start "" "C:\Users\IHR_USERNAME\AppData\Local\Programs\cursor\Cursor.exe"
```

---

## Zusammenfassung

✅ **Model Service läuft** auf Port 8001  
✅ **MCP-Konfiguration** in Cursor erstellt  
✅ **Cursor neu gestartet**  
✅ **MCP-Server verbunden**  
✅ **Lokales Modell geladen**  
✅ **Tools getestet**  

**Ihr lokales Modell ist jetzt über MCP in Cursor verfügbar!**

---

## Nächste Schritte

1. **Experimentieren Sie mit den MCP-Tools** in Cursor
2. **Verwenden Sie `chat`** für direkte Kommunikation mit Ihrem lokalen Modell
3. **Automatisieren Sie Workflows** mit MCP-Tools
4. **Erweitern Sie die Funktionalität** nach Bedarf

Bei Fragen oder Problemen:
- Prüfen Sie die Cursor-Logs (`View` → `Output` → `MCP`)
- Prüfen Sie die Model Service-Logs
- Testen Sie den MCP-Server manuell
