# Lokales Modell in Cursor über "Add New Models" hinzufügen

## Einfachste Methode: Custom Model in Cursor

Cursor hat eine "Add New Models"-Funktion, mit der Sie benutzerdefinierte Modelle hinzufügen können, die über eine OpenAI-kompatible API erreichbar sind.

## Schritt 1: Model Service als OpenAI-kompatible API

Unser Model Service muss einen OpenAI-kompatiblen Chat-Completion-Endpoint bereitstellen.

### Aktueller Stand:
- Endpoint: `POST http://127.0.0.1:8001/chat`
- Format: Custom (nicht OpenAI-kompatibel)

### Lösung:
Wir müssen einen OpenAI-kompatiblen Endpoint hinzufügen:
- Endpoint: `POST http://127.0.0.1:8001/v1/chat/completions`
- Format: OpenAI Chat Completion API

## Schritt 2: In Cursor hinzufügen

Nach Implementierung des OpenAI-kompatiblen Endpoints:

1. **Öffnen Sie Cursor Settings**
   - `Settings` → `Models` → `Add New Model`

2. **Modell-Details eingeben:**
   - **Name**: `Local` (oder `Local AI`)
   - **API Endpoint**: `http://127.0.0.1:8001/v1/chat/completions`
   - **API Key**: (kann leer bleiben oder einen Dummy-Wert wie "local")
   - **Model ID**: `qwen-2.5-7b-instruct` (oder wie Ihr Modell heißt)

3. **Speichern und testen**

## Schritt 3: Model Service neu starten

Der OpenAI-kompatible Endpoint wurde implementiert. Starten Sie den Model Service neu:

1. **Model Service stoppen:**
   - Schließen Sie das Model Service Fenster
   - Oder führen Sie `scripts/stop_model_service.bat` aus

2. **Model Service neu starten:**
   - Führen Sie `scripts/start_model_service.bat` aus
   - Oder: `python backend/model_service.py`

3. **Testen Sie den Endpoint:**
   ```
   http://127.0.0.1:8001/v1/chat/completions
   ```

## Schritt 4: In Cursor hinzufügen

1. **Öffnen Sie Cursor Settings**
   - Klicken Sie auf das Zahnrad-Symbol (Settings)
   - Gehen Sie zu `Models` oder `AI Models`
   - Klicken Sie auf `Add New Model` oder `+ Add Model`

2. **Modell-Details eingeben:**

   **Name**: `Local`

   **Provider**: `OpenAI` (oder `Custom` falls verfügbar)

   **API Base URL**: `http://127.0.0.1:8001/v1`

   **API Key**: `local` (oder leer lassen - wird nicht geprüft)

   **Model ID**: `qwen-2.5-7b-instruct`

3. **Speichern**

## Schritt 5: Lokales Modell verwenden

1. Öffnen Sie den Cursor Chat (`Ctrl+L`)
2. Klicken Sie auf die Modellauswahl (oben im Chat)
3. Wählen Sie **"Local"** aus der Liste
4. Stellen Sie eine Frage - das lokale Modell sollte antworten!

## Troubleshooting

### Problem: "Connection refused" oder "Network error"

**Lösung:**
- Prüfen Sie, ob der Model Service läuft: `http://127.0.0.1:8001`
- Prüfen Sie, ob ein Modell geladen ist: `http://127.0.0.1:8001/models/text/status`

### Problem: "No model loaded"

**Lösung:**
- Laden Sie ein Modell über die Model Manager UI: `http://127.0.0.1:8001`
- Oder über API: `POST http://127.0.0.1:8001/models/text/load` mit Body `{"model_id": "qwen-2.5-7b-instruct"}`

### Problem: Modell erscheint nicht in der Liste

**Lösung:**
- Starten Sie Cursor neu
- Prüfen Sie die API Base URL (muss `/v1` am Ende haben)
- Prüfen Sie, ob der Endpoint erreichbar ist

## Vorteile dieser Methode

✅ **Einfach**: Nur Name eingeben in Cursor  
✅ **Direkt**: Erscheint in der Modellauswahl wie "auto"  
✅ **Standard**: Verwendet OpenAI-kompatible API  
✅ **Flexibel**: Kann wie jedes andere Modell verwendet werden

## Nächste Schritte

1. ✅ Model Service neu starten
2. ✅ Modell in Cursor über "Add New Model" hinzufügen
3. ✅ "Local" aus der Modellauswahl wählen
4. ✅ Testen mit einer Frage

Ihr lokales Modell ist jetzt vollständig in Cursor integriert!
