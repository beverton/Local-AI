# Cursor - Lokales Modell über Custom API einrichten

## Wenn "Add New Model" nur einen Namen erlaubt

Falls Cursor's "Add New Model" nur einen Namen akzeptiert, müssen Sie die API-Konfiguration woanders eingeben.

## Option 1: API-Einstellungen in Cursor

1. **Öffnen Sie Cursor Settings**
   - `File` → `Preferences` → `Settings`
   - Oder drücken Sie `Ctrl+,`

2. **Suchen Sie nach:**
   - `API`
   - `OpenAI`
   - `Base URL`
   - `Custom API`

3. **Mögliche Einstellungen:**
   - **OpenAI API Base URL**: `http://127.0.0.1:8001/v1`
   - **OpenAI API Key**: `local`

## Option 2: Cursor Rules / Settings JSON

1. **Öffnen Sie die Settings-Datei direkt:**
   - `Ctrl+Shift+P` → "Preferences: Open User Settings (JSON)"

2. **Fügen Sie hinzu:**
   ```json
   {
     "openai.apiBase": "http://127.0.0.1:8001/v1",
     "openai.apiKey": "local"
   }
   ```

## Option 3: Environment Variables

1. **Setzen Sie Umgebungsvariablen:**
   ```bash
   OPENAI_API_BASE=http://127.0.0.1:8001/v1
   OPENAI_API_KEY=local
   ```

2. **Starten Sie Cursor neu**

## Option 4: Cursor Rules File

1. **Erstellen Sie `.cursorrules` im Projekt-Root:**
   ```
   API_BASE_URL=http://127.0.0.1:8001/v1
   ```

## Was ist die genaue Situation?

Bitte prüfen Sie:
1. Wenn Sie "Add New Model" klicken und nur einen Namen eingeben können:
   - Welche anderen Felder oder Optionen sehen Sie?
   - Gibt es ein Dropdown für "Provider" oder "Model Type"?

2. In den Cursor Settings:
   - Suchen Sie nach "API" oder "OpenAI"
   - Gibt es Felder für "Base URL" oder "API Endpoint"?

3. Alternativ:
   - Können Sie einen Screenshot von der "Add New Model"-Oberfläche machen?
   - Oder beschreiben, welche Optionen genau verfügbar sind?

Dann kann ich Ihnen die exakte Vorgehensweise für Ihre Cursor-Version zeigen!
