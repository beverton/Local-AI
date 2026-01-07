# Schnellstart

## 1. Dependencies installieren

```bash
pip install -r requirements.txt
```

## 2. Server starten

**Windows:**
```bash
scripts\start_server.bat
```

**Linux/Mac:**
```bash
bash scripts/start_server.sh
```

**Oder manuell:**
```bash
cd backend
python main.py
```

## 3. Frontend öffnen

Nachdem der Server läuft, öffnen Sie eine der folgenden URLs im Browser:

- `http://127.0.0.1:8000/static/index.html`
- Oder öffnen Sie direkt `frontend/index.html` (funktioniert auch, aber API-Calls gehen dann an `http://127.0.0.1:8000`)

## 4. Erstes Modell laden

1. Wählen Sie im Dropdown in der Sidebar ein Modell aus (z.B. "Qwen 2.5 3B")
2. Warten Sie, bis "Modell geladen" angezeigt wird
3. Stellen Sie Ihre erste Frage!

## Troubleshooting

**"Modell-Pfad existiert nicht":**
- Prüfen Sie `config.json` - die Pfade müssen zu Ihren Modell-Verzeichnissen zeigen
- Windows-Pfade müssen doppelte Backslashes haben: `G:\\KI Modelle\\...`

**"Kein Modell geladen":**
- Laden Sie manuell ein Modell über das Dropdown
- Oder setzen Sie `default_model` in `config.json`

**Port bereits belegt:**
- Ändern Sie den Port in `backend/main.py` (Zeile 229) oder in `config.json`











