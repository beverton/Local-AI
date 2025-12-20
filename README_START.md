# Schnellstart - Local AI

## Einfacher Start

**Doppelklick auf:** `start_local_ai.bat`

Das Skript macht automatisch:
1. ✅ Beendet existierende Server
2. ✅ Startet den Server neu
3. ✅ Öffnet den Browser automatisch

## Manuelle Steuerung

### Server starten
```bash
start_local_ai.bat
```

### Server beenden
```bash
stop_server.bat
```

Oder einfach das Terminal-Fenster schließen, in dem der Server läuft.

## Troubleshooting

**Browser öffnet sich nicht:**
- Öffnen Sie manuell: `http://127.0.0.1:8000/static/index.html`
- Oder direkt: `frontend/index.html`

**Port bereits belegt:**
- Führen Sie `stop_server.bat` aus
- Oder ändern Sie den Port in `backend/main.py` (Zeile 231)

**Server startet nicht:**
- Prüfen Sie, ob alle Dependencies installiert sind: `pip install -r requirements.txt`
- Prüfen Sie die Logs im Terminal


