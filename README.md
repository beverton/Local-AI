# Local AI Service

Ein webbasierter lokaler AI-Dienst für Ihren PC. Nutzen Sie Ihre eigenen AI-Modelle (Qwen, Phi-3, etc.) komplett offline und kostenlos.

## Features

- 🤖 **Lokale AI-Modelle**: Nutzen Sie Hugging Face Modelle direkt auf Ihrem PC
- 💬 **Modernes Chat-Interface**: Schön gestaltetes UI mit Gesprächsverlauf
- 📝 **Conversation History**: Gespräche speichern und fortsetzen
- 🎯 **Preference Learning** (optional): Die AI lernt aus Ihren Interaktionen
- ⚙️ **Flexibel**: Einfaches Wechseln zwischen verschiedenen Modellen
- 🔒 **100% Offline**: Keine Cloud-Abhängigkeiten, alles lokal
- 🌐 **Smart Browser Tabs**: Intelligentes Tab-Management - refresht existierende Tabs statt neue zu öffnen

## Voraussetzungen

- Python 3.10 oder höher
- CUDA-fähige GPU (optional, aber empfohlen für bessere Performance)
  - **RTX 50-Serie (Blackwell)**: Benötigt PyTorch 2.7.0+ mit CUDA 12.8+
  - **Andere GPUs**: PyTorch mit CUDA 11.8+ oder 12.4+
- Mindestens 8GB RAM (16GB+ empfohlen)
- Genug Speicherplatz für die Modelle

## Installation

1. **GPU-Unterstützung prüfen (empfohlen):**
```bash
# Windows
scripts\check_gpu.bat

# Linux/Mac
python scripts/check_gpu.py
```
   - Prüft ob Ihre GPU erkannt wird
   - Zeigt an, ob PyTorch CUDA-Unterstützung hat
   - Gibt Installationsempfehlungen basierend auf Ihrer GPU

2. **PyTorch mit CUDA installieren (falls nötig):**
   - **RTX 50-Serie (Blackwell)**: `scripts\install_pytorch_cuda.bat` (CUDA 12.8)
   - **Falls Probleme**: `scripts\install_pytorch_nightly.bat` (Nightly mit CUDA 12.9)
   - **Andere GPUs**: Siehe Empfehlungen in `check_gpu.bat`

3. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

4. **Konfiguration prüfen:**
   - Öffnen Sie `config.json` und prüfen Sie, ob die Modell-Pfade korrekt sind
   - Die Pfade sollten zu Ihren Modell-Verzeichnissen zeigen

5. **Server starten:**

   **Automatisch (empfohlen):**
   ```bash
   # Windows
   start_local_ai.bat
   ```
   - Startet automatisch Model Service und Local AI Server
   - Öffnet Browser-Tabs (oder refresht existierende)
   - Zum Beenden: `stop_server.bat`

   **Manuell:**
   ```bash
   cd backend
   python main.py
   ```

6. **Frontend öffnen:**
   - Bei automatischem Start: Browser öffnet sich automatisch
   - Oder navigieren Sie zu `http://127.0.0.1:8000/static/index.html`
   - Model Manager: `http://127.0.0.1:8001`

## Konfiguration

### Modelle hinzufügen

Bearbeiten Sie `config.json`:

```json
{
  "models": {
    "mein-modell": {
      "name": "Mein Modell",
      "path": "G:\\Pfad\\zum\\Modell",
      "type": "qwen2",
      "description": "Beschreibung"
    }
  }
}
```

### Standard-Modell

Setzen Sie `default_model` in `config.json` auf die ID Ihres bevorzugten Modells.

## Nutzung

1. **Modell laden**: Wählen Sie ein Modell aus dem Dropdown in der Sidebar
2. **Gespräch starten**: Klicken Sie auf "+ Neues Gespräch" oder stellen Sie direkt eine Frage
3. **Gespräch fortsetzen**: Klicken Sie auf ein Gespräch in der Sidebar
4. **Einstellungen**: Klicken Sie auf "⚙️ Einstellungen" für erweiterte Optionen

## Smart Browser Tab Management

Das Startskript `start_local_ai.bat` verwendet intelligentes Tab-Management:

- ✅ **Erster Start**: Öffnet neue Browser-Tabs für Model Manager und Frontend
- 🔄 **Wiederholter Start**: Refresht existierende Tabs statt neue zu öffnen
- 🧹 **Automatisches Cleanup**: `stop_server.bat` löscht den Tab-Status

**Vorteile:**
- Keine Tab-Flut mehr bei mehrmaligem Neustart
- Automatischer Refresh der Seiten
- Funktioniert mit allen Standard-Browsern (Chrome, Edge, Firefox)

**Mehr Informationen:** Siehe [docs/SMART_BROWSER_TABS.md](docs/SMART_BROWSER_TABS.md)

## API Endpunkte

- `GET /status` - Server-Status
- `GET /models` - Verfügbare Modelle
- `POST /models/load` - Modell laden
- `POST /chat` - Chat-Nachricht senden
- `GET /conversations` - Alle Gespräche
- `GET /conversations/{id}` - Gespräch laden
- `POST /conversations` - Neues Gespräch
- `DELETE /conversations/{id}` - Gespräch löschen
- `GET /preferences` - Präferenzen anzeigen
- `POST /preferences/toggle` - Preference Learning ein/aus
- `POST /preferences/reset` - Präferenzen zurücksetzen

## Projektstruktur

```
.
├── backend/
│   ├── main.py                 # FastAPI Server
│   ├── model_manager.py        # Modell-Verwaltung
│   ├── conversation_manager.py # Gesprächsverwaltung
│   └── preference_learner.py   # Preference Learning
├── frontend/
│   ├── index.html              # Hauptinterface
│   ├── style.css               # Styling
│   └── app.js                  # Frontend-Logik
├── data/
│   ├── conversations/          # Gespeicherte Gespräche
│   └── preferences.json        # Gelernte Präferenzen
├── config.json                 # Konfiguration
├── requirements.txt            # Python Dependencies
└── README.md                   # Diese Datei
```

## Troubleshooting

**GPU wird nicht erkannt / CUDA nicht verfügbar:**
- Führen Sie `scripts\check_gpu.bat` aus, um die Ursache zu finden
- **RTX 50-Serie (Blackwell)**: Stellen Sie sicher, dass PyTorch 2.7.0+ mit CUDA 12.8+ installiert ist
  - Verwenden Sie `scripts\install_pytorch_cuda.bat` für stabile Version
  - Oder `scripts\install_pytorch_nightly.bat` für neueste Nightly-Version
- **Fehlermeldung "sm_120 not compatible"**: Ihre PyTorch-Version unterstützt Blackwell nicht
  - Installieren Sie PyTorch 2.7.0+ mit CUDA 12.8+
- **Andere GPUs**: Prüfen Sie, ob NVIDIA-Treiber installiert sind (`nvidia-smi`)

**Modell lädt nicht:**
- Prüfen Sie, ob der Pfad in `config.json` korrekt ist
- Stellen Sie sicher, dass das Modell im Hugging Face Format vorliegt
- Prüfen Sie die Logs im Terminal

**Out of Memory:**
- Verwenden Sie ein kleineres Modell
- Reduzieren Sie `max_length` in den Einstellungen
- Schließen Sie andere Anwendungen
- Prüfen Sie, ob die GPU verwendet wird (nicht CPU)

**Langsame Antworten:**
- Prüfen Sie mit `scripts\check_gpu.bat`, ob die GPU aktiv ist
- Falls CPU verwendet wird: Installieren Sie PyTorch mit CUDA-Unterstützung
- Reduzieren Sie `max_length`
- Verwenden Sie ein kleineres Modell

### RTX 50-Serie (Blackwell-Architektur) - Spezielle Hinweise

Die RTX 50-Serie verwendet die neue Blackwell-Architektur (Compute Capability sm_120), die spezielle Anforderungen hat:

- **Erforderlich**: PyTorch 2.7.0 oder höher
- **Erforderlich**: CUDA 12.8 oder höher
- **Empfohlen**: Neueste NVIDIA-Treiber (unterstützen CUDA 13.1+)

Falls Sie Probleme haben:
1. Prüfen Sie mit `scripts\check_gpu.bat`, ob Blackwell erkannt wird
2. Installieren Sie PyTorch mit CUDA 12.8: `scripts\install_pytorch_cuda.bat`
3. Falls das nicht funktioniert, versuchen Sie Nightly-Builds: `scripts\install_pytorch_nightly.bat`

## Lizenz

Dieses Projekt ist für den persönlichen Gebrauch gedacht.


