# Speech Input App - Executable Build

Diese Anleitung erklärt, wie Sie eine standalone ausführbare Datei (`.exe`) für die Speech Input App erstellen.

## Voraussetzungen

1. **Python 3.10 oder höher** muss installiert sein
2. **Alle Dependencies** müssen installiert sein:
   ```bash
   pip install -r requirements.txt
   ```

## Build-Prozess

### Automatisch (empfohlen)

Führen Sie einfach das Build-Script aus:

```bash
cd speech_input_app
build_executable.bat
```

Das Script:
- Prüft ob PyInstaller installiert ist und installiert es bei Bedarf
- Prüft ob alle Dependencies vorhanden sind
- Erstellt die ausführbare Datei
- Öffnet das `dist`-Verzeichnis nach erfolgreichem Build

### Manuell

Falls Sie den Build-Prozess manuell durchführen möchten:

```bash
cd speech_input_app

# Installiere PyInstaller (falls nicht vorhanden)
pip install pyinstaller

# Erstelle Executable
pyinstaller speech_input.spec
```

Die ausführbare Datei befindet sich nach dem Build in:
```
dist/speech_input.exe
```

## Verwendung der Executable

1. **Kopieren Sie die Datei** `dist/speech_input.exe` an einen gewünschten Ort
2. **Stellen Sie sicher**, dass `config.json` im gleichen Verzeichnis liegt (wird automatisch mit eingebunden)
3. **Starten Sie** `speech_input.exe`

## Hinweise

- Die Executable ist **standalone** - Python muss nicht installiert sein auf dem Zielsystem
- Die Datei ist relativ groß (~50-100 MB), da alle Dependencies eingebunden sind
- Beim ersten Start kann es etwas länger dauern, da PyInstaller temporäre Dateien extrahiert
- Die App benötigt weiterhin den Local AI Server (Backend) für die Transkription

## Troubleshooting

### "PyInstaller nicht gefunden"
Installieren Sie PyInstaller manuell:
```bash
pip install pyinstaller
```

### "Module nicht gefunden" beim Start
Stellen Sie sicher, dass alle Dependencies installiert sind:
```bash
pip install -r requirements.txt
```

### Executable startet nicht
- Prüfen Sie ob `config.json` im gleichen Verzeichnis liegt
- Führen Sie die Executable aus der Kommandozeile aus, um Fehlermeldungen zu sehen
- Prüfen Sie ob alle Windows-Dependencies (pywin32) korrekt installiert sind

### Antivirus meldet False Positive
PyInstaller-Executables werden manchmal von Antivirus-Software als verdächtig eingestuft. Dies ist ein bekanntes Problem. Sie können:
- Die Datei zur Whitelist hinzufügen
- Den Build mit Code-Signing durchführen (erfordert Zertifikat)


