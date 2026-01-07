# Speech Input App

Globale Sprach-Eingabe für Windows mit Whisper-Transkription.

## Installation

### Automatische Installation (Empfohlen)

1. Führe den Installer aus:
```bash
install_speech_input.bat
```

Der Installer:
- Installiert alle Dependencies automatisch
- Erstellt Desktop-Verknüpfung
- Richtet optional Auto-Start ein

### Manuelle Installation

1. Installiere Dependencies:
```bash
pip install -r requirements-speech-input.txt
```

2. Stelle sicher, dass der Local AI Server läuft (Port 8000)

3. Starte die App:
```bash
start_speech_input.bat
```

## Deinstallation

Führe den Deinstaller aus:
```bash
uninstall_speech_input.bat
```

Der Deinstaller:
- Entfernt Auto-Start (falls eingerichtet)
- Entfernt Desktop-Verknüpfung
- Entfernt optional App-Dateien
- Gibt Anweisungen für Dependencies-Deinstallation

## Funktionen

- **Globale Tastenkürzel**: Standardmäßig `Alt+Enter` (konfigurierbar)
- **Drei Aufnahme-Modi**:
  - **Toggle**: Tastenkürzel drücken = Start/Stop
  - **Push-to-Talk**: Tastenkürzel gedrückt halten = Aufnahme
  - **Auto**: Automatische Erkennung basierend auf Schwellenwert
- **System-Tray-Integration**: App läuft im Hintergrund
- **Konfigurierbare Sprache**: Optional Sprache für Transkription (leer = Auto)
- **Text-Injection**: Automatisches Einfügen in aktives Textfeld

## Konfiguration

Rechtsklick auf System-Tray-Icon → Einstellungen

- **Tastenkürzel**: z.B. `alt+enter`, `shift+x`
- **Aufnahme-Modus**: Toggle, Push-to-Talk oder Auto
- **Schwellenwert**: Für Auto-Modus (0.0 - 1.0)
- **Sprache**: Optional (leer = Auto, z.B. `de`, `en`)
- **Auto-Start**: Beim Windows-Start starten

## Nutzung

1. Starte die App über `start_speech_input.bat`
2. Konfiguriere Tastenkürzel und Modus in den Einstellungen
3. Öffne ein beliebiges Textfeld (z.B. Notepad, Browser)
4. Drücke das konfigurierte Tastenkürzel
5. Sprich - die Aufnahme startet automatisch
6. Text wird nach Transkription automatisch eingefügt

## App beenden

**Rechtsklick auf das System-Tray-Icon → "Beenden"**

Die App läuft im Hintergrund. Um sie zu beenden:
- Rechtsklick auf das Speech Input Icon im System-Tray (rechts unten)
- Klicke auf "Beenden" im Menü

Siehe auch: `HOW_TO_QUIT.md` für weitere Details

## API-Integration

Die App nutzt die existierende `/audio/transcribe` Schnittstelle (Port 8000) wie das Frontend. Keine separate Modell-Ladung nötig.

## Troubleshooting

- **Kein Audio**: Prüfe Mikrofon-Berechtigungen
- **Transkription fehlgeschlagen**: Stelle sicher, dass der Server läuft und ein Audio-Modell geladen ist
- **Hotkey funktioniert nicht**: Prüfe ob Tastenkombination bereits von anderer Software verwendet wird
- **Text wird nicht eingefügt**: Prüfe ob aktives Fenster ein Textfeld ist

