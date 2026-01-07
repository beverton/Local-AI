@echo off
echo ========================================
echo Local AI - Speech Input App
echo ========================================
echo.

REM Wechsle ins Projekt-Verzeichnis
cd /d "%~dp0"

REM Prüfe ob Python verfügbar ist
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH!
    echo Bitte installieren Sie Python oder fügen Sie es zum PATH hinzu.
    pause
    exit /b 1
)

REM Prüfe ob Speech Input App existiert
if not exist "speech_input_app\speech_input.py" (
    echo FEHLER: speech_input.py nicht gefunden!
    echo Bitte stellen Sie sicher, dass die Speech Input App installiert ist.
    pause
    exit /b 1
)

REM Optional: Prüfe ob Server läuft (nicht-blockierend)
REM Hinweis: Server-Prüfung übersprungen - App prüft selbst beim Start
echo.

REM Starte Speech Input App
echo Starte Speech Input App...
echo.
python speech_input_app\speech_input.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo FEHLER: Speech Input App konnte nicht gestartet werden!
    echo ========================================
    echo.
    echo Moegliche Ursachen:
    echo - Dependencies nicht installiert (pip install -r requirements-speech-input.txt)
    echo - Server laeuft nicht (Port 8000)
    echo - Fehler in der App-Konfiguration
    echo.
    echo Bitte pruefen Sie die Fehlermeldungen oben.
    echo.
    pause
    exit /b 1
)

echo.
echo Speech Input App wurde beendet.
timeout /t 2 /nobreak >nul

