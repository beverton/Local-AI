@echo off
echo ========================================
echo Speech Input App - Deinstaller
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

REM Starte Deinstaller
python speech_input_app\installer.py uninstall

pause

