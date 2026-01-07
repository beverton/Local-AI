@echo off
echo ========================================
echo GPU-Unterstuetzung pruefen
echo ========================================
echo.

REM Wechsle ins Projekt-Verzeichnis
cd /d "%~dp0\.."

REM Prüfe ob Python verfügbar ist
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH!
    pause
    exit /b 1
)

echo Starte GPU-Check...
echo.

python scripts\check_gpu.py

echo.
echo ========================================
echo Pruefung abgeschlossen
echo ========================================
echo.
pause










