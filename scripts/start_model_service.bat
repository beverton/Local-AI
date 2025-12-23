@echo off
echo ========================================
echo Model Service - Start
echo ========================================
echo.

REM Wechsle ins Projekt-Verzeichnis
cd /d "%~dp0\.."

REM Prüfe ob Python verfügbar ist
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH!
    echo Bitte installieren Sie Python oder fügen Sie es zum PATH hinzu.
    pause
    exit /b 1
)

REM Prüfe ob Port 8001 bereits belegt ist
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo WARNUNG: Port 8001 ist bereits belegt von Prozess %%a
    echo Versuche Prozess zu beenden...
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
    timeout /t 2 /nobreak >nul
)

REM Starte Model-Service
echo Starte Model-Service...
cd backend
if not exist model_service.py (
    echo FEHLER: model_service.py nicht gefunden!
    pause
    exit /b 1
)

start "Model Service" cmd /K "python model_service.py"
if errorlevel 1 (
    echo FEHLER: Model-Service konnte nicht gestartet werden!
    pause
    exit /b 1
)

cd ..

REM Warte kurz bis Service bereit ist
echo.
echo Warte auf Service-Start...
timeout /t 3 /nobreak >nul

REM Öffne Browser
echo Öffne Model Manager im Browser...
start "" "http://127.0.0.1:8001/"

echo.
echo ========================================
echo Model Service gestartet!
echo Laeuft auf: http://127.0.0.1:8001
echo Browser sollte sich automatisch geoeffnet haben.
echo.
echo HINWEIS: Der Service laeuft in einem separaten Fenster "Model Service"
echo          Zum Beenden: Schliessen Sie das Fenster oder verwenden Sie stop_model_service.bat
echo ========================================
echo.
pause

