@echo off
echo ========================================
echo Local AI - Automatischer Start
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

REM 1. Beende existierende Python-Server und KI-Prozesse
echo [1/7] Beende existierende Server und KI-Prozesse...
echo.

REM Beende Server-Fenster
echo Beende Server-Fenster...
taskkill /FI "WINDOWTITLE eq Local AI Server*" /F >nul 2>&1
if not errorlevel 1 (
    echo "Local AI Server" Fenster beendet
)

taskkill /FI "WINDOWTITLE eq Model Service*" /F >nul 2>&1
if not errorlevel 1 (
    echo "Model Service" Fenster beendet
)

REM Beende Prozesse auf Ports (vereinfacht - ohne mehrfache Pruefungen)
echo.
echo Beende Prozesse auf Port 8000 und 8001...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo Beende Prozess %%a auf Port 8000...
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo Beende Prozess %%a auf Port 8001...
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
)

REM Warte auf Port-Freigabe
echo.
echo Warte auf Port-Freigabe...
timeout /t 3 /nobreak >nul

REM 2. Pruefe ob Port 8001 frei ist, bevor Model-Service gestartet wird
echo [2/7] Pruefe Port 8001 vor Model-Service-Start...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo FEHLER: Port 8001 ist noch belegt von Prozess %%a - kann Model-Service nicht starten!
    pause
    exit /b 1
)

REM 3. Starte Model-Service in neuem Fenster
echo [3/7] Starte Model-Service...
echo Ein neues Fenster "Model Service" wird geoeffnet - dort sehen Sie die Logs.
echo Aktuelles Verzeichnis: %CD%
cd backend
if errorlevel 1 (
    echo FEHLER: Konnte nicht ins backend-Verzeichnis wechseln!
    pause
    exit /b 1
)
echo Backend-Verzeichnis: %CD%
if not exist model_service.py (
    echo FEHLER: model_service.py nicht gefunden in %CD%!
    pause
    exit /b 1
)
echo model_service.py gefunden, starte Model-Service...
start "Model Service" cmd /K "python model_service.py"
if errorlevel 1 (
    echo FEHLER: Model-Service konnte nicht gestartet werden!
    pause
    exit /b 1
)
cd ..

REM 4. Warte bis Model-Service bereit ist
echo [4/7] Warte auf Model-Service-Start...
echo Bitte warten Sie 5 Sekunden, bis der Model-Service gestartet ist...
timeout /t 5 /nobreak >nul

REM Optional: Prüfe ob Model-Service läuft (nicht-blockierend)
echo Prüfe Model-Service...
python scripts\check_server_health.py 127.0.0.1 8001 /status >nul 2>&1
if errorlevel 1 (
    echo WARNUNG: Model-Service scheint noch nicht bereit zu sein.
    echo Bitte prüfen Sie das "Model Service" Fenster für Fehlermeldungen.
    echo Der Start wird trotzdem fortgesetzt...
) else (
    echo Model-Service ist bereit!
)

REM 4b. Öffne Model Manager im Browser
echo Öffne Model Manager im Browser...
start "" "http://127.0.0.1:8001/"

REM 5. Pruefe ob Port 8000 frei ist, bevor Local AI Server gestartet wird
echo [5/7] Pruefe Port 8000 vor Local AI Server-Start...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo FEHLER: Port 8000 ist noch belegt von Prozess %%a - kann Local AI Server nicht starten!
    pause
    exit /b 1
)

REM 6. Starte Local AI Server in neuem Fenster (sichtbar für Logs)
echo [6/7] Starte Local AI Server...
echo Ein neues Fenster "Local AI Server" wird geoeffnet - dort sehen Sie die Logs.
cd backend
if not exist main.py (
    echo FEHLER: main.py nicht gefunden in %CD%!
    pause
    exit /b 1
)
echo main.py gefunden, starte Server...
start "Local AI Server" cmd /K "python main.py"
if errorlevel 1 (
    echo FEHLER: Server konnte nicht gestartet werden!
    pause
    exit /b 1
)
cd ..

REM 7. Warte bis Server bereit ist
echo [7/7] Warte auf Server-Start...
echo Bitte warten Sie 5 Sekunden, bis der Server gestartet ist...
timeout /t 5 /nobreak >nul

REM Optional: Prüfe ob Server läuft (nicht-blockierend)
echo Prüfe Local AI Server...
python scripts\check_server_health.py 127.0.0.1 8000 /status >nul 2>&1
if errorlevel 1 (
    echo WARNUNG: Local AI Server scheint noch nicht bereit zu sein.
    echo Bitte prüfen Sie das "Local AI Server" Fenster für Fehlermeldungen.
    echo Der Browser wird trotzdem geöffnet...
) else (
    echo Local AI Server ist bereit!
)
echo.
:open_browser

REM 8. Öffne Browser (nicht nummeriert, da optional)
echo Öffne Browser...
start "" "http://127.0.0.1:8000/static/index.html"

echo.
echo ========================================
echo Local AI ist gestartet!
echo Model-Service laeuft auf: http://127.0.0.1:8001
echo Server laeuft auf: http://127.0.0.1:8000
echo Browser sollten sich automatisch geoeffnet haben:
echo          - Model Manager: http://127.0.0.1:8001
echo          - Local AI Frontend: http://127.0.0.1:8000/static/index.html
echo.
echo HINWEIS: 
echo          - Model-Service laeuft in einem separaten Fenster "Model Service"
echo          - Local AI Server laeuft in einem separaten Fenster "Local AI Server"
echo          - Dort sehen Sie alle Logs und Server-Ausgaben.
echo          - Zum Beenden: Schliessen Sie beide Fenster
echo          - Oder verwenden Sie: stop_server.bat
echo ========================================
echo.
echo Dieses Fenster kann jetzt geschlossen werden.
timeout /t 3 /nobreak >nul

