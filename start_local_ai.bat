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

REM 1. Beende existierende Python-Server
echo [1/4] Beende existierende Server...
echo Pruefe auf laufende Server auf Port 8000...

REM Beende zuerst alle Python-Prozesse mit "Local AI Server" im Titel
taskkill /FI "WINDOWTITLE eq Local AI Server*" /F >nul 2>&1

REM Finde und beende Prozesse auf Port 8000 (mehrfach versuchen)
REM Suche nach LISTENING (EN) und ABHOEREN (DE) - Windows zeigt unterschiedliche Sprachen
set ATTEMPTS=0
:kill_port_loop
set /a ATTEMPTS+=1
if %ATTEMPTS% gtr 5 goto port_check
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo Gefundener Prozess auf Port 8000: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    ) else (
        echo Konnte Prozess %%a nicht beenden - versuche erneut...
    )
)
timeout /t 1 /nobreak >nul
goto kill_port_loop

:port_check
REM Pruefe ob Port jetzt frei ist (beide Sprachen)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo WARNUNG: Port 8000 ist immer noch belegt von Prozess %%a
    echo Versuche Prozess erneut zu beenden...
    taskkill /F /PID %%a
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
    timeout /t 2 /nobreak >nul
)

REM Warte kurz damit Ports freigegeben werden
timeout /t 2 /nobreak >nul

REM 2. Starte Server in neuem Fenster (sichtbar für Logs)
echo [2/4] Starte Server...
echo Ein neues Fenster "Local AI Server" wird geoeffnet - dort sehen Sie die Logs.
echo Aktuelles Verzeichnis: %CD%
cd backend
if errorlevel 1 (
    echo FEHLER: Konnte nicht ins backend-Verzeichnis wechseln!
    pause
    exit /b 1
)
echo Backend-Verzeichnis: %CD%
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

REM 3. Warte bis Server bereit ist (reduzierte Pruefung um Flackern zu vermeiden)
echo [3/4] Warte auf Server-Start...
echo Bitte warten Sie 5 Sekunden, bis der Server gestartet ist...
timeout /t 5 /nobreak >nul
echo Server sollte jetzt bereit sein.
echo.
:open_browser

REM 4. Öffne Browser
echo [4/4] Öffne Browser...
start "" "http://127.0.0.1:8000/static/index.html"

echo.
echo ========================================
echo Local AI ist gestartet!
echo Server laeuft auf: http://127.0.0.1:8000
echo Browser sollte sich automatisch geoeffnet haben.
echo.
echo HINWEIS: Der Server laeuft in einem separaten Fenster "Local AI Server"
echo          Dort sehen Sie alle Logs und Server-Ausgaben.
echo          Zum Beenden: Schliessen Sie das "Local AI Server" Fenster
echo          Oder verwenden Sie: stop_server.bat
echo ========================================
echo.
echo Dieses Fenster kann jetzt geschlossen werden.
timeout /t 3 /nobreak >nul

