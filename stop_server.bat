@echo off
echo ========================================
echo Local AI - Server beenden
echo ========================================
echo.

REM 1. Beende Local AI Server (Port 8000)
echo [1/3] Beende Local AI Server...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Gefundener Prozess auf Port 8000: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
)

REM Beende auch Python-Prozesse mit "Local AI Server" im Titel
taskkill /FI "WINDOWTITLE eq Local AI Server*" /F >nul 2>&1
if not errorlevel 1 (
    echo Server-Prozess beendet
)

REM 2. Beende Model Service (Port 8001)
echo.
echo [2/3] Beende Model Service...
taskkill /FI "WINDOWTITLE eq Model Service*" /F >nul 2>&1
if not errorlevel 1 (
    echo Model-Service-Fenster beendet
)

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr "LISTENING"') do (
    echo Gefundener Prozess auf Port 8001: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
)

REM 3. Lösche Browser-Tab-Status
echo.
echo [3/3] Lösche Browser-Tab-Status...
python scripts\open_or_refresh_browser.py --clear >nul 2>&1
if not errorlevel 1 (
    echo Browser-Tab-Status gelöscht
)

echo.
echo ========================================
echo Alle Services beendet!
echo ========================================
timeout /t 2 /nobreak >nul
