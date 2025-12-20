@echo off
echo ========================================
echo Local AI - Server beenden
echo ========================================
echo.

REM Finde und beende Prozesse auf Port 8000
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

echo.
echo Server beendet
timeout /t 2 /nobreak >nul
