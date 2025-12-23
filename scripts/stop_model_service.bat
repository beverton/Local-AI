@echo off
echo ========================================
echo Model Service - Stop
echo ========================================
echo.

REM Beende Model-Service-Prozesse
echo Beende Model-Service-Prozesse...

REM Beende Fenster mit "Model Service" im Titel
taskkill /FI "WINDOWTITLE eq Model Service*" /F >nul 2>&1
if not errorlevel 1 (
    echo Model-Service-Fenster beendet
)

REM Beende Python-Prozesse die model_service.py ausführen
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /C:"python.exe"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /C:"model_service.py" >nul
    if not errorlevel 1 (
        echo Gefundener Python-Prozess mit model_service.py: PID %%a
        taskkill /F /PID %%a >nul 2>&1
        if not errorlevel 1 (
            echo Prozess %%a erfolgreich beendet
        )
    )
)

REM Beende Prozesse auf Port 8001
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo Gefundener Prozess auf Port 8001: PID %%a
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo Prozess %%a erfolgreich beendet
    )
)

timeout /t 1 /nobreak >nul

REM Prüfe ob Port jetzt frei ist
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8001" ^| findstr /C:"LISTENING" /C:"ABH"') do (
    echo WARNUNG: Port 8001 ist immer noch belegt von Prozess %%a
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo.
echo ========================================
echo Model Service gestoppt!
echo ========================================
echo.
pause







