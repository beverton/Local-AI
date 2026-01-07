@echo off
REM Build-Script für Speech Input App
REM Erstellt eine standalone ausführbare Datei mit PyInstaller

echo ========================================
echo Speech Input App - Build Script
echo ========================================
echo.

REM Prüfe ob PyInstaller installiert ist
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller nicht gefunden. Installiere PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo Fehler beim Installieren von PyInstaller!
        pause
        exit /b 1
    )
)

echo.
echo Prüfe Dependencies...
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo PyQt6 nicht gefunden. Bitte installieren Sie die Requirements:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Starte Build-Prozess...
echo.

REM Wechsle ins speech_input_app Verzeichnis
cd /d "%~dp0"

REM Lösche alte Build-Artefakte
if exist "build" (
    echo Lösche alte Build-Artefakte...
    rmdir /s /q build
)
if exist "dist" (
    echo Lösche alte Dist-Artefakte...
    rmdir /s /q dist
)
if exist "__pycache__" (
    echo Lösche __pycache__...
    rmdir /s /q __pycache__
)

REM Erstelle Executable mit PyInstaller
echo.
echo Erstelle Executable...
pyinstaller speech_input.spec

if errorlevel 1 (
    echo.
    echo ========================================
    echo FEHLER beim Build!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build erfolgreich!
echo ========================================
echo.
echo Die ausführbare Datei befindet sich in:
echo   dist\speech_input.exe
echo.
echo Sie können die Datei jetzt verwenden oder kopieren.
echo.

REM Frage ob dist-Verzeichnis geöffnet werden soll
set /p open="Dist-Verzeichnis öffnen? (j/n): "
if /i "%open%"=="j" (
    explorer dist
)

pause

