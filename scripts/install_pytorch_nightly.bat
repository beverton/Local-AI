@echo off
echo ========================================
echo PyTorch NIGHTLY mit CUDA 12.9 installieren
echo ========================================
echo.
echo WARNUNG: Dies wird die aktuelle PyTorch-Installation ersetzen!
echo.
echo HINWEIS: Nightly-Builds sind Vorabversionen und koennen:
echo   - Weniger stabil sein als stabile Versionen
echo   - Unerwartete Fehler verursachen
echo   - Nicht mit allen Bibliotheken kompatibel sein
echo.
echo Vorteile:
echo   - Neueste Features und Verbesserungen
echo   - Beste Unterstützung fuer RTX 50-Serie (Blackwell)
echo   - CUDA 12.9 Unterstützung
echo.
echo Empfohlen fuer: RTX 50-Serie GPUs, wenn stabile Version Probleme hat
echo.
pause

REM Pruefe GPU-Informationen
echo.
echo [INFO] Pruefe GPU-Informationen...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNUNG] nvidia-smi nicht gefunden.
) else (
    REM Verwende einfache nvidia-smi Ausgabe
    nvidia-smi -L 2>nul | findstr /C:"RTX 50" >nul
    if not errorlevel 1 (
        echo [INFO] RTX 50-Serie Blackwell-Architektur gefunden
    )
    nvidia-smi -L 2>nul
)

echo.
echo [1/3] Deinstalliere alte PyTorch-Version...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2/3] Installiere PyTorch Nightly mit CUDA 12.9...
echo Bitte warten Sie, dies kann einige Minuten dauern...
echo.
echo Installiere torch (Hauptpaket)...
echo Hinweis: Falls die Datei bereits heruntergeladen wurde, wird der Cache verwendet.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129
if errorlevel 1 (
    echo.
    echo [FEHLER] torch Installation fehlgeschlagen!
    echo.
    echo Moegliche Ursachen:
    echo   - Kein Speicherplatz mehr auf dem Laufwerk (No space left on device)
    echo   - Netzwerkprobleme
    echo   - Nightly-Builds nicht verfuegbar
    echo   - Python-Version nicht kompatibel
    echo.
    echo TIPPS:
    echo   - Pruefen Sie den freien Speicherplatz auf C:\
    echo   - Loeschen Sie temporaere Dateien oder andere Programme
    echo   - Pip verwendet automatisch den Cache - kein erneutes Herunterladen noetig
    echo   - Versuchen Sie die Installation erneut nach Freigabe von Speicherplatz
    echo.
    echo Alternativ: Versuchen Sie die stabile Version: scripts\install_pytorch_cuda.bat
    pause
    exit /b 1
)

echo.
echo Versuche torchvision zu installieren (optional)...
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu129 2>nul
if errorlevel 1 (
    echo [INFO] torchvision nicht verfuegbar fuer Nightly CUDA 12.9 - wird uebersprungen
    echo [INFO] torch allein ist ausreichend fuer AI-Modelle
) else (
    echo [OK] torchvision installiert
)

echo.
echo Versuche torchaudio zu installieren (optional)...
pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129 2>nul
if errorlevel 1 (
    echo [INFO] torchaudio nicht verfuegbar - wird uebersprungen (nicht erforderlich)
) else (
    echo [OK] torchaudio installiert
)

echo.
echo [3/3] Pruefe Installation...
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA verfuegbar:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo [WARNUNG] Konnte PyTorch-Installation nicht pruefen.
)

echo.
echo ========================================
echo Installation abgeschlossen!
echo ========================================
echo.
echo Bitte pruefen Sie die Installation mit: scripts\check_gpu.bat
echo.
echo WICHTIG: Falls Probleme auftreten, koennen Sie zurueck zur stabilen Version:
echo          scripts\install_pytorch_cuda.bat
echo.
pause

