@echo off
echo ========================================
echo PyTorch mit CUDA-Unterstuetzung installieren
echo ========================================
echo.
echo WARNUNG: Dies wird die aktuelle PyTorch-Installation ersetzen!
echo.

REM Pruefe GPU-Informationen
echo [INFO] Pruefe GPU-Informationen...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNUNG] nvidia-smi nicht gefunden. GPU-Informationen koennen nicht geprueft werden.
    echo.
) else (
    REM Pruefe auf RTX 50-Serie
    nvidia-smi -L 2>nul | findstr /C:"RTX 50" >nul
    if not errorlevel 1 (
        echo [INFO] RTX 50-Serie Blackwell-Architektur gefunden
        echo [INFO] Benoetigt: PyTorch 2.7.0+ mit CUDA 12.8+
    )
    echo [OK] GPU-Informationen:
    nvidia-smi -L 2>nul
)

echo.
echo Installation: PyTorch mit CUDA 12.8 (fuer RTX 50-Serie)
echo Falls CUDA 12.8 nicht verfuegbar ist, wird CUDA 12.4 versucht.
echo.
pause

echo.
echo [1/3] Deinstalliere alte PyTorch-Version...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2/3] Installiere PyTorch mit CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo [WARNUNG] CUDA 12.8 nicht verfuegbar, versuche CUDA 12.4...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo.
        echo [FEHLER] Installation fehlgeschlagen!
        echo Bitte pruefen Sie die PyTorch-Website fuer verfuegbare Versionen.
        pause
        exit /b 1
    )
)

echo.
echo [3/3] Pruefe Installation...
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA verfuegbar:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
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
echo HINWEIS: Falls Sie eine RTX 50-Serie GPU haben und Probleme auftreten,
echo          versuchen Sie die Nightly-Version: scripts\install_pytorch_nightly.bat
echo.
pause

