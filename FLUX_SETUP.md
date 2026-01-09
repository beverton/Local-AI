# Flux Model Setup - Separate CUDA Environment

## Problem

Flux-Modelle können abstürzen, wenn die CUDA-Version nicht kompatibel ist. Dies kann passieren, wenn:
- Die aktuelle CUDA-Version zu alt ist (z.B. CUDA 11.7 oder älter)
- Es Konflikte zwischen verschiedenen CUDA-Versionen gibt
- xformers/Triton nicht korrekt für die CUDA-Version installiert sind

## Lösung: Separate Python-Umgebung für Flux

### Option 1: Separate virtuelle Umgebung (Empfohlen)

1. **Erstelle eine neue virtuelle Umgebung für Flux:**
```bash
# Im Projekt-Verzeichnis
python -m venv venv_flux
```

2. **Aktiviere die Umgebung:**
```bash
# Windows
venv_flux\Scripts\activate

# Linux/Mac
source venv_flux/bin/activate
```

3. **Installiere PyTorch mit der richtigen CUDA-Version:**
```bash
# Für CUDA 12.4+ (empfohlen für Flux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Oder für CUDA 11.8 (falls 12.4 nicht verfügbar)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Installiere Flux-Dependencies:**
```bash
pip install diffusers>=0.21.0
pip install pillow>=10.0.0
pip install protobuf>=4.21.0
pip install xformers>=0.0.23
pip install sentencepiece>=0.1.99
```

5. **Prüfe die Installation:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### Option 2: Konfiguration für separate Python-Interpreter

Falls Sie bereits eine separate Umgebung haben, können Sie diese in der `config.json` konfigurieren:

```json
{
  "models": {
    "flux-1-schnell": {
      "name": "Flux 1 Schnell",
      "path": "G:\\KI Modelle\\image\\flux-1-schnell",
      "type": "image",
      "description": "Bildgenerierung mit Flux 1 Schnell",
      "python_interpreter": "G:\\path\\to\\venv_flux\\Scripts\\python.exe"
    }
  }
}
```

## CUDA-Version prüfen

Führen Sie aus, um die aktuelle CUDA-Version zu prüfen:
```bash
python scripts/check_gpu.py
```

Oder direkt in Python:
```python
import torch
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Empfohlene CUDA-Versionen für Flux

- **Minimum**: CUDA 11.8
- **Empfohlen**: CUDA 12.4 oder höher
- **Für RTX 50-Serie (Blackwell)**: CUDA 12.8+

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'triton'"
- **Lösung**: Triton ist optional. Die Warnung kann ignoriert werden, aber Flux läuft langsamer.

### Problem: Thread stürzt ab ohne Fehlermeldung
- **Mögliche Ursache**: Inkompatible CUDA-Version oder fehlerhafte xformers-Installation
- **Lösung**: 
  1. Prüfen Sie die CUDA-Version mit `scripts/check_gpu.py`
  2. Erstellen Sie eine separate Umgebung mit der richtigen CUDA-Version
  3. Deinstallieren und neu installieren Sie xformers: `pip uninstall xformers && pip install xformers`

### Problem: "Out of Memory" beim Laden
- **Lösung**: Flux benötigt viel GPU-Speicher. Stellen Sie sicher, dass:
  - Mindestens 8GB GPU-Speicher frei sind
  - Keine anderen GPU-intensiven Prozesse laufen
  - Sie `enable_model_cpu_offload()` verwenden (wird automatisch aktiviert bei "memory" Optimierung)

## Aktuelle CUDA-Version im System prüfen

Die Logs zeigen jetzt die CUDA-Version beim Laden von Flux-Modellen. Prüfen Sie die Logs in `.cursor/debug.log` für Details.











