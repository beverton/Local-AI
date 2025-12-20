"""
GPU-Check Skript - Prüft GPU-Unterstützung für AI-Modelle
"""
import sys
import subprocess

def check_python_version():
    """Prüft Python-Version"""
    print("=" * 60)
    print("1. PYTHON-VERSION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Python Pfad: {sys.executable}")
    print()

def check_torch():
    """Prüft PyTorch Installation und CUDA-Unterstützung"""
    print("=" * 60)
    print("2. PYTORCH-INSTALLATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch installiert: Version {torch.__version__}")
        
        # Prüfe ob CUDA verfügbar ist
        cuda_available = torch.cuda.is_available()
        print(f"[OK] CUDA verfügbar: {cuda_available}")
        
        if cuda_available:
            print(f"  - CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  - Anzahl GPUs: {torch.cuda.device_count()}")
            
            # Pruefe Blackwell-Architektur (sm_120) Kompatibilitaet
            blackwell_detected = False
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                compute_cap = f"{props.major}.{props.minor}"
                print(f"\n  GPU {i}:")
                print(f"    - Name: {torch.cuda.get_device_name(i)}")
                print(f"    - Speicher gesamt: {props.total_memory / 1024**3:.2f} GB")
                print(f"    - Compute Capability: {compute_cap}")
                
                # Erkenne Blackwell-Architektur (sm_120 = major 12, minor 0)
                if props.major == 12 and props.minor == 0:
                    blackwell_detected = True
                    print(f"    - [BLACKWELL] RTX 50-Serie erkannt!")
                    # Pruefe ob PyTorch-Version Blackwell unterstuetzt
                    torch_version_parts = torch.__version__.split('.')
                    torch_major = int(torch_version_parts[0])
                    torch_minor = int(torch_version_parts[1]) if len(torch_version_parts) > 1 else 0
                    
                    if torch_major > 2 or (torch_major == 2 and torch_minor >= 7):
                        cuda_version_parts = torch.version.cuda.split('.')
                        cuda_major = int(cuda_version_parts[0])
                        cuda_minor = int(cuda_version_parts[1]) if len(cuda_version_parts) > 1 else 0
                        
                        if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                            print(f"    - [OK] PyTorch {torch.__version__} mit CUDA {torch.version.cuda} unterstuetzt Blackwell")
                        else:
                            print(f"    - [WARNUNG] CUDA {torch.version.cuda} ist zu alt fuer Blackwell")
                            print(f"    - Benoetigt: CUDA 12.8 oder hoeher")
                    else:
                        print(f"    - [WARNUNG] PyTorch {torch.__version__} ist zu alt fuer Blackwell")
                        print(f"    - Benoetigt: PyTorch 2.7.0+ mit CUDA 12.8+")
        else:
            print("  [WARNUNG] CUDA ist NICHT verfuegbar!")
            print("  -> PyTorch wurde wahrscheinlich ohne CUDA-Unterstuetzung installiert")
            print("  -> Die GPU wird nicht verwendet, alles laeuft auf CPU")
        
        print()
        return cuda_available
        
    except ImportError:
        print("[FEHLER] PyTorch ist NICHT installiert!")
        print()
        return False

def check_nvidia_driver():
    """Prüft NVIDIA Treiber (Windows)"""
    print("=" * 60)
    print("3. NVIDIA-TREIBER")
    print("=" * 60)
    
    try:
        # Versuche nvidia-smi aufzurufen
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("[OK] NVIDIA Treiber gefunden:")
            print()
            # Zeige nur die ersten Zeilen (GPU-Info)
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines[:15]):  # Erste 15 Zeilen
                if line.strip():
                    print(f"  {line}")
            print()
            return True
        else:
            print("[FEHLER] nvidia-smi konnte nicht ausgefuehrt werden")
            print()
            return False
            
    except FileNotFoundError:
        print("[FEHLER] nvidia-smi nicht gefunden!")
        print("  -> NVIDIA Treiber sind moeglicherweise nicht installiert")
        print("  -> Oder nvidia-smi ist nicht im PATH")
        print()
        return False
    except Exception as e:
        print(f"[FEHLER] Fehler beim Pruefen der Treiber: {e}")
        print()
        return False

def check_cuda_toolkit():
    """Prüft ob CUDA Toolkit installiert ist"""
    print("=" * 60)
    print("4. CUDA TOOLKIT")
    print("=" * 60)
    
    try:
        # Prüfe nvcc (CUDA Compiler)
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("[OK] CUDA Toolkit gefunden:")
            print()
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line}")
            print()
            return True
        else:
            print("[FEHLER] nvcc nicht gefunden")
            print("  -> CUDA Toolkit ist moeglicherweise nicht installiert")
            print("  -> Oder nicht im PATH")
            print()
            return False
            
    except FileNotFoundError:
        print("[FEHLER] nvcc nicht gefunden!")
        print("  -> CUDA Toolkit ist moeglicherweise nicht installiert")
        print("  -> Hinweis: Fuer PyTorch ist das CUDA Toolkit NICHT zwingend erforderlich")
        print("  -> PyTorch enthaelt bereits die benoetigten CUDA-Bibliotheken")
        print()
        return False
    except Exception as e:
        print(f"[FEHLER] Fehler beim Pruefen: {e}")
        print()
        return False

def get_pytorch_install_command():
    """Gibt den richtigen PyTorch-Installationsbefehl basierend auf CUDA-Version und GPU-Architektur"""
    print("=" * 60)
    print("5. INSTALLATIONS-EMPFEHLUNG")
    print("=" * 60)
    
    blackwell_detected = False
    
    # Pruefe ob Blackwell-Architektur vorhanden ist
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.major == 12 and props.minor == 0:
                    blackwell_detected = True
                    break
            
            if blackwell_detected:
                cuda_version = torch.version.cuda
                cuda_version_parts = cuda_version.split('.')
                cuda_major = int(cuda_version_parts[0])
                cuda_minor = int(cuda_version_parts[1]) if len(cuda_version_parts) > 1 else 0
                
                if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                    print(f"[OK] PyTorch mit CUDA {cuda_version} ist bereits installiert!")
                    print("     Blackwell-Architektur wird unterstuetzt.")
                    print()
                    return None
                else:
                    print(f"[WARNUNG] PyTorch mit CUDA {cuda_version} erkannt")
                    print("          ABER: CUDA-Version ist zu alt fuer Blackwell (benoetigt 12.8+)")
                    print()
            else:
                cuda_version = torch.version.cuda
                print(f"[OK] PyTorch mit CUDA {cuda_version} ist bereits installiert!")
                print()
                return None
    except:
        pass
    
    # Pruefe nvidia-smi fuer CUDA-Version und GPU-Name
    gpu_name = ""
    cuda_driver_version = ""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Suche nach GPU-Name (RTX 50-Serie)
            for line in output.split('\n'):
                if "RTX 50" in line or "GeForce RTX 50" in line:
                    # Extrahiere GPU-Name
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "RTX" in part and i + 1 < len(parts):
                            gpu_name = " ".join(parts[parts.index(part):parts.index(part)+3])
                            break
                    if "5060" in line or "5070" in line or "5080" in line or "5090" in line:
                        blackwell_detected = True
                
                if "CUDA Version:" in line:
                    cuda_driver_version = line.split("CUDA Version:")[1].strip().split()[0]
                    cuda_major = int(cuda_driver_version.split('.')[0])
                    print(f"[OK] Gefundene CUDA-Version (Treiber): {cuda_driver_version}")
                    if gpu_name:
                        print(f"[OK] Gefundene GPU: {gpu_name}")
                    print()
                    
                    # Spezielle Empfehlung fuer Blackwell/RTX 50-Serie
                    if blackwell_detected:
                        print("=" * 60)
                        print("BLACKWELL-ARCHITEKTUR (RTX 50-Serie) ERKANNT!")
                        print("=" * 60)
                        print()
                        print("WICHTIG: RTX 50-Serie benoetigt PyTorch 2.7.0+ mit CUDA 12.8+")
                        print()
                        print("Option 1: Stabile Version (empfohlen)")
                        print("  pip uninstall torch torchvision torchaudio -y")
                        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
                        print()
                        print("  HINWEIS: Falls cu128 nicht verfuegbar ist, verwenden Sie:")
                        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                        print()
                        print("Option 2: Nightly-Build (neueste Features, CUDA 12.9)")
                        print("  pip uninstall torch torchvision torchaudio -y")
                        print("  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129")
                        print()
                        print("  Verwenden Sie: scripts\\install_pytorch_nightly.bat")
                        print()
                        return cuda_major
                    else:
                        # Normale Empfehlung fuer andere GPUs
                        if cuda_major >= 12:
                            print("Empfohlener Installationsbefehl fuer CUDA 12.x:")
                            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                            print()
                        elif cuda_major >= 11:
                            print("Empfohlener Installationsbefehl fuer CUDA 11.8:")
                            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                            print()
                        else:
                            print("[WARNUNG] Alte CUDA-Version erkannt. Bitte aktualisiere deine Treiber.")
                            print()
                        return cuda_major
    except:
        pass
    
    # Fallback: Zeige allgemeine Optionen
    print("Allgemeine Installationsoptionen:")
    print()
    print("Fuer RTX 50-Serie (Blackwell, CUDA 12.8+):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    print()
    print("Fuer andere neue GPUs (CUDA 12.4):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print()
    print("Fuer aeltere GPUs (CUDA 11.8):")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("Fuer CPU-only (falls keine GPU verfuegbar):")
    print("  pip install torch torchvision torchaudio")
    print()
    
    return None

def main():
    """Hauptfunktion"""
    print()
    print("=" * 60)
    print(" " * 15 + "GPU-UNTERSTUETZUNG PRUEFEN")
    print("=" * 60)
    print()
    
    check_python_version()
    cuda_available = check_torch()
    driver_available = check_nvidia_driver()
    cuda_toolkit_available = check_cuda_toolkit()
    get_pytorch_install_command()
    
    print("=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    
    if cuda_available:
        print("[OK] GPU-UNTERSTUETZUNG: AKTIV")
        print("  -> Ihre GPU wird fuer AI-Modelle verwendet")
        print("  -> Performance sollte optimal sein")
    else:
        print("[FEHLER] GPU-UNTERSTUETZUNG: INAKTIV")
        print("  -> AI-Modelle laufen auf CPU (langsamer)")
        
        if driver_available:
            print("  -> NVIDIA Treiber sind installiert")
            print("  -> ABER: PyTorch hat keine CUDA-Unterstuetzung")
            print("  -> LOESUNG: PyTorch mit CUDA neu installieren (siehe oben)")
        else:
            print("  -> NVIDIA Treiber moeglicherweise nicht installiert")
            print("  -> LOESUNG: Installiere zuerst NVIDIA Treiber")
    
    print()
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()

