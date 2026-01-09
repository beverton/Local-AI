"""
Minimaler Test für Modell-Laden mit disable_cpu_offload Fix
Testet ob das "meta" device Problem behoben ist
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

print("=" * 60)
print("TEST: Modell-Laden mit disable_cpu_offload Fix")
print("=" * 60)

# Lade Performance Settings
perf_settings = {}
perf_settings_path = os.path.join("data", "performance_settings.json")
if os.path.exists(perf_settings_path):
    with open(perf_settings_path, 'r', encoding='utf-8') as f:
        perf_settings = json.load(f)
    print(f"[OK] Performance Settings geladen: {perf_settings}")
else:
    print("[WARN] Performance Settings nicht gefunden, verwende Defaults")
    perf_settings = {"disable_cpu_offload": False}

disable_cpu_offload = perf_settings.get("disable_cpu_offload", False)
use_quantization = perf_settings.get("use_quantization", False)

print(f"\nSettings:")
print(f"  - disable_cpu_offload: {disable_cpu_offload}")
print(f"  - use_quantization: {use_quantization}")
print(f"  - CUDA verfügbar: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("[ERROR] CUDA nicht verfügbar - Test kann nicht durchgeführt werden")
    exit(1)

model_path = r"G:\KI Modelle\coding\qwen-2.5-7b-instruct"

if not os.path.exists(model_path):
    print(f"[ERROR] Modell-Pfad nicht gefunden: {model_path}")
    exit(1)

print(f"\n[1/4] Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("[OK] Tokenizer geladen")

print(f"\n[2/4] Bestimme device_map und max_memory...")

# Gleiche Logik wie in model_manager.py
device_map = None
max_memory = None

if torch.cuda.is_available():
    if use_quantization:
        device_map = "auto"
        if disable_cpu_offload:
            import torch.cuda as cuda
            gpu_memory = cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            max_memory_gb = int(gpu_memory * 0.9)
            max_memory = {0: f"{max_memory_gb}GB"}
            print(f"  - device_map: 'auto' (mit Quantisierung)")
            print(f"  - max_memory: {max_memory_gb}GB (90% der {gpu_memory:.1f}GB GPU)")
        else:
            print(f"  - device_map: 'auto' (mit Quantisierung, CPU-Offloading erlaubt)")
    elif disable_cpu_offload:
        # Versuche verschiedene Strategien
        # Strategie 1: device_map="cuda" (ohne Nummer) - sollte alles auf erste GPU laden
        device_map = "cuda"
        print(f"  - device_map: 'cuda' (ohne Nummer, sollte alles auf GPU laden)")
        # max_memory nicht setzen, da device_map="cuda" explizit alles auf GPU zwingt
    else:
        device_map = "auto"
        print(f"  - device_map: 'auto' (CPU-Offloading erlaubt)")

model_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": device_map,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True
}

if max_memory is not None:
    model_kwargs["max_memory"] = max_memory

print(f"\n[3/4] Lade Modell mit folgenden Parametern:")
print(f"  - torch_dtype: {model_kwargs['torch_dtype']}")
print(f"  - device_map: {model_kwargs['device_map']}")
if 'max_memory' in model_kwargs:
    print(f"  - max_memory: {model_kwargs['max_memory']}")
print(f"  - low_cpu_mem_usage: {model_kwargs['low_cpu_mem_usage']}")

try:
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    print("[OK] Modell geladen")
except Exception as e:
    print(f"[ERROR] Fehler beim Laden: {e}")
    exit(1)

print(f"\n[4/4] Validiere dass Modell auf echten GPU-Devices geladen wurde...")

meta_modules = []
cuda_modules = []
cpu_modules = []

for name, module in model.named_modules():
    try:
        first_param = next(module.parameters(), None)
        if first_param is not None:
            device_str = str(first_param.device)
            if device_str == "meta":
                meta_modules.append(name)
            elif "cuda" in device_str:
                cuda_modules.append(name)
            elif "cpu" in device_str:
                cpu_modules.append(name)
    except StopIteration:
        continue

print(f"\nErgebnis:")
print(f"  - Module auf CUDA: {len(cuda_modules)}")
print(f"  - Module auf CPU: {len(cpu_modules)}")
print(f"  - Module auf 'meta': {len(meta_modules)}")

if meta_modules:
    print(f"\n[ERROR] Modell wurde nicht korrekt geladen!")
    print(f"Folgende Module sind auf 'meta' device:")
    for mod in meta_modules[:10]:  # Zeige nur erste 10
        print(f"  - {mod}")
    if len(meta_modules) > 10:
        print(f"  ... und {len(meta_modules) - 10} weitere")
    exit(1)
elif cpu_modules and disable_cpu_offload:
    print(f"\n[WARN] {len(cpu_modules)} Module sind auf CPU, obwohl disable_cpu_offload aktiv ist!")
    print(f"Erste CPU-Module:")
    for mod in cpu_modules[:5]:
        print(f"  - {mod}")
else:
    print(f"\n[OK] Alle Module erfolgreich auf GPU geladen (kein 'meta' device)")
    if cpu_modules:
        print(f"[INFO] {len(cpu_modules)} Module auf CPU (erlaubt wenn disable_cpu_offload=false)")
    else:
        print(f"[OK] Keine Module auf CPU")

print("\n" + "=" * 60)
print("TEST ERFOLGREICH!")
print("=" * 60)
