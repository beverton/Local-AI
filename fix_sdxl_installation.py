"""
Fix SDXL Installation - Lädt aus Cache und speichert korrekt
"""
import os
import sys
import json
from pathlib import Path

def fix_sdxl_installation():
    """Lädt SDXL aus Cache und speichert es korrekt"""
    
    # Zielverzeichnis
    model_dir = Path("G:/KI Modelle/image/sdxl-base-1.0")
    
    # Cache-Verzeichnis
    cache_dir = Path("G:/KI Modelle/KI-Temp/huggingface_cache")
    
    # Setze Umgebungsvariablen
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HF_HUB_CACHE'] = str(cache_dir / "hub")
    
    print("=" * 70)
    print("SDXL Installation Fix")
    print("=" * 70)
    print(f"Lade aus Cache: {cache_dir}")
    print(f"Speichere nach: {model_dir}")
    print()
    
    try:
        print("1. Importiere diffusers...")
        from diffusers import DiffusionPipeline
        import torch
        print("[OK] diffusers importiert")
        
        print("\n2. Lade Pipeline aus lokalem Cache (sehr schnell)...")
        # local_files_only=True stellt sicher, dass nur aus Cache geladen wird
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,  # Nur aus Cache
            use_safetensors=True
        )
        print("[OK] Pipeline geladen")
        
        print("\n3. Speichere Pipeline...")
        # Erstelle Zielverzeichnis neu
        if model_dir.exists():
            import shutil
            print(f"   Loesche altes Verzeichnis...")
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Speichere
        pipeline.save_pretrained(
            str(model_dir),
            safe_serialization=True
        )
        print(f"[OK] Pipeline gespeichert in: {model_dir}")
        
        # Prüfe Dateigröße
        total_size = sum(
            f.stat().st_size 
            for f in model_dir.rglob('*') 
            if f.is_file()
        )
        size_gb = total_size / (1024**3)
        print(f"[OK] Gesamtgroesse: {size_gb:.2f} GB")
        
        # Aktualisiere config.json
        print("\n4. Aktualisiere config.json...")
        config_path = Path("config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Füge SDXL hinzu
        config["models"]["sdxl-base-1.0"] = {
            "name": "Stable Diffusion XL Base 1.0",
            "path": str(model_dir).replace('/', '\\'),
            "type": "image",
            "description": "Bildgenerierung mit SDXL - frei nutzbar, stabil, 7GB VRAM"
        }
        
        # Speichere
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] config.json aktualisiert")
        
        print("\n" + "=" * 70)
        print("[OK] INSTALLATION ERFOLGREICH!")
        print("=" * 70)
        print("\nDas Modell ist jetzt einsatzbereit:")
        print("- Modell-ID: sdxl-base-1.0")
        print(f"- Pfad: {model_dir}")
        print(f"- Groesse: {size_gb:.2f} GB")
        print("- VRAM: ~7 GB")
        print("- Lizenz: CreativeML OpenRAIL++ (kommerziell nutzbar)")
        
        return True
        
    except Exception as e:
        print(f"\n[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_sdxl_installation()
    sys.exit(0 if success else 1)

