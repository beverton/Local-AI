"""
Download Script für Stable Diffusion XL Base 1.0
Lädt das Modell von Hugging Face herunter und speichert es lokal.
"""
import os
import sys
import json
from pathlib import Path

def download_sdxl():
    """Lädt SDXL von Hugging Face herunter"""
    
    # Zielverzeichnis
    model_dir = Path("G:/KI Modelle/image/sdxl-base-1.0")
    
    # WICHTIG: Setze Hugging Face Cache auf G:\ Laufwerk (C:\ hat nicht genug Platz)
    cache_dir = Path("G:/KI Modelle/KI-Temp/huggingface_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Setze Umgebungsvariablen für Hugging Face Cache
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / "transformers")
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
    os.environ['DIFFUSERS_CACHE'] = str(cache_dir / "diffusers")
    os.environ['HF_HUB_CACHE'] = str(cache_dir / "hub")
    
    print(f"Cache-Verzeichnis: {cache_dir}")
    
    print("=" * 70)
    print("SDXL Base 1.0 Download")
    print("=" * 70)
    print(f"Zielverzeichnis: {model_dir}")
    print()
    
    # Erstelle Verzeichnis falls nicht vorhanden
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Verzeichnis erstellt/ueberprueft: {model_dir}")
    
    try:
        # Importiere diffusers
        print("\n1. Importiere diffusers...")
        from diffusers import DiffusionPipeline
        import torch
        print("[OK] diffusers erfolgreich importiert")
        
        # Prüfe CUDA
        cuda_available = torch.cuda.is_available()
        print(f"\n2. CUDA verfügbar: {cuda_available}")
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Download-Parameter
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"\n3. Starte Download von: {model_id}")
        print("   Dies kann einige Minuten dauern (ca. 6-7 GB)...")
        print()
        
        # Lade Pipeline herunter
        # torch_dtype=torch.float16 spart Speicherplatz (fp16 statt fp32)
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",  # Verwende fp16-Variante (kleiner)
            use_safetensors=True  # Sichereres Format
        )
        
        print("\n4. Download abgeschlossen! Speichere lokal...")
        
        # Speichere Pipeline lokal
        pipeline.save_pretrained(str(model_dir))
        
        print(f"[OK] Modell erfolgreich gespeichert in: {model_dir}")
        
        # Zeige Dateigröße
        total_size = sum(
            f.stat().st_size 
            for f in model_dir.rglob('*') 
            if f.is_file()
        )
        size_gb = total_size / (1024**3)
        print(f"[OK] Gesamtgroesse: {size_gb:.2f} GB")
        
        # Aktualisiere config.json
        print("\n5. Aktualisiere config.json...")
        config_path = Path("config.json")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Füge SDXL hinzu
            config["models"]["sdxl-base-1.0"] = {
                "name": "Stable Diffusion XL Base 1.0",
                "path": str(model_dir).replace('/', '\\'),
                "type": "image",
                "description": "Bildgenerierung mit SDXL - frei nutzbar, stabil, 7GB VRAM"
            }
            
            # Speichere aktualisierte Config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"[OK] config.json aktualisiert")
            print(f"   Modell-ID: sdxl-base-1.0")
        else:
            print("[WARNUNG] config.json nicht gefunden - bitte manuell hinzufuegen")
        
        print("\n" + "=" * 70)
        print("[OK] INSTALLATION ERFOLGREICH!")
        print("=" * 70)
        print("\nDas Modell kann jetzt verwendet werden:")
        print("- Modell-ID: sdxl-base-1.0")
        print(f"- Pfad: {model_dir}")
        print("- VRAM: ~7 GB")
        print("- Lizenz: CreativeML OpenRAIL++ (kommerziell nutzbar)")
        print("\nVorteile gegenueber FLUX:")
        print("  [+] Viel stabiler (keine CUDA-Probleme)")
        print("  [+] Weniger VRAM (~7GB statt ~30GB)")
        print("  [+] Schneller zu laden")
        print("  [+] Gut dokumentiert")
        
        return True
        
    except ImportError as e:
        print(f"\n[FEHLER] diffusers nicht installiert")
        print(f"   Fehler: {e}")
        print("\nBitte installieren Sie diffusers:")
        print("   pip install diffusers transformers accelerate safetensors")
        return False
        
    except Exception as e:
        print(f"\n[FEHLER] Fehler beim Download: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_sdxl()
    sys.exit(0 if success else 1)

