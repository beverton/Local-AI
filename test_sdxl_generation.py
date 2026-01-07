"""
Automatischer Test: SDXL Bildgenerierung
Testet ob SDXL erfolgreich ein Bild generieren kann
"""
import os
import sys
import time
from pathlib import Path

def test_sdxl_generation():
    """Testet SDXL Bildgenerierung"""
    
    print("=" * 70)
    print("SDXL Bildgenerierungs-Test")
    print("=" * 70)
    
    try:
        # 1. Setze Cache-Verzeichnis
        cache_dir = Path("G:/KI Modelle/KI-Temp/huggingface_cache")
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['HF_HUB_CACHE'] = str(cache_dir / "hub")
        
        print("\n1. Importiere erforderliche Bibliotheken...")
        from diffusers import DiffusionPipeline
        import torch
        from PIL import Image
        print("[OK] Bibliotheken importiert")
        
        # 2. Prüfe CUDA
        print("\n2. Pruefe CUDA-Verfuegbarkeit...")
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"[OK] Device: {device}")
        if cuda_available:
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"    GPU Speicher: {gpu_mem_total:.2f} GB")
            
            # CUDA-Optimierungen für Stabilität
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("    [OK] CUDA-Backend konfiguriert")
        
        # 3. Lade Pipeline
        print("\n3. Lade SDXL Pipeline...")
        model_path = Path("G:/KI Modelle/image/sdxl-base-1.0")
        
        if not model_path.exists():
            print(f"[FEHLER] Modell nicht gefunden: {model_path}")
            return False
        
        start_load = time.time()
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            use_safetensors=True,
            local_files_only=True
        )
        
        # Auf GPU verschieben
        if cuda_available:
            print("    Verschiebe Pipeline auf GPU...")
            pipeline = pipeline.to(device)
            
            # WICHTIG: VAE muss auf float32 laufen (bekanntes SDXL-Problem)
            # Verhindert "GET was unable to find an engine" Fehler
            print("    Konvertiere VAE zu float32 (verhindert CUDA-Fehler)...")
            pipeline.vae.to(torch.float32)
            
            # GPU-Optimierungen
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
                print("    [OK] Attention Slicing aktiviert")
        
        load_time = time.time() - start_load
        print(f"[OK] Pipeline geladen in {load_time:.2f} Sekunden")
        
        # GPU-Speicher nach Laden
        if cuda_available:
            gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"    GPU Speicher verwendet: {gpu_mem_used:.2f} GB")
        
        # 4. Generiere Testbild
        print("\n4. Generiere Testbild...")
        print("    Prompt: 'A majestic lion in the savanna at sunset'")
        print("    Resolution: 1024x1024")
        print("    Steps: 20")
        
        start_gen = time.time()
        
        with torch.no_grad():
            result = pipeline(
                prompt="A majestic lion in the savanna at sunset, photorealistic, detailed",
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=1024,
                height=1024,
                output_type="pil"
            )
        
        gen_time = time.time() - start_gen
        print(f"[OK] Bild generiert in {gen_time:.2f} Sekunden")
        
        # 5. Extrahiere und speichere Bild
        print("\n5. Speichere Testbild...")
        if isinstance(result, dict):
            images = result.get("images", [])
            if images:
                image = images[0]
            else:
                print("[FEHLER] Keine Bilder im Result")
                return False
        elif hasattr(result, 'images'):
            image = result.images[0]
        else:
            image = result
        
        # Speichere Testbild
        output_path = Path("test_sdxl_output.png")
        image.save(output_path)
        
        # Prüfe Dateigröße
        file_size = output_path.stat().st_size / 1024
        print(f"[OK] Bild gespeichert: {output_path}")
        print(f"    Dategroesse: {file_size:.2f} KB")
        print(f"    Bildgroesse: {image.size[0]}x{image.size[1]} Pixel")
        
        # 6. Zusammenfassung
        print("\n" + "=" * 70)
        print("[OK] TEST ERFOLGREICH!")
        print("=" * 70)
        print("\nErgebnisse:")
        print(f"  Device: {device}")
        print(f"  Ladezeit: {load_time:.2f}s")
        print(f"  Generierungszeit: {gen_time:.2f}s")
        print(f"  Gesamtzeit: {load_time + gen_time:.2f}s")
        if cuda_available:
            print(f"  GPU Speicher: {gpu_mem_used:.2f} GB")
        print(f"  Ausgabe: {output_path}")
        
        print("\nFazit:")
        print("  [+] SDXL funktioniert einwandfrei")
        print("  [+] Bildgenerierung erfolgreich")
        print("  [+] Bereit fuer den produktiven Einsatz")
        
        # Cleanup
        del pipeline
        if cuda_available:
            torch.cuda.empty_cache()
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[FEHLER] GPU Out of Memory")
        print(f"  Das Modell benoetigt mehr VRAM als verfuegbar")
        print(f"  Loesungsvorschlaege:")
        print(f"    - Kleinere Bildgroesse verwenden (z.B. 768x768)")
        print(f"    - CPU-Offloading aktivieren")
        print(f"    - Andere GPU-intensive Programme schliessen")
        return False
        
    except Exception as e:
        print(f"\n[FEHLER] Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sdxl_generation()
    sys.exit(0 if success else 1)

