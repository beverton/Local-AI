"""
Finaler SDXL Test mit garantierter Bildspeicherung
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

def test_sdxl():
    """Testet SDXL und speichert garantiert ein Bild"""
    
    print("=" * 70)
    print("SDXL Bildgenerierungs-Test (Final)")
    print("=" * 70)
    
    # Ausgabepfad
    output_dir = Path("G:/KI Modelle/image/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"sdxl_test_{timestamp}.png"
    
    try:
        # 1. Setup
        cache_dir = Path("G:/KI Modelle/KI-Temp/huggingface_cache")
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['HF_HUB_CACHE'] = str(cache_dir / "hub")
        
        print("\n[1/6] Importiere Bibliotheken...")
        from diffusers import DiffusionPipeline
        import torch
        from PIL import Image
        print("      [OK]")
        
        # 2. CUDA Check
        print("\n[2/6] Pruefe GPU...")
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"      Device: {device}")
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"      GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            
            # CUDA-Stabilität
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 3. Lade Modell
        print("\n[3/6] Lade SDXL-Modell...")
        model_path = Path("G:/KI Modelle/image/sdxl-base-1.0")
        
        start = time.time()
        
        # Lade alles in float32 (stabil, aber langsamer)
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
            use_safetensors=True,
            local_files_only=True
        )
        
        # GPU-Transfer - alles in float32 bleibt
        if cuda_available:
            pipeline = pipeline.to(device)
            # Optimierungen
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            print("    [INFO] Pipeline in float32 (stabil, aber langsamer)")
        
        load_time = time.time() - start
        print(f"      Geladen in {load_time:.1f}s")
        
        if cuda_available:
            mem_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"      VRAM: {mem_used:.2f} GB")
        
        # 4. Generiere Bild
        print("\n[4/6] Generiere Testbild...")
        print("      Prompt: 'A beautiful sunset over mountains'")
        print("      Size: 1024x1024, Steps: 20")
        
        start = time.time()
        with torch.no_grad():
            result = pipeline(
                prompt="A beautiful sunset over mountains, vibrant colors, photorealistic, highly detailed",
                negative_prompt="blurry, low quality, distorted, artifacts",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=1024,
                height=1024
            )
        
        gen_time = time.time() - start
        print(f"      Generiert in {gen_time:.1f}s")
        
        # 5. Extrahiere Bild
        print("\n[5/6] Extrahiere Bild...")
        if isinstance(result, dict) and "images" in result:
            image = result["images"][0]
        elif hasattr(result, 'images'):
            image = result.images[0]
        else:
            # Fallback: result ist direkt das Bild
            image = result[0] if isinstance(result, (list, tuple)) else result
        
        if image is None:
            print("      [FEHLER] Kein Bild erhalten!")
            return False
        
        print(f"      Bild: {image.size[0]}x{image.size[1]} Pixel")
        
        # 6. Speichere Bild
        print(f"\n[6/6] Speichere Bild...")
        print(f"      Pfad: {output_path}")
        
        # Garantierte Speicherung mit mehreren Formaten
        image.save(output_path, "PNG", optimize=True)
        
        # Zusätzliche Kopie im Projektverzeichnis
        local_copy = Path("test_sdxl_output.png")
        image.save(local_copy, "PNG", optimize=True)
        
        # Prüfe beide Dateien
        if output_path.exists() and local_copy.exists():
            size1 = output_path.stat().st_size / 1024
            size2 = local_copy.stat().st_size / 1024
            print(f"      [OK] Gespeichert ({size1:.1f} KB)")
            print(f"      Kopie: {local_copy} ({size2:.1f} KB)")
        else:
            print("      [FEHLER] Speichern fehlgeschlagen!")
            return False
        
        # Erfolg!
        print("\n" + "=" * 70)
        print("[SUCCESS] SDXL TEST ERFOLGREICH!")
        print("=" * 70)
        print(f"\nErgebnisse:")
        print(f"  Ladezeit:          {load_time:.1f}s")
        print(f"  Generierungszeit:  {gen_time:.1f}s")
        print(f"  Gesamtzeit:        {load_time + gen_time:.1f}s")
        if cuda_available:
            print(f"  VRAM-Nutzung:      {mem_used:.2f} GB")
        print(f"\nAusgabe-Dateien:")
        print(f"  1. {output_path}")
        print(f"  2. {local_copy}")
        print(f"\nFazit: SDXL ist einsatzbereit! ✓")
        
        # Cleanup
        del pipeline
        if cuda_available:
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        
        print("\nDiagnose:")
        if "out of memory" in str(e).lower():
            print("  -> GPU hat nicht genug Speicher")
            print("     Loesungen: Kleinere Aufloesung, CPU-Offload")
        elif "engine" in str(e).lower():
            print("  -> CUDA-Engine Problem")
            print("     Loesung: VAE auf float32 setzen (bereits implementiert)")
        else:
            print("  -> Unbekannter Fehler")
        
        return False

if __name__ == "__main__":
    success = test_sdxl()
    sys.exit(0 if success else 1)

