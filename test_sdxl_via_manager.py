"""
Test SDXL über den Image Manager (optimal)
"""
import sys
import os
from pathlib import Path

# Füge backend zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_sdxl_via_manager():
    """Testet SDXL über den optimierten Image Manager"""
    
    print("=" * 70)
    print("SDXL Test via Image Manager")
    print("=" * 70)
    
    try:
        from image_manager import ImageManager
        from PIL import Image
        from datetime import datetime
        
        print("\n[1/4] Initialisiere Image Manager...")
        manager = ImageManager(config_path="config.json")
        print("      [OK]")
        
        print("\n[2/4] Lade SDXL-Modell...")
        print("      Modell-ID: sdxl-base-1.0")
        success = manager.load_model("sdxl-base-1.0")
        
        if not success:
            print("      [FEHLER] Modell konnte nicht geladen werden!")
            return False
        
        print("      [OK] Modell geladen")
        
        print("\n[3/4] Generiere Testbild...")
        print("      Prompt: 'A majestic lion in the savanna'")
        print("      Size: 1024x1024")
        
        result = manager.generate_image(
            prompt="A majestic lion in the savanna at sunset, photorealistic, highly detailed",
            negative_prompt="blurry, low quality",
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        if result is None:
            print("      [FEHLER] Bildgenerierung fehlgeschlagen!")
            return False
        
        # Extrahiere Bild
        if isinstance(result, dict):
            image = result.get("image")
            width = result.get("width", 1024)
            height = result.get("height", 1024)
            auto_resized = result.get("auto_resized", False)
            cpu_offload = result.get("cpu_offload_used", False)
        else:
            image = result
            width, height = 1024, 1024
            auto_resized = False
            cpu_offload = False
        
        print(f"      [OK] Bild generiert ({width}x{height})")
        if auto_resized:
            print(f"      [INFO] Aufloesung wurde automatisch angepasst")
        if cpu_offload:
            print(f"      [INFO] CPU-Offload wurde verwendet")
        
        print("\n[4/4] Speichere Bild...")
        
        # Speichere an zwei Orten
        output_dir = Path("G:/KI Modelle/image/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path1 = output_dir / f"sdxl_manager_test_{timestamp}.png"
        output_path2 = Path("test_sdxl_output.png")
        
        image.save(output_path1, "PNG", optimize=True)
        image.save(output_path2, "PNG", optimize=True)
        
        size1 = output_path1.stat().st_size / 1024
        size2 = output_path2.stat().st_size / 1024
        
        print(f"      [OK] Gespeichert:")
        print(f"           1. {output_path1} ({size1:.1f} KB)")
        print(f"           2. {output_path2} ({size2:.1f} KB)")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] SDXL IST EINSATZBEREIT!")
        print("=" * 70)
        print(f"\nErgebnis:")
        print(f"  Aufloesung:  {width}x{height}")
        print(f"  Ausgabe 1:   {output_path1}")
        print(f"  Ausgabe 2:   {output_path2}")
        print(f"\nSDXL kann jetzt in Ihrem System verwendet werden!")
        print(f"Verwenden Sie die Modell-ID: sdxl-base-1.0")
        
        return True
        
    except Exception as e:
        print(f"\n[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sdxl_via_manager()
    sys.exit(0 if success else 1)

