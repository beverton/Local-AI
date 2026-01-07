"""
Test SDXL mit 1024x1024 Auflösung
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_sdxl_1024():
    """Testet SDXL mit voller 1024x1024 Auflösung"""
    
    print("=" * 70)
    print("SDXL Test - 1024x1024 Auflösung")
    print("=" * 70)
    
    try:
        from image_manager import ImageManager
        from datetime import datetime
        
        print("\n[1/4] Initialisiere Image Manager...")
        manager = ImageManager(config_path="config.json")
        
        print("\n[2/4] Lade SDXL...")
        success = manager.load_model("sdxl-base-1.0")
        if not success:
            print("[FEHLER] Modell konnte nicht geladen werden!")
            return False
        print("      [OK]")
        
        print("\n[3/4] Generiere 1024x1024 Bild...")
        print("      Prompt: 'A futuristic cityscape at night'")
        
        result = manager.generate_image(
            prompt="A futuristic cityscape at night with neon lights, cyberpunk style, highly detailed, photorealistic, 8k quality",
            negative_prompt="blurry, low quality, distorted, oversaturated",
            width=1024,
            height=1024,
            num_inference_steps=25,
            guidance_scale=7.5
        )
        
        if result is None:
            print("[FEHLER] Generierung fehlgeschlagen!")
            return False
        
        # Extrahiere Bild und Info
        if isinstance(result, dict):
            image = result.get("image")
            width = result.get("width", 1024)
            height = result.get("height", 1024)
            auto_resized = result.get("auto_resized", False)
        else:
            image = result
            width, height = 1024, 1024
            auto_resized = False
        
        print(f"      [OK] Generiert: {width}x{height}")
        if auto_resized:
            print(f"      [INFO] Aufloesung angepasst wegen GPU-Speicher")
        
        print("\n[4/4] Speichere Bild...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere an zwei Orten
        output1 = Path(f"G:/KI Modelle/image/test_outputs/sdxl_1024_{timestamp}.png")
        output2 = Path("test_sdxl_1024.png")
        
        output1.parent.mkdir(parents=True, exist_ok=True)
        
        image.save(output1, "PNG", optimize=True)
        image.save(output2, "PNG", optimize=True)
        
        size1 = output1.stat().st_size / 1024
        size2 = output2.stat().st_size / 1024
        
        print(f"      [OK] Gespeichert:")
        print(f"           1. {output1} ({size1:.1f} KB)")
        print(f"           2. {output2} ({size2:.1f} KB)")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] 1024x1024 TESTBILD ERFOLGREICH!")
        print("=" * 70)
        print(f"\nErgebnis:")
        print(f"  Aufloesung:     {width}x{height}")
        print(f"  Auto-Resize:    {'Ja' if auto_resized else 'Nein'}")
        print(f"  Ausgabe 1:      {output1}")
        print(f"  Ausgabe 2:      {output2}")
        
        return True
        
    except Exception as e:
        print(f"\n[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sdxl_1024()
    sys.exit(0 if success else 1)

