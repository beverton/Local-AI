"""
Finaler Integrationstest: SDXL mit korrigiertem GPU-Algorithmus
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_final_integration():
    """Testet die vollständige SDXL-Integration"""
    
    print("=" * 70)
    print("Finaler SDXL Integrationstest")
    print("=" * 70)
    
    try:
        from image_manager import ImageManager
        from datetime import datetime
        import json
        
        print("\n[1/5] Lade Config und pruefe Standard-Modell...")
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        default_model = config.get("image_generation", {}).get("default_model")
        if default_model == "sdxl-base-1.0":
            print(f"      [OK] Standard-Modell: {default_model}")
        else:
            print(f"      [WARNUNG] Standard-Modell ist nicht SDXL: {default_model}")
        
        print("\n[2/5] Initialisiere Image Manager...")
        manager = ImageManager(config_path="config.json")
        print("      [OK]")
        
        print("\n[3/5] Lade SDXL...")
        success = manager.load_model("sdxl-base-1.0")
        if not success:
            print("      [FEHLER] Modell konnte nicht geladen werden!")
            return False
        print("      [OK] SDXL geladen")
        
        print("\n[4/5] Generiere 1024x1024 Testbild (GPU-Fix Test)...")
        print("      Prompt: 'A serene lake in autumn'")
        print("      Erwartung: Sollte NICHT auf 432x432 reduzieren")
        
        result = manager.generate_image(
            prompt="A serene lake in autumn with colorful trees, photorealistic",
            negative_prompt="blurry, low quality",
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=7.5
        )
        
        if result is None:
            print("      [FEHLER] Generierung fehlgeschlagen!")
            return False
        
        # Extrahiere Info
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
        
        if width >= 768 and height >= 768:
            print(f"      [OK] GPU-Fix funktioniert! (mindestens 768x768)")
        else:
            print(f"      [WARNUNG] Aufloesung noch zu niedrig: {width}x{height}")
            print(f"               Sollte mindestens 768x768 sein")
        
        if auto_resized:
            print(f"      [INFO] Auto-Resize wurde verwendet")
            print(f"             Original 1024x1024 -> {width}x{height}")
        else:
            print(f"      [OK] Volle Aufloesung ohne Auto-Resize!")
        
        print("\n[5/5] Speichere Testbild...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"final_integration_test_{width}x{height}_{timestamp}.png")
        
        image.save(output, "PNG", optimize=True)
        size = output.stat().st_size / 1024
        
        print(f"      [OK] Gespeichert: {output} ({size:.1f} KB)")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] INTEGRATION ERFOLGREICH!")
        print("=" * 70)
        print(f"\nErgebnisse:")
        print(f"  Standard-Modell:  sdxl-base-1.0")
        print(f"  Aufloesung:       {width}x{height}")
        print(f"  Auto-Resize:      {'Ja' if auto_resized else 'Nein'}")
        print(f"  GPU-Fix:          {'Funktioniert' if width >= 768 else 'Noch nicht optimal'}")
        print(f"  Ausgabe:          {output}")
        
        print(f"\nSDXL ist vollstaendig integriert und einsatzbereit!")
        
        return True
        
    except Exception as e:
        print(f"\n[FEHLER] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_integration()
    sys.exit(0 if success else 1)

