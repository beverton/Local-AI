"""
Einfacher Test zum manuellen Laden und Testen von Flux-Modellen
Führt das Modell direkt ohne Server, um Fehlerquellen zu minimieren
"""
import sys
import os

# Füge backend zum Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

print("=" * 60)
print("Flux Modell-Lade-Test")
print("=" * 60)

# 1. Test: ImageManager Import
print("\n[1/5] Teste ImageManager Import...")
try:
    from image_manager import ImageManager
    print("[OK] ImageManager erfolgreich importiert")
except Exception as e:
    print(f"[FEHLER] Fehler beim Import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Test: ImageManager Initialisierung
print("\n[2/5] Teste ImageManager Initialisierung...")
try:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    image_manager = ImageManager(config_path=config_path)
    print(f"[OK] ImageManager initialisiert")
    print(f"  - Device: {image_manager.device}")
    print(f"  - DIFFUSERS_AVAILABLE: {image_manager.is_model_loaded() if hasattr(image_manager, 'is_model_loaded') else 'N/A'}")
except Exception as e:
    print(f"[FEHLER] Fehler bei Initialisierung: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test: Verfügbare Modelle prüfen
print("\n[3/5] Prüfe verfügbare Modelle...")
try:
    available_models = image_manager.get_available_models()
    print(f"[OK] Verfügbare Modelle gefunden: {len(available_models)}")
    for model_id, model_info in available_models.items():
        print(f"  - {model_id}: {model_info.get('path', 'N/A')}")
    
    if not available_models:
        print("[FEHLER] Keine Modelle gefunden!")
        sys.exit(1)
    
    # Waehle erstes Modell oder flux-1-schnell falls vorhanden
    test_model_id = "flux-1-schnell" if "flux-1-schnell" in available_models else list(available_models.keys())[0]
    print(f"\n  -> Verwende Modell: {test_model_id}")
except Exception as e:
    print(f"[FEHLER] Fehler beim Prüfen der Modelle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test: Modell laden
print(f"\n[4/5] Lade Modell: {test_model_id}...")
print("  (Dies kann mehrere Minuten dauern...)")
print("  WARNUNG: Wenn ein anderer Prozess (Server/Agent) gleichzeitig ein Modell lädt,")
print("  kann es zu GPU-Speicherkonflikten kommen. Bitte Server/Agenten vorher beenden.")
try:
    success = image_manager.load_model(test_model_id)
    
    if not success:
        print(f"[WARNUNG] load_model() returned False - möglicherweise lädt bereits eine andere Instanz ein Modell")
        print("  Versuche trotzdem fortzufahren...")
        # Prüfe ob Modell trotzdem geladen wurde
        if image_manager.is_model_loaded() and image_manager.get_current_model() == test_model_id:
            print("[OK] Modell wurde trotzdem erfolgreich geladen (andere Instanz hat möglicherweise geladen)")
            success = True
    
    if success:
        print(f"[OK] Modell erfolgreich geladen!")
        
        # Prüfe Status
        is_loaded = image_manager.is_model_loaded()
        current_model = image_manager.get_current_model()
        print(f"  - is_model_loaded(): {is_loaded}")
        print(f"  - get_current_model(): {current_model}")
        print(f"  - pipeline exists: {image_manager.pipeline is not None}")
        
        if image_manager.pipeline:
            print(f"  - pipeline type: {type(image_manager.pipeline).__name__}")
            if hasattr(image_manager.pipeline, 'unet'):
                print(f"  - has unet: {image_manager.pipeline.unet is not None}")
            if hasattr(image_manager.pipeline, 'vae'):
                print(f"  - has vae: {image_manager.pipeline.vae is not None}")
    else:
        print(f"[FEHLER] Modell-Laden fehlgeschlagen (load_model() returned False)")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n[ABGEBROCHEN] Laden wurde unterbrochen (Ctrl+C)")
    sys.exit(1)
except Exception as e:
    print(f"[FEHLER] Fehler beim Laden: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Test: Bild generieren
print(f"\n[5/5] Teste Bildgenerierung...")
try:
    test_prompt = "a beautiful sunset over mountains"
    print(f"  Prompt: '{test_prompt}'")
    print("  (Dies kann 30-60 Sekunden dauern...)")
    
    image = image_manager.generate_image(
        prompt=test_prompt,
        num_inference_steps=10,  # Weniger Steps für schnelleren Test
        width=512,  # Kleinere Auflösung für schnelleren Test
        height=512
    )
    
    if image:
        print(f"[OK] Bild erfolgreich generiert!")
        print(f"  - Groesse: {image.size}")
        print(f"  - Modus: {image.mode}")
        
        # Speichere Test-Bild
        test_output = os.path.join(os.path.dirname(__file__), "test_output.png")
        image.save(test_output)
        print(f"  - Gespeichert als: {test_output}")
    else:
        print(f"[FEHLER] Bildgenerierung fehlgeschlagen (None zurueckgegeben)")
        sys.exit(1)
        
except Exception as e:
    print(f"[FEHLER] Fehler bei Bildgenerierung: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("[OK] ALLE TESTS ERFOLGREICH!")
print("=" * 60)
print(f"\nModell '{test_model_id}' funktioniert korrekt.")
print("Sie koennen es jetzt im Model-Service verwenden.")

