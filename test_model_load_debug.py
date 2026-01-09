"""
Debug-Script für Modell-Laden
Zeigt detaillierte Logs während des Ladens
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
import os

# Konfiguriere Logging für detaillierte Ausgabe
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Füge Backend zum Path hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from model_manager import ModelManager
import json

def main():
    """Testet Modell-Laden mit detailliertem Logging"""
    print("="*60)
    print("MODEL LOAD DEBUG TEST")
    print("="*60)
    
    config_path = "config.json"
    print(f"\n[TEST] Initialisiere ModelManager mit config_path={config_path}")
    mm = ModelManager(config_path=config_path)
    
    model_id = "qwen-2.5-7b-instruct"
    print(f"\n[TEST] Versuche Modell zu laden: {model_id}")
    print(f"[TEST] Modell in Config: {model_id in mm.config.get('models', {})}")
    
    if model_id in mm.config.get("models", {}):
        model_info = mm.config["models"][model_id]
        model_path = model_info.get("path", "")
        print(f"[TEST] Modell-Pfad: {model_path}")
        print(f"[TEST] Pfad existiert: {os.path.exists(model_path)}")
    
    print(f"\n[TEST] Rufe mm.load_model('{model_id}') auf...")
    try:
        result = mm.load_model(model_id)
        print(f"\n[TEST] load_model() zurückgegeben: {result}")
        print(f"[TEST] is_model_loaded(): {mm.is_model_loaded()}")
        print(f"[TEST] get_current_model(): {mm.get_current_model()}")
        
        if result and mm.is_model_loaded():
            print("\n[TEST] ✓ Modell erfolgreich geladen!")
            
            # Teste einfache Generierung
            print("\n[TEST] Teste einfache Generierung...")
            try:
                response = mm.generate(
                    [{"role": "user", "content": "Was ist 2+3?"}],
                    max_length=512,
                    temperature=0.3
                )
                print(f"[TEST] Generierung erfolgreich: {len(response)} Zeichen")
                print(f"[TEST] Response (erste 200 Zeichen): {response[:200]}")
                return True
            except Exception as e:
                print(f"[TEST] ✗ Generierung fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("\n[TEST] ✗ Modell konnte nicht geladen werden")
            return False
    except Exception as e:
        print(f"\n[TEST] ✗ Exception beim Laden: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
