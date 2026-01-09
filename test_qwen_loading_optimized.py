"""
Test: Prüft ob die optimierte ModelManager-Implementierung schneller ist
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import sys

# Füge Backend zum Path hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from model_manager import ModelManager

QWEN_MODEL_ID = "qwen-2.5-7b-instruct"

def test_loading():
    """Testet das Laden mit dem optimierten ModelManager"""
    print("="*70)
    print("TEST: Optimierte ModelManager-Implementierung")
    print("="*70)
    
    # Initialisiere ModelManager
    print("Initialisiere ModelManager...")
    manager = ModelManager(config_path="config.json")
    
    # Prüfe ob Modell in Config existiert
    if QWEN_MODEL_ID not in manager.config.get("models", {}):
        print(f"✗ Modell {QWEN_MODEL_ID} nicht in Config gefunden!")
        return False
    
    # Teste Laden
    print(f"\nLade Modell: {QWEN_MODEL_ID}")
    print("Starte Zeitmessung...")
    
    start_time = time.time()
    success = manager.load_model(QWEN_MODEL_ID)
    load_time = time.time() - start_time
    
    if success:
        print(f"\n✓ Modell erfolgreich geladen!")
        print(f"✓ Ladezeit: {load_time:.2f}s")
        
        # Teste Inferenz
        print("\nTeste Inferenz...")
        messages = [{"role": "user", "content": "Was ist 2+3? Antworte kurz."}]
        
        start_time = time.time()
        response = manager.generate(messages, max_length=50, temperature=0.3)
        inference_time = time.time() - start_time
        
        print(f"✓ Inferenzzeit: {inference_time:.2f}s")
        print(f"✓ Antwort: {response[:100]}...")
        
        # Vergleich mit Test-Ergebnissen
        print("\n" + "="*70)
        print("VERGLEICH:")
        print("="*70)
        print(f"Optimierte Implementierung: {load_time:.2f}s")
        print(f"Test-Script (Method 6):      4.79s")
        print(f"Alte Implementierung:        180.89s")
        print(f"\nVerbesserung: {((180.89 - load_time) / 180.89 * 100):.1f}% schneller")
        
        if load_time < 10:
            print("\n✓ Optimierung erfolgreich! Ladezeit deutlich verbessert.")
        else:
            print(f"\n⚠ Ladezeit noch relativ hoch ({load_time:.2f}s). Möglicherweise weitere Optimierungen nötig.")
        
        return True
    else:
        print(f"\n✗ Fehler beim Laden des Modells!")
        return False

if __name__ == "__main__":
    try:
        success = test_loading()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest abgebrochen durch Benutzer")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
