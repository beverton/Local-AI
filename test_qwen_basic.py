"""
Test-Script für Qwen Basis-Funktionalität
Testet einfache, komplexe und Coding-Fragen
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time
import os

# Füge Backend zum Path hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

BASE_URL = "http://127.0.0.1:8000"
MODEL_SERVICE_URL = "http://127.0.0.1:8001"

def wait_for_server(max_wait=30):
    """Wartet bis Server verfügbar ist"""
    print("Warte auf Server...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/status", timeout=2)
            if response.status_code == 200:
                print("Server ist verfügbar!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_model_load(max_wait=300):
    """Wartet bis Modell geladen ist"""
    print("Warte auf Modell-Laden...")
    
    # Prüfe zuerst ob Model Service verfügbar ist
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
        if response.status_code == 200:
            use_model_service = True
        else:
            use_model_service = False
    except:
        use_model_service = False
    
    # Wenn Model Service nicht verfügbar, prüfe Main Server
    if not use_model_service:
        for i in range(max_wait):
            try:
                response = requests.get(f"{BASE_URL}/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded", False):
                        print("Modell ist geladen!")
                        return True
            except:
                pass
            time.sleep(2)
            if i % 10 == 0:
                print(f"  Warte... ({i*2}s)")
    else:
        # Prüfe Model Service
        for i in range(max_wait):
            try:
                response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    text_model = data.get("text_model", {})
                    if text_model.get("loaded", False):
                        print("Modell ist geladen!")
                        return True
                    elif text_model.get("loading", False):
                        print(f"  Modell lädt noch... ({i*2}s)")
            except:
                pass
            time.sleep(2)
            if i % 10 == 0:
                print(f"  Warte... ({i*2}s)")
    
    return False

def test_chat(question, expected_keywords=None):
    """Testet eine Chat-Anfrage"""
    print(f"\n{'='*60}")
    print(f"Frage: {question}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": question,
                "conversation_id": None,
                "max_length": 2048,
                "temperature": 0.3
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            
            print(f"\nAntwort ({elapsed:.2f}s):")
            print(f"{'-'*60}")
            print(answer[:500] + ("..." if len(answer) > 500 else ""))
            print(f"{'-'*60}")
            print(f"Länge: {len(answer)} Zeichen")
            
            # Prüfe erwartete Keywords
            if expected_keywords:
                found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
                print(f"Erwartete Keywords: {expected_keywords}")
                print(f"Gefundene Keywords: {found}")
                if len(found) >= len(expected_keywords) * 0.5:  # Mindestens 50%
                    print("[OK] Keywords gefunden")
                else:
                    print("[WARN] Nicht alle Keywords gefunden")
            
            # Prüfe ob Antwort vollständig ist
            if len(answer) < 10:
                print("[FAIL] Antwort zu kurz!")
                return False
            elif answer.strip() == "":
                print("[FAIL] Antwort ist leer!")
                return False
            else:
                print("[OK] Antwort erhalten")
                return True
        else:
            print(f"[FAIL] Fehler: Status {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("[FAIL] Timeout - Antwort dauerte zu lange")
        return False
    except Exception as e:
        print(f"[FAIL] Fehler: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("="*60)
    print("Qwen Basis-Tests")
    print("="*60)
    
    # Warte auf Server
    if not wait_for_server():
        print("[FAIL] Server nicht verfügbar!")
        return False
    
    # Warte auf Modell
    if not wait_for_model_load():
        print("[WARN] Modell konnte nicht geladen werden - teste trotzdem...")
    
    # Test 1: Einfache Frage
    print("\n" + "="*60)
    print("TEST 1: Einfache Frage")
    print("="*60)
    test1_ok = test_chat(
        "Was ist 2+3?",
        expected_keywords=["5", "fünf"]
    )
    
    # Test 2: Komplexe Frage
    print("\n" + "="*60)
    print("TEST 2: Komplexe Frage")
    print("="*60)
    test2_ok = test_chat(
        "Erkläre Python Decorators",
        expected_keywords=["decorator", "funktion", "python"]
    )
    
    # Test 3: Coding-Frage
    print("\n" + "="*60)
    print("TEST 3: Coding-Frage")
    print("="*60)
    test3_ok = test_chat(
        "Schreibe eine Fibonacci-Funktion in Python",
        expected_keywords=["def", "fibonacci", "python"]
    )
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    print(f"Test 1 (Einfach): {'[OK]' if test1_ok else '[FAIL]'}")
    print(f"Test 2 (Komplex): {'[OK]' if test2_ok else '[FAIL]'}")
    print(f"Test 3 (Coding): {'[OK]' if test3_ok else '[FAIL]'}")
    
    all_ok = test1_ok and test2_ok and test3_ok
    print(f"\nGesamt: {'[OK] Alle Tests bestanden' if all_ok else '[FAIL] Einige Tests fehlgeschlagen'}")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
