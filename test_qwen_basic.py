"""
Test-Script für Qwen Basis-Funktionalität
Testet einfache, komplexe und Coding-Fragen
"""
import requests
import json
import time
import sys
import os

# Füge Backend zum Path hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

BASE_URL = "http://127.0.0.1:8000"

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
        print(f"  Versuch {i+1}/{max_wait}...")
    return False

def wait_for_model_load(max_wait=120):
    """Wartet bis Modell geladen ist"""
    print("Warte auf Modell-Laden...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("text_model_loaded", False):
                    print("Modell ist geladen!")
                    return True
        except Exception as e:
            print(f"  Fehler beim Status-Check: {e}")
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
                    print("✓ Keywords gefunden")
                else:
                    print("⚠ Nicht alle Keywords gefunden")
            
            # Prüfe ob Antwort vollständig ist
            if len(answer) < 10:
                print("❌ Antwort zu kurz!")
                return False
            elif answer.strip() == "":
                print("❌ Antwort ist leer!")
                return False
            else:
                print("✓ Antwort erhalten")
                return True
        else:
            print(f"❌ Fehler: Status {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Timeout - Antwort dauerte zu lange")
        return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("="*60)
    print("Qwen Basis-Tests")
    print("="*60)
    
    # Warte auf Server
    if not wait_for_server():
        print("❌ Server nicht verfügbar!")
        return False
    
    # Warte auf Modell
    if not wait_for_model_load():
        print("❌ Modell konnte nicht geladen werden!")
        return False
    
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
    print(f"Test 1 (Einfach): {'✓' if test1_ok else '❌'}")
    print(f"Test 2 (Komplex): {'✓' if test2_ok else '❌'}")
    print(f"Test 3 (Coding): {'✓' if test3_ok else '❌'}")
    
    all_ok = test1_ok and test2_ok and test3_ok
    print(f"\nGesamt: {'✓ Alle Tests bestanden' if all_ok else '❌ Einige Tests fehlgeschlagen'}")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
