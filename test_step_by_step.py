"""
Schrittweises Testing der Quality Management Features
1. Clean (alle OFF)
2. Dann eine Option nach der anderen
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def set_quality_settings(settings: dict):
    """Setze Quality Settings"""
    try:
        response = requests.post(f"{API_BASE}/quality/settings", json=settings, timeout=5)
        if response.status_code == 200:
            print(f"[OK] Settings gesetzt: {json.dumps(settings, indent=2)}")
            return True
        else:
            print(f"[ERROR] Fehler beim Setzen: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def wait_for_model(max_wait: int = 120):
    """Warte bis Modell geladen ist"""
    print("Warte auf Modell-Laden...")
    waited = 0
    while waited < max_wait:
        try:
            response = requests.get(f"{API_BASE.replace(':8000', ':8001')}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                if status.get('text_model_loaded'):
                    print(f"[OK] Modell ist geladen nach {waited}s")
                    return True
        except:
            pass
        
        time.sleep(3)
        waited += 3
        if waited % 15 == 0:
            print(f"  ... noch warten ({waited}/{max_wait}s)")
    
    print(f"[WARN] Modell lädt zu lange ({max_wait}s) - teste trotzdem...")
    return False

def test_chat(question: str = "Was ist 2+2?", timeout: int = 30):
    """Teste Chat-Endpoint"""
    print(f"\nFrage: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/chat", json={
            "message": question,
            "max_length": 512,
            "temperature": 0.7
        }, timeout=timeout)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            print(f"[OK] Antwort erhalten nach {duration:.1f}s")
            print(f"Antwort-Laenge: {len(response_text)} Zeichen")
            if len(response_text) > 0:
                print(f"\nAntwort (erste 300 Zeichen):")
                print(response_text[:300])
                return True
            else:
                print("[ERROR] Antwort ist leer!")
                return False
        elif response.status_code == 202:
            # Modell wird noch geladen
            result = response.json()
            detail = result.get('detail', {})
            print(f"[INFO] Modell wird noch geladen: {detail.get('message', '')}")
            print("Warte auf Modell-Laden...")
            wait_for_model()
            # Retry nach Warten
            return test_chat(question, timeout)
        else:
            print(f"[ERROR] Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout nach {timeout}s - Server hängt!")
        return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def test_clean():
    """TEST 1: Clean - Alle Quality Options OFF"""
    print("\n" + "=" * 60)
    print("TEST 1: CLEAN - Alle Quality Options OFF")
    print("=" * 60)
    
    settings = {
        "web_validation": False,
        "contradiction_check": False,
        "hallucination_check": False,
        "actuality_check": False,
        "source_quality_check": False,
        "completeness_check": False,
        "auto_web_search": False
    }
    
    if not set_quality_settings(settings):
        return False
    
    time.sleep(1)  # Warte kurz damit Settings gespeichert werden
    
    return test_chat("Was ist 2+2?", timeout=30)

def test_auto_web_search():
    """TEST 2: Nur auto_web_search ON"""
    print("\n" + "=" * 60)
    print("TEST 2: Nur auto_web_search ON")
    print("=" * 60)
    
    settings = {
        "web_validation": False,
        "contradiction_check": False,
        "hallucination_check": False,
        "actuality_check": False,
        "source_quality_check": False,
        "completeness_check": False,
        "auto_web_search": True
    }
    
    if not set_quality_settings(settings):
        return False
    
    time.sleep(1)
    
    return test_chat("Wie lange kann man unter Wasser atmen?", timeout=60)

def test_web_validation():
    """TEST 3: Nur web_validation ON"""
    print("\n" + "=" * 60)
    print("TEST 3: Nur web_validation ON")
    print("=" * 60)
    
    settings = {
        "web_validation": True,
        "contradiction_check": False,
        "hallucination_check": False,
        "actuality_check": False,
        "source_quality_check": False,
        "completeness_check": False,
        "auto_web_search": False
    }
    
    if not set_quality_settings(settings):
        return False
    
    time.sleep(1)
    
    return test_chat("Was ist 2+2?", timeout=60)

def test_hallucination_check():
    """TEST 4: Nur hallucination_check ON"""
    print("\n" + "=" * 60)
    print("TEST 4: Nur hallucination_check ON")
    print("=" * 60)
    
    settings = {
        "web_validation": False,
        "contradiction_check": False,
        "hallucination_check": True,
        "actuality_check": False,
        "source_quality_check": False,
        "completeness_check": False,
        "auto_web_search": False
    }
    
    if not set_quality_settings(settings):
        return False
    
    time.sleep(1)
    
    return test_chat("Erkläre mir was Python ist", timeout=30)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCHRITTWEISES QUALITY MANAGEMENT TESTING")
    print("=" * 60)
    
    # Prüfe ob Server läuft
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Server ist erreichbar")
        else:
            print("[ERROR] Server antwortet nicht korrekt")
            exit(1)
    except Exception as e:
        print(f"[ERROR] Server nicht erreichbar: {e}")
        print("Bitte starten Sie die Server mit start_local_ai.bat")
        exit(1)
    
    # Führe Tests Schritt für Schritt aus
    results = []
    
    print("\n>>> STARTE TEST 1: CLEAN")
    results.append(("1. Clean (alle OFF)", test_clean()))
    
    if results[-1][1]:  # Nur weiter wenn Clean funktioniert
        print("\n>>> STARTE TEST 2: auto_web_search")
        results.append(("2. auto_web_search ON", test_auto_web_search()))
        
        print("\n>>> STARTE TEST 3: web_validation")
        results.append(("3. web_validation ON", test_web_validation()))
        
        print("\n>>> STARTE TEST 4: hallucination_check")
        results.append(("4. hallucination_check ON", test_hallucination_check()))
    else:
        print("\n[WARN] Clean-Test fehlgeschlagen - weitere Tests übersprungen")
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("TEST ZUSAMMENFASSUNG")
    print("=" * 60)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
