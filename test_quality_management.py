"""
Test-Script für Quality Management Features
Testet RAG, Validation und Hallucination-Check
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_quality_settings():
    """Test 1: Prüfe ob Quality Settings verfügbar sind"""
    print("=" * 60)
    print("TEST 1: Quality Settings prüfen")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/quality/settings", timeout=5)
        if response.status_code == 200:
            settings = response.json()
            print("[OK] Quality Settings verfügbar:")
            print(json.dumps(settings, indent=2))
            return settings
        else:
            print(f"[ERROR] Fehler: {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return None

def test_rag_only(question: str = "Wie lange kann man unter Wasser atmen?"):
    """Test 2: Nur RAG aktiv (auto_web_search ON, web_validation OFF)"""
    print("\n" + "=" * 60)
    print("TEST 2: RAG nur (auto_web_search=ON, web_validation=OFF)")
    print("=" * 60)
    
    # Setze Settings
    try:
        requests.post(f"{API_BASE}/quality/settings", json={
            "auto_web_search": True,
            "web_validation": False,
            "hallucination_check": False
        }, timeout=5)
        print("[OK] Settings gesetzt: RAG ON, Validation OFF")
    except Exception as e:
        print(f"[WARN] Konnte Settings nicht setzen: {e}")
    
    # Stelle Frage
    print(f"\nFrage: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/chat", json={
            "message": question,
            "max_length": 1024,
            "temperature": 0.7
        }, timeout=60)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Antwort erhalten nach {duration:.1f}s")
            print(f"Antwort-Laenge: {len(result.get('response', ''))} Zeichen")
            print(f"\nAntwort (erste 500 Zeichen):")
            print(result.get('response', '')[:500])
            return True
        else:
            print(f"[ERROR] Fehler: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def test_validation_only(question: str = "Was ist 2+2?"):
    """Test 3: Nur Validation aktiv (auto_web_search OFF, web_validation ON)"""
    print("\n" + "=" * 60)
    print("TEST 3: Validation nur (auto_web_search=OFF, web_validation=ON)")
    print("=" * 60)
    
    # Setze Settings
    try:
        requests.post(f"{API_BASE}/quality/settings", json={
            "auto_web_search": False,
            "web_validation": True,
            "hallucination_check": False
        }, timeout=5)
        print("[OK] Settings gesetzt: RAG OFF, Validation ON")
    except Exception as e:
        print(f"[WARN] Konnte Settings nicht setzen: {e}")
    
    # Stelle Frage
    print(f"\nFrage: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/chat", json={
            "message": question,
            "max_length": 1024,
            "temperature": 0.7
        }, timeout=60)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Antwort erhalten nach {duration:.1f}s")
            print(f"Antwort-Laenge: {len(result.get('response', ''))} Zeichen")
            print(f"\nAntwort (erste 500 Zeichen):")
            print(result.get('response', '')[:500])
            return True
        else:
            print(f"[ERROR] Fehler: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def test_hallucination_only(question: str = "Erkläre mir was Python ist"):
    """Test 4: Nur Hallucination-Check aktiv"""
    print("\n" + "=" * 60)
    print("TEST 4: Hallucination-Check nur (hallucination_check=ON)")
    print("=" * 60)
    
    # Setze Settings
    try:
        requests.post(f"{API_BASE}/quality/settings", json={
            "auto_web_search": False,
            "web_validation": False,
            "hallucination_check": True
        }, timeout=5)
        print("[OK] Settings gesetzt: Hallucination-Check ON")
    except Exception as e:
        print(f"[WARN] Konnte Settings nicht setzen: {e}")
    
    # Stelle Frage
    print(f"\nFrage: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/chat", json={
            "message": question,
            "max_length": 1024,
            "temperature": 0.7
        }, timeout=60)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Antwort erhalten nach {duration:.1f}s")
            print(f"Antwort-Laenge: {len(result.get('response', ''))} Zeichen")
            print(f"\nAntwort (erste 500 Zeichen):")
            print(result.get('response', '')[:500])
            return True
        else:
            print(f"[ERROR] Fehler: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def test_all_features(question: str = "Wie lange kann man unter Wasser atmen?"):
    """Test 5: Alle Features aktiv (RAG + Validation + Hallucination)"""
    print("\n" + "=" * 60)
    print("TEST 5: Alle Features (RAG + Validation + Hallucination)")
    print("=" * 60)
    
    # Setze Settings
    try:
        requests.post(f"{API_BASE}/quality/settings", json={
            "auto_web_search": True,
            "web_validation": True,
            "hallucination_check": True
        }, timeout=5)
        print("[OK] Settings gesetzt: ALLE Features ON")
    except Exception as e:
        print(f"[WARN] Konnte Settings nicht setzen: {e}")
    
    # Stelle Frage
    print(f"\nFrage: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/chat", json={
            "message": question,
            "max_length": 1024,
            "temperature": 0.7
        }, timeout=120)  # Längerer Timeout für alle Features
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Antwort erhalten nach {duration:.1f}s")
            print(f"Antwort-Laenge: {len(result.get('response', ''))} Zeichen")
            print(f"\nAntwort (erste 500 Zeichen):")
            print(result.get('response', '')[:500])
            return True
        else:
            print(f"[ERROR] Fehler: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QUALITY MANAGEMENT TEST SUITE")
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
    
    # Führe Tests aus
    results = []
    
    results.append(("Settings Check", test_quality_settings()))
    results.append(("RAG Only", test_rag_only()))
    results.append(("Validation Only", test_validation_only()))
    results.append(("Hallucination Only", test_hallucination_only()))
    results.append(("All Features", test_all_features()))
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("TEST ZUSAMMENFASSUNG")
    print("=" * 60)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
