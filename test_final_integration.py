"""
Finale Integration-Tests für Qwen mit Quality Management
Smoke-Tests, Integration-Tests, Performance-Tests
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time
import os

API_BASE = "http://127.0.0.1:8000"

def wait_for_server(max_wait=30):
    """Wartet bis Server verfügbar ist"""
    print("Warte auf Server...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/status", timeout=2)
            if response.status_code == 200:
                print("Server ist verfügbar!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_model_load(max_wait=120):
    """Wartet bis Modell geladen ist"""
    print("Warte auf Modell-Laden...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("text_model_loaded", False):
                    print("Modell ist geladen!")
                    return True
        except:
            pass
        time.sleep(2)
        if i % 10 == 0:
            print(f"  Warte... ({i*2}s)")
    return False

def test_chat(question, timeout=120):
    """Testet eine Chat-Anfrage"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": question,
                "conversation_id": None,
                "max_length": 2048,
                "temperature": 0.3
            },
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            return {
                "success": True,
                "answer": answer,
                "elapsed": elapsed,
                "length": len(answer)
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}",
                "elapsed": elapsed
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout",
            "elapsed": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": 0
        }

def main():
    """Hauptfunktion"""
    print("="*60)
    print("Finale Integration-Tests")
    print("="*60)
    
    # Warte auf Server
    if not wait_for_server():
        print("[ERROR] Server nicht verfügbar!")
        return False
    
    # Warte auf Modell
    if not wait_for_model_load():
        print("[WARN] Modell konnte nicht geladen werden - teste trotzdem...")
    
    # Smoke-Tests
    print("\n" + "="*60)
    print("SMOKE-TESTS")
    print("="*60)
    
    smoke_tests = [
        ("Einfache Frage", "Was ist 2+3?"),
        ("Coding-Frage", "Schreibe eine Python-Funktion die Fibonacci berechnet"),
        ("Komplexe Frage", "Erkläre Machine Learning")
    ]
    
    smoke_results = []
    for name, question in smoke_tests:
        print(f"\n--- {name}: {question[:50]}...")
        result = test_chat(question)
        smoke_results.append((name, result))
        if result["success"]:
            print(f"[OK] Antwort erhalten ({result['elapsed']:.2f}s, {result['length']} Zeichen)")
        else:
            print(f"[FAIL] {result.get('error', 'Unbekannter Fehler')}")
    
    # Integration-Tests
    print("\n" + "="*60)
    print("INTEGRATION-TESTS")
    print("="*60)
    
    # Test mit Quality Management
    print("\n--- Test mit Quality Management (alle Optionen deaktiviert)")
    result = test_chat("Was ist Python?")
    if result["success"]:
        print(f"[OK] Antwort erhalten ({result['elapsed']:.2f}s)")
    else:
        print(f"[FAIL] {result.get('error', 'Unbekannter Fehler')}")
    
    # Performance-Tests
    print("\n" + "="*60)
    print("PERFORMANCE-TESTS")
    print("="*60)
    
    performance_tests = [
        ("Kurze Frage", "Was ist 2+2?", 5.0),
        ("Mittlere Frage", "Erkläre Python Decorators", 30.0),
        ("Lange Frage", "Beschreibe vollständig wie Neural Networks funktionieren", 60.0)
    ]
    
    performance_results = []
    for name, question, max_time in performance_tests:
        print(f"\n--- {name}: {question[:50]}...")
        result = test_chat(question, timeout=int(max_time))
        performance_results.append((name, result, max_time))
        if result["success"]:
            print(f"[OK] Antwort erhalten ({result['elapsed']:.2f}s, max: {max_time}s)")
            if result['elapsed'] > max_time:
                print(f"[WARN] Antwort dauerte länger als erwartet")
        else:
            print(f"[FAIL] {result.get('error', 'Unbekannter Fehler')}")
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    smoke_ok = sum(1 for _, r in smoke_results if r["success"])
    print(f"Smoke-Tests: {smoke_ok}/{len(smoke_results)} erfolgreich")
    
    perf_ok = sum(1 for _, r, _ in performance_results if r["success"])
    print(f"Performance-Tests: {perf_ok}/{len(performance_results)} erfolgreich")
    
    all_ok = smoke_ok == len(smoke_results) and perf_ok == len(performance_results)
    print(f"\nGesamt: {'✓ Alle Tests bestanden' if all_ok else '❌ Einige Tests fehlgeschlagen'}")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
