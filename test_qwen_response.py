"""
Test-Script um Qwen auf Hängen und korrekte Antworten zu testen
"""
import requests
import time
import json

MODEL_SERVICE = "http://127.0.0.1:8001"
MAIN_SERVICE = "http://127.0.0.1:8000"

def test_qwen_simple():
    """Testet Qwen mit einer einfachen Frage"""
    print("=" * 60)
    print("TEST 1: Einfache Frage")
    print("=" * 60)
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{MAIN_SERVICE}/chat",
            json={
                "message": "Antworte nur mit dem Wort: Test",
                "conversation_id": None,
                "max_length": 50,  # Sehr kurz für schnellen Test
                "temperature": 0.3
            },
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.ok:
            data = response.json()
            answer = data.get("response", "")
            print(f"[OK] Erfolg nach {elapsed:.2f}s")
            print(f"Antwort: {answer}")
            print(f"Antwort-Länge: {len(answer)} Zeichen")
            
            # Prüfe ob Antwort korrekt ist
            if "test" in answer.lower():
                print("[OK] Antwort enthält 'test' - korrekt!")
            else:
                print(f"[WARN] Antwort enthält nicht 'test': '{answer}'")
        else:
            print(f"[FEHLER] Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"[TIMEOUT] Nach {elapsed:.2f}s - Qwen hängt!")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FEHLER] Nach {elapsed:.2f}s: {e}")

def test_qwen_multiple():
    """Testet mehrere Anfragen hintereinander"""
    print("\n" + "=" * 60)
    print("TEST 2: Mehrere Anfragen hintereinander")
    print("=" * 60)
    
    questions = [
        "Was ist 2+2? Antworte nur mit der Zahl.",
        "Wie heißt die Hauptstadt von Deutschland? Antworte nur mit dem Namen.",
        "Antworte nur mit: OK"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Frage {i}/{len(questions)} ---")
        print(f"Frage: {question}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{MAIN_SERVICE}/chat",
                json={
                    "message": question,
                    "conversation_id": None,
                    "max_length": 50,
                    "temperature": 0.3
                },
                timeout=60
            )
            elapsed = time.time() - start_time
            
            if response.ok:
                data = response.json()
                answer = data.get("response", "")
                print(f"[OK] Antwort nach {elapsed:.2f}s: {answer[:100]}")
            else:
                print(f"[FEHLER] Status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"[TIMEOUT] Nach {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[FEHLER] Nach {elapsed:.2f}s: {e}")
        
        time.sleep(2)  # Kurze Pause zwischen Anfragen

def check_status():
    """Prüft den Status der Modelle"""
    print("\n" + "=" * 60)
    print("STATUS CHECK")
    print("=" * 60)
    
    try:
        r = requests.get(f"{MODEL_SERVICE}/status", timeout=5)
        if r.ok:
            status = r.json()
            print(f"Text Model: {'OK - GELADEN' if status.get('text_model', {}).get('loaded') else 'FEHLER - NICHT GELADEN'}")
            print(f"Audio Model: {'OK - GELADEN' if status.get('audio_model', {}).get('loaded') else 'FEHLER - NICHT GELADEN'}")
            
            gpu_alloc = status.get('gpu_allocation', {})
            print(f"\nGPU-Allokation:")
            print(f"  Primary Budget: {gpu_alloc.get('primary_budget_percent')}%")
            print(f"  Secondary Budget: {gpu_alloc.get('secondary_budget_percent')}%")
            print(f"  Allocations: {gpu_alloc.get('allocations', {})}")
    except Exception as e:
        print(f"[FEHLER] Beim Status-Check: {e}")

if __name__ == "__main__":
    check_status()
    test_qwen_simple()
    test_qwen_multiple()
    print("\n" + "=" * 60)
    print("TESTS ABGESCHLOSSEN")
    print("=" * 60)
