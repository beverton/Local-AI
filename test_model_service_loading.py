"""
Test-Script für Model Service Loading-Optimierungen
- Prüft ob nur Qwen beim Start geladen wird
- Prüft Memory-Informationen im Status
- Prüft ob keine gleichzeitigen Ladevorgänge stattfinden
"""
import requests
import time
import json

MODEL_SERVICE_URL = "http://127.0.0.1:8001"

def test_status_endpoint():
    """Testet den Status-Endpoint mit Memory-Informationen"""
    print("\n" + "="*60)
    print("TEST 1: Status-Endpoint mit Memory-Informationen")
    print("="*60)
    
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"[OK] Status-Endpoint erfolgreich")
            print(f"\nText-Modell Status:")
            text_model = status.get("text_model", {})
            print(f"  - Geladen: {text_model.get('loaded', False)}")
            print(f"  - Model ID: {text_model.get('model_id', 'None')}")
            print(f"  - Loading: {text_model.get('loading', False)}")
            
            # Memory-Informationen
            memory = text_model.get("memory")
            if memory:
                print(f"\n  Memory-Informationen:")
                print(f"  - GPU Allocated: {memory.get('gpu_allocated_gb', 0)} GB")
                print(f"  - GPU Reserved: {memory.get('gpu_reserved_gb', 0)} GB")
                print(f"  - GPU Total: {memory.get('gpu_total_gb', 0)} GB")
                print(f"  - GPU Free: {memory.get('gpu_free_gb', 0)} GB")
                print(f"  - GPU Usage: {memory.get('gpu_usage_percent', 0)}%")
            else:
                print(f"  [WARN] Keine Memory-Informationen verfügbar")
            
            print(f"\nAudio-Modell Status:")
            audio_model = status.get("audio_model", {})
            print(f"  - Geladen: {audio_model.get('loaded', False)}")
            print(f"  - Model ID: {audio_model.get('model_id', 'None')}")
            
            print(f"\nImage-Modell Status:")
            image_model = status.get("image_model", {})
            print(f"  - Geladen: {image_model.get('loaded', False)}")
            print(f"  - Model ID: {image_model.get('model_id', 'None')}")
            
            return True
        else:
            print(f"[ERROR] Status-Endpoint Fehler: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Model Service nicht erreichbar auf {MODEL_SERVICE_URL}")
        print(f"  Bitte starten Sie den Model Service zuerst!")
        return False
    except Exception as e:
        print(f"✗ Fehler: {e}")
        return False

def test_model_loading():
    """Testet Modell-Laden (nur ein Modell gleichzeitig)"""
    print("\n" + "="*60)
    print("TEST 2: Modell-Laden (nur ein Modell gleichzeitig)")
    print("="*60)
    
    try:
        # Prüfe aktuellen Status
        response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
        if response.status_code != 200:
            print(f"[ERROR] Konnte Status nicht abrufen: {response.status_code}")
            return False
        
        status = response.json()
        current_model = status.get("text_model", {}).get("model_id")
        is_loading = status.get("text_model", {}).get("loading", False)
        
        print(f"Aktuelles Modell: {current_model}")
        print(f"Wird geladen: {is_loading}")
        
        if is_loading:
            print(f"[INFO] Modell wird bereits geladen, warte auf Abschluss...")
            # Warte bis Laden abgeschlossen ist
            max_wait = 300  # 5 Minuten
            wait_time = 0
            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    if not status.get("text_model", {}).get("loading", False):
                        print(f"[OK] Modell-Laden abgeschlossen")
                        break
                print(f"  Warte... ({wait_time}s)")
        
        # Test: Versuche Modell zu laden (sollte bereits geladen sein oder laden)
        if current_model:
            print(f"\n[OK] Modell bereits geladen: {current_model}")
            print(f"  Test: Versuche dasselbe Modell nochmal zu laden...")
            response = requests.post(
                f"{MODEL_SERVICE_URL}/models/text/load",
                json={"model_id": current_model},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                print(f"  Status: {result.get('status')}")
                print(f"  Message: {result.get('message')}")
                if result.get('status') == 'success':
                    print(f"  [OK] Modell bereits geladen (korrekt erkannt)")
                else:
                    print(f"  [WARN] Unerwarteter Status: {result.get('status')}")
            else:
                print(f"  [ERROR] Fehler beim Laden: {response.status_code}")
        else:
            print(f"[WARN] Kein Modell geladen")
        
        return True
    except Exception as e:
        print(f"✗ Fehler: {e}")
        return False

def test_concurrent_loading():
    """Testet ob gleichzeitige Ladevorgänge verhindert werden"""
    print("\n" + "="*60)
    print("TEST 3: Verhindere gleichzeitige Ladevorgänge")
    print("="*60)
    
    try:
        # Versuche zwei Modell-Ladevorgänge gleichzeitig zu starten
        print("Starte ersten Ladevorgang...")
        response1 = requests.post(
            f"{MODEL_SERVICE_URL}/models/text/load",
            json={"model_id": "qwen-2.5-7b-instruct"},
            timeout=5
        )
        
        if response1.status_code == 200:
            result1 = response1.json()
            print(f"  Erster Request: {result1.get('status')} - {result1.get('message')}")
            
            # Warte kurz
            time.sleep(1)
            
            # Versuche zweiten Ladevorgang zu starten
            print("Starte zweiten Ladevorgang (sollte blockiert werden)...")
            response2 = requests.post(
                f"{MODEL_SERVICE_URL}/models/text/load",
                json={"model_id": "qwen-2.5-7b-instruct"},
                timeout=5
            )
            
            if response2.status_code == 200:
                result2 = response2.json()
                print(f"  Zweiter Request: {result2.get('status')} - {result2.get('message')}")
                
                if result2.get('status') == 'loading' and 'bereits geladen' in result2.get('message', '').lower():
                    print(f"  [OK] Gleichzeitige Ladevorgänge werden korrekt verhindert")
                    return True
                else:
                    print(f"  [WARN] Zweiter Request wurde nicht blockiert")
                    return False
            else:
                print(f"  [ERROR] Fehler beim zweiten Request: {response2.status_code}")
                return False
        else:
            print(f"  [ERROR] Fehler beim ersten Request: {response1.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def wait_for_model_load():
    """Wartet bis Modell geladen ist"""
    print("\nWarte auf Modell-Laden...")
    max_wait = 300  # 5 Minuten
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                text_model = status.get("text_model", {})
                is_loaded = text_model.get("loaded", False)
                is_loading = text_model.get("loading", False)
                
                if is_loaded and not is_loading:
                    print(f"[OK] Modell geladen: {text_model.get('model_id')}")
                    return True
                elif is_loading:
                    print(f"  Lade... ({wait_time}s)")
                else:
                    print(f"  Warte... ({wait_time}s)")
            
            time.sleep(5)
            wait_time += 5
        except Exception as e:
            print(f"  Fehler beim Status-Check: {e}")
            time.sleep(5)
            wait_time += 5
    
    print(f"[ERROR] Timeout beim Warten auf Modell-Laden")
    return False

def main():
    """Hauptfunktion"""
    print("\n" + "="*60)
    print("MODEL SERVICE LOADING TESTS")
    print("="*60)
    print(f"Model Service URL: {MODEL_SERVICE_URL}")
    
    # Prüfe ob Service erreichbar ist
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"\n[ERROR] Model Service nicht erreichbar oder nicht gesund")
            print(f"  Bitte starten Sie den Model Service zuerst!")
            return
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Model Service nicht erreichbar auf {MODEL_SERVICE_URL}")
        print(f"  Bitte starten Sie den Model Service zuerst!")
        return
    except Exception as e:
        print(f"\n[ERROR] Fehler beim Verbinden: {e}")
        return
    
    print(f"[OK] Model Service erreichbar")
    
    # Warte auf Modell-Laden (falls beim Start geladen wird)
    wait_for_model_load()
    
    # Führe Tests aus
    results = []
    
    results.append(("Status-Endpoint", test_status_endpoint()))
    results.append(("Modell-Laden", test_model_loading()))
    results.append(("Gleichzeitige Ladevorgänge", test_concurrent_loading()))
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("TEST-ZUSAMMENFASSUNG")
    print("="*60)
    
    for test_name, result in results:
        status = "[OK] BESTANDEN" if result else "[ERROR] FEHLGESCHLAGEN"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALLE TESTS BESTANDEN")
    else:
        print("[ERROR] EINIGE TESTS FEHLGESCHLAGEN")
    print("="*60)

if __name__ == "__main__":
    main()
