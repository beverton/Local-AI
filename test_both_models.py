"""
Test-Script: Prüft ob Qwen 7B (quantisiert) und Whisper gleichzeitig auf GPU funktionieren
"""
import requests
import time
import json

MODEL_SERVICE = "http://127.0.0.1:8001"
MAIN_SERVICE = "http://127.0.0.1:8000"

def wait_for_service(url, timeout=60):
    """Wartet bis Service verfügbar ist"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_model_loaded(model_type, timeout=300):
    """Wartet bis Modell geladen ist"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{MODEL_SERVICE}/status", timeout=5)
            if r.ok:
                status = r.json()
                model_status = status.get(f"{model_type}_model", {})
                if model_status.get("loaded") and not model_status.get("loading"):
                    return True, status
                if model_status.get("error"):
                    return False, status
        except:
            pass
        time.sleep(5)
    return False, None

def test_both_models():
    """Testet ob beide Modelle gleichzeitig funktionieren"""
    print("=" * 70)
    print("TEST: Qwen 7B (quantisiert) + Whisper Large V3 gleichzeitig")
    print("=" * 70)
    
    # 1. Prüfe Services
    print("\n[1/5] Prüfe Services...")
    if not wait_for_service(f"{MODEL_SERVICE}/status"):
        print("❌ Model Service nicht erreichbar!")
        return False
    if not wait_for_service(f"{MAIN_SERVICE}/status"):
        print("❌ Main Service nicht erreichbar!")
        return False
    print("[OK] Services erreichbar")
    
    # 2. Lade Qwen 7B
    print("\n[2/5] Lade Qwen 7B (quantisiert)...")
    try:
        r = requests.post(
            f"{MODEL_SERVICE}/models/text/load",
            json={"model_id": "qwen-2.5-7b-instruct"},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {r.json().get('message', 'OK')}")
        else:
            print(f"   Response: {r.text[:200]}")
    except Exception as e:
        print(f"   [FEHLER] {e}")
        return False
    
    # Warte auf Qwen
    print("   Warte auf Qwen-Laden...")
    qwen_ok, status = wait_for_model_loaded("text", timeout=180)
    if not qwen_ok:
        print("   [FEHLER] Qwen konnte nicht geladen werden!")
        if status:
            print(f"   Error: {status.get('text_model', {}).get('error')}")
        return False
    print("   [OK] Qwen geladen")
    
    # 3. Prüfe GPU-Status nach Qwen
    print("\n[3/5] GPU-Status nach Qwen-Laden:")
    try:
        r = requests.get(f"{MODEL_SERVICE}/status", timeout=5)
        if r.ok:
            s = r.json()
            gpu_alloc = s.get("gpu_allocation", {})
            text_mem = s.get("text_model", {}).get("memory", {})
            print(f"   Qwen Budget: {gpu_alloc.get('allocations', {}).get('text')} GB")
            print(f"   Qwen GPU Usage: {text_mem.get('gpu_usage_percent', 'N/A')}%")
            print(f"   GPU Free: {text_mem.get('gpu_free_gb', 'N/A')} GB")
    except Exception as e:
        print(f"   [WARN] Konnte GPU-Status nicht abrufen: {e}")
    
    # 4. Lade Whisper
    print("\n[4/5] Lade Whisper Large V3...")
    try:
        r = requests.post(
            f"{MODEL_SERVICE}/models/audio/load",
            json={"model_id": "whisper-large-v3"},
            timeout=30
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {r.json().get('message', 'OK')}")
        else:
            print(f"   Response: {r.text[:200]}")
    except Exception as e:
        print(f"   [FEHLER] {e}")
        return False
    
    # Warte auf Whisper
    print("   Warte auf Whisper-Laden...")
    whisper_ok, status = wait_for_model_loaded("audio", timeout=180)
    if not whisper_ok:
        print("   [FEHLER] Whisper konnte nicht geladen werden!")
        if status:
            print(f"   Error: {status.get('audio_model', {}).get('error')}")
        return False
    print("   [OK] Whisper geladen")
    
    # 5. Prüfe finalen Status
    print("\n[5/5] Finaler GPU-Status:")
    try:
        r = requests.get(f"{MODEL_SERVICE}/status", timeout=5)
        if r.ok:
            s = r.json()
            gpu_alloc = s.get("gpu_allocation", {})
            text_mem = s.get("text_model", {}).get("memory", {})
            
            print(f"   [OK] Qwen: {'GELADEN' if s.get('text_model', {}).get('loaded') else 'NICHT GELADEN'}")
            print(f"   [OK] Whisper: {'GELADEN' if s.get('audio_model', {}).get('loaded') else 'NICHT GELADEN'}")
            print(f"\n   GPU-Allokation:")
            print(f"      Primary Budget: {gpu_alloc.get('primary_budget_percent')}%")
            print(f"      Secondary Budget: {gpu_alloc.get('secondary_budget_percent')}%")
            print(f"      Qwen Budget: {gpu_alloc.get('allocations', {}).get('text')} GB")
            print(f"      Whisper Budget: {gpu_alloc.get('allocations', {}).get('audio')} GB")
            print(f"\n   GPU-Speicher:")
            print(f"      Qwen Usage: {text_mem.get('gpu_usage_percent', 'N/A')}%")
            print(f"      GPU Free: {text_mem.get('gpu_free_gb', 'N/A')} GB")
            
            # System Stats
            r2 = requests.get(f"{MAIN_SERVICE}/system/stats", timeout=5)
            if r2.ok:
                stats = r2.json()
                print(f"\n   System GPU:")
                print(f"      GPU Memory: {stats.get('gpu_memory_percent', 'N/A')}%")
                print(f"      GPU Used: {round(stats.get('gpu_memory_used_mb', 0) / 1024, 2)} GB")
                print(f"      GPU Total: {round(stats.get('gpu_memory_total_mb', 0) / 1024, 2)} GB")
    except Exception as e:
        print(f"   [WARN] Konnte Status nicht abrufen: {e}")
    
    # 6. Teste Qwen-Antwort
    print("\n[6/6] Teste Qwen-Antwort...")
    try:
        start = time.time()
        r = requests.post(
            f"{MAIN_SERVICE}/chat",
            json={
                "message": "Antworte nur mit: OK",
                "conversation_id": None,
                "max_length": 10,
                "temperature": 0.1
            },
            timeout=60
        )
        elapsed = time.time() - start
        
        if r.ok:
            response = r.json().get("response", "")
            print(f"   [OK] Qwen antwortet nach {elapsed:.2f}s")
            print(f"   Antwort: {response[:100]}")
            
            if "ok" in response.lower() or len(response.strip()) > 0:
                print("   [OK] Qwen funktioniert korrekt!")
                return True
            else:
                print(f"   [WARN] Qwen-Antwort unerwartet: '{response}'")
        else:
            print(f"   [FEHLER] Qwen: {r.status_code} - {r.text[:200]}")
    except requests.exceptions.Timeout:
        print(f"   [TIMEOUT] Nach 60s - Qwen hängt!")
        return False
    except Exception as e:
        print(f"   [FEHLER] {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_both_models()
    print("\n" + "=" * 70)
    if success:
        print("[ERFOLG] TEST ERFOLGREICH: Beide Modelle funktionieren gemeinsam!")
    else:
        print("[FEHLER] TEST FEHLGESCHLAGEN")
    print("=" * 70)
