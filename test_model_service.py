"""Test Model-Service Status"""
import requests
import json

print("=" * 60)
print("Test Model-Service Status")
print("=" * 60)

# Test 1: Model-Service Health
print("\n[Test 1] Pruefe Model-Service Health...")
try:
    r = requests.get('http://127.0.0.1:8001/health', timeout=2)
    if r.status_code == 200:
        print("[OK] Model-Service ist erreichbar")
        print(f"Status: {json.dumps(r.json(), indent=2)}")
    else:
        print(f"[FEHLER] Status {r.status_code}")
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"[FEHLER] Model-Service nicht erreichbar: {e}")
    print("Model-Service laeuft moeglicherweise nicht")

# Test 2: Text-Modell Status
print("\n[Test 2] Pruefe Text-Modell Status...")
try:
    r = requests.get('http://127.0.0.1:8001/models/text/status', timeout=2)
    if r.status_code == 200:
        status = r.json()
        print(f"[OK] Text-Modell Status:")
        print(f"  Geladen: {status.get('loaded', False)}")
        print(f"  Modell ID: {status.get('model_id', 'N/A')}")
        print(f"  Gesund: {status.get('healthy', False)}")
        
        if not status.get('loaded', False):
            print("\n[WARN] Kein Text-Modell geladen!")
            print("Das ist wahrscheinlich die Ursache des 500-Fehlers.")
    else:
        print(f"[FEHLER] Status {r.status_code}")
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"[FEHLER] {e}")

# Test 3: Versuche Chat direkt am Model-Service
print("\n[Test 3] Teste Chat direkt am Model-Service...")
try:
    r = requests.post(
        'http://127.0.0.1:8001/chat',
        json={
            'message': 'Hallo',
            'max_length': 64,
            'temperature': 0.3
        },
        timeout=5
    )
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        print("[OK] Chat funktioniert direkt am Model-Service")
        print(f"Response: {r.json().get('response', '')[:100]}")
    else:
        print(f"[FEHLER] Status {r.status_code}")
        try:
            error = r.json()
            print(f"Error: {json.dumps(error, indent=2)}")
        except:
            print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"[FEHLER] {e}")

print("\n" + "=" * 60)




