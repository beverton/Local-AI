"""Load Qwen2.5-7B-Instruct Model"""
import requests
import time

print("Loading Qwen2.5-7B-Instruct...")

try:
    response = requests.post(
        'http://127.0.0.1:8000/models/load',
        json={'model_id': 'qwen-2.5-7b-instruct'},
        timeout=300
    )
    
    if response.status_code == 200:
        print("[SUCCESS] Qwen2.5-7B-Instruct wird geladen!")
        print(response.json())
        print("\nWarte 60 Sekunden auf vollständiges Laden...")
        time.sleep(60)
        print("[READY] Modell sollte jetzt bereit sein!")
    else:
        print(f"[ERROR] Status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"[ERROR] {e}")

