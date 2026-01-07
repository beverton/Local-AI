"""Load Mistral-7B Model"""
import requests
import time

print("Loading Mistral-7B-Instruct...")

try:
    # Load Mistral model
    response = requests.post(
        'http://127.0.0.1:8000/models/load',
        json={'model_id': 'mistral-7b-instruct'},
        timeout=300
    )
    
    if response.status_code == 200:
        print("[SUCCESS] Mistral-7B-Instruct loaded!")
        print(response.json())
    else:
        print(f"[ERROR] Status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"[ERROR] {e}")


