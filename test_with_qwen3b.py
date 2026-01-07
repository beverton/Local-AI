#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test mit Qwen 2.5 3B (kleineres Modell)"""
import requests
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Zuerst Model laden
print("Lade Qwen 2.5 3B...")
load_response = requests.post(
    "http://localhost:8001/models/text/load",
    json={"model_id": "qwen-2.5-3b"},
    timeout=120
)
print(f"Load Status: {load_response.status_code}")
print(f"Load Response: {load_response.json()}")

print("\nWarte 30 Sekunden...")
import time
time.sleep(30)

# Test Chat
print("\nSende Chat-Request...")
payload = {
    "message": "Wer ist Brot und Spüle? Suche nach einer Website.",
    "model": "qwen-2.5-3b"
}

try:
    r = requests.post("http://localhost:8000/chat", json=payload, timeout=60)
    print(f"Status Code: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        response = result.get("response", "")
        print(f"\n✅ Response ({len(response)} chars):")
        print(response)
        
        if "https://www.brotundspuele.de" in response:
            print("\n✅ URL gefunden!")
        else:
            print("\n⚠️ URL nicht gefunden (aber Response erhalten)")
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

