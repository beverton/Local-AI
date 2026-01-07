#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detaillierter Test"""
import requests
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

payload = {
    "message": "Wer ist Brot und Spüle? Suche nach einer Website.",
    "model": "qwen-2.5-7b-instruct"
}

try:
    print("Sende Request...")
    r = requests.post("http://localhost:8000/chat", json=payload, timeout=60)
    print(f"Status Code: {r.status_code}")
    print(f"Response Headers: {dict(r.headers)}")
    
    if r.status_code == 200:
        result = r.json()
        print(f"\nJSON Keys: {list(result.keys())}")
        response = result.get("response", "")
        print(f"\nResponse Length: {len(response)}")
        print(f"\nResponse: '{response}'")
        
        if "https://www.brotundspuele.de" in response:
            print("\n✅ URL gefunden!")
        else:
            print("\n❌ URL nicht gefunden")
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

