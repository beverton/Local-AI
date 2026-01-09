#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Einfacher Test ohne viel Output"""
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

payload = {
    "message": "Wer ist Brot und Spüle? Suche nach einer Website.",
    "model": "qwen-2.5-7b-instruct"
}

try:
    r = requests.post("http://localhost:8000/chat", json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()
    response = result.get("response", "")
    
    # Prüfe URL
    if "https://www.brotundspuele.de" in response:
        print("✅ ERFOLG: Korrekte URL gefunden!")
        print(f"Antwort: {response[:200]}...")
    else:
        print("❌ FEHLER: URL nicht korrekt")
        print(f"Antwort: {response}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")




