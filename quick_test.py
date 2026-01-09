#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Quick Test...")
r = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hallo, kannst du mich hören?"},
    timeout=30
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Response: {r.json().get('response', '')}")
else:
    print(f"Error: {r.text}")




