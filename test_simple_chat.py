"""Einfacher Chat-Test"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

# Test 1: Model-Service direkt
print("=" * 60)
print("TEST: Model-Service direkt (Port 8001)")
print("=" * 60)

try:
    r = requests.post('http://127.0.0.1:8001/chat', json={
        'message': 'Was ist 2+2?',
        'messages': [{'role': 'user', 'content': 'Was ist 2+2?'}]
    }, timeout=60)
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        response_text = result.get('response', '')
        print(f"Response length: {len(response_text)}")
        print(f"Response: {response_text[:500]}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Main Server (Port 8000)
print("\n" + "=" * 60)
print("TEST: Main Server (Port 8000)")
print("=" * 60)

try:
    r = requests.post('http://127.0.0.1:8000/chat', json={
        'message': 'Was ist 2+2?'
    }, timeout=60)
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        response_text = result.get('response', '')
        print(f"Response length: {len(response_text)}")
        print(f"Response: {response_text[:500]}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Error: {e}")
