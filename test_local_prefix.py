"""Test für "local:" Prefix Erkennung"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import requests
import json

def test_model_service_status():
    """Prüft Model Service Status"""
    print("=" * 60)
    print("Test: Model Service Status")
    print("=" * 60)
    
    try:
        r = requests.get('http://127.0.0.1:8001/models/text/status', timeout=5)
        if r.status_code == 200:
            status = r.json()
            print("\n[OK] Model Service ist erreichbar")
            print(f"\nText-Modell Status:")
            print(f"  Geladen: {status.get('loaded', False)}")
            print(f"  Modell ID: {status.get('model_id', 'N/A')}")
            return status.get('loaded', False)
        else:
            print(f"\n[FEHLER] Status {r.status_code}")
            return False
    except Exception as e:
        print(f"\n[FEHLER] Model Service nicht erreichbar: {e}")
        print("\nHinweis: Model Service muss auf Port 8001 laufen")
        return False

def test_local_prefix_detection():
    """Testet die "local:" Prefix Erkennung"""
    print("\n" + "=" * 60)
    print("Test: 'local:' Prefix Erkennung")
    print("=" * 60)
    
    from mcp_server import MCPServer
    
    server = MCPServer()
    
    # Test 1: Mit "local:" Prefix
    print("\n[Test 1] Nachricht MIT 'local:' Prefix:")
    test_message_1 = "local: Was ist 2+5?"
    cleaned_1, was_local_1 = server._strip_local_prefix(test_message_1)
    print(f"  Original: '{test_message_1}'")
    print(f"  Bereinigt: '{cleaned_1}'")
    print(f"  War local: {was_local_1}")
    
    if was_local_1 and cleaned_1 == "Was ist 2+5?":
        print("  [OK] Prefix wurde korrekt erkannt und entfernt")
    else:
        print("  [FEHLER] Prefix wurde nicht korrekt erkannt")
    
    # Test 2: Ohne "local:" Prefix
    print("\n[Test 2] Nachricht OHNE 'local:' Prefix:")
    test_message_2 = "Was ist 2+5?"
    cleaned_2, was_local_2 = server._strip_local_prefix(test_message_2)
    print(f"  Original: '{test_message_2}'")
    print(f"  Bereinigt: '{cleaned_2}'")
    print(f"  War local: {was_local_2}")
    
    if not was_local_2 and cleaned_2 == "Was ist 2+5?":
        print("  [OK] Kein Prefix erkannt (korrekt)")
    else:
        print("  [FEHLER] Falsche Erkennung")
    
    # Test 3: Case-insensitive
    print("\n[Test 3] Case-insensitive Test:")
    test_message_3 = "LOCAL: Test"
    cleaned_3, was_local_3 = server._strip_local_prefix(test_message_3)
    print(f"  Original: '{test_message_3}'")
    print(f"  Bereinigt: '{cleaned_3}'")
    print(f"  War local: {was_local_3}")
    
    if was_local_3:
        print("  [OK] Case-insensitive funktioniert")
    else:
        print("  [FEHLER] Case-insensitive funktioniert nicht")

def test_chat_with_local_prefix():
    """Testet Chat mit "local:" Prefix über Model Service"""
    print("\n" + "=" * 60)
    print("Test: Chat mit 'local:' Prefix")
    print("=" * 60)
    
    # Prüfe erst ob Modell geladen ist
    try:
        r = requests.get('http://127.0.0.1:8001/models/text/status', timeout=5)
        if r.status_code == 200:
            status = r.json()
            if not status.get('loaded', False):
                print("\n[WARN] Kein Text-Modell geladen!")
                print("Bitte laden Sie zuerst ein Modell:")
                print("  POST http://127.0.0.1:8001/models/text/load")
                print('  Body: {"model_id": "qwen-2.5-7b-instruct"}')
                return False
        else:
            print(f"\n[FEHLER] Status {r.status_code}")
            return False
    except Exception as e:
        print(f"\n[FEHLER] Model Service nicht erreichbar: {e}")
        return False
    
    # Test Chat mit "local:" Prefix
    print("\n[Test] Sende Chat-Request mit 'local:' Prefix...")
    try:
        # Simuliere MCP Chat Request
        test_message = "local: Was ist 2+5?"
        
        # Entferne Prefix (wie im MCP Server)
        if test_message.strip().lower().startswith("local:"):
            cleaned_message = test_message[6:].strip()
            print(f"  Original: '{test_message}'")
            print(f"  Bereinigt: '{cleaned_message}'")
        else:
            cleaned_message = test_message
        
        # Sende an Model Service
        response = requests.post(
            'http://127.0.0.1:8001/chat',
            json={
                "message": cleaned_message,
                "messages": [{"role": "user", "content": cleaned_message}],
                "max_length": 512,
                "temperature": 0.3
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            print(f"\n[OK] Antwort erhalten:")
            print(f"  {answer[:200]}...")
            return True
        else:
            print(f"\n[FEHLER] Status {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"\n[FEHLER] Chat-Request fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Test-Suite: 'local:' Prefix Erkennung")
    print("=" * 60)
    
    # Test 1: Model Service Status
    model_loaded = test_model_service_status()
    
    # Test 2: Prefix Erkennung
    test_local_prefix_detection()
    
    # Test 3: Chat mit Prefix (nur wenn Modell geladen)
    if model_loaded:
        test_chat_with_local_prefix()
    else:
        print("\n" + "=" * 60)
        print("Hinweis: Chat-Test übersprungen (kein Modell geladen)")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Tests abgeschlossen")
    print("=" * 60)
