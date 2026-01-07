"""Test Chat-Endpoint nach Fix"""
import requests
import time
import json

print("=" * 60)
print("Test Chat-Endpoint")
print("=" * 60)

# Test 1: Health-Check
print("\n[Test 1] Pruefe Server-Erreichbarkeit...")
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=2)
    if r.status_code == 200:
        print("[OK] Server ist erreichbar")
        print(f"Status: {r.json()}")
    else:
        print(f"[FEHLER] Status {r.status_code}")
        exit(1)
except Exception as e:
    print(f"[FEHLER] Server nicht erreichbar: {e}")
    exit(1)

# Test 2: Chat-Request
print("\n[Test 2] Sende Chat-Request...")
message = "Hallo, antworte kurz mit 'Hallo, ich funktioniere!'"

print(f"Nachricht: {message}")
print("Sende Request...")

start = time.time()

try:
    r = requests.post(
        'http://127.0.0.1:8000/chat',
        json={
            'message': message,
            'conversation_id': None,
            'max_length': 128,
            'temperature': 0.3
        },
        timeout=60
    )
    
    elapsed = time.time() - start
    
    print(f'\nStatus Code: {r.status_code}')
    print(f'Antwortzeit: {elapsed:.1f} Sekunden')
    
    if r.status_code == 200:
        data = r.json()
        response = data.get('response', '')
        conversation_id = data.get('conversation_id', '')
        
        print(f'\n[OK] Erfolg!')
        print(f'Conversation ID: {conversation_id}')
        print(f'Response-Laenge: {len(response)} Zeichen')
        print(f'\nResponse:')
        print(response)
        print("\n" + "=" * 60)
        print("[ERFOLG] Chat-Endpoint funktioniert korrekt!")
        print("=" * 60)
    else:
        print(f'\n[FEHLER] Status {r.status_code}')
        try:
            error_detail = r.json()
            print(f'Error Detail: {json.dumps(error_detail, indent=2)}')
        except:
            print(f'Response Text: {r.text[:500]}')
        print("\n" + "=" * 60)
        print("[FEHLER] Chat-Endpoint gibt Fehler zurueck!")
        print("=" * 60)
        
except requests.exceptions.Timeout:
    elapsed = time.time() - start
    print(f'\n[TIMEOUT] Nach {elapsed:.1f} Sekunden')
    print("Modell-Generierung dauert zu lange (normal bei grossen Modellen)")
except requests.exceptions.ConnectionError as e:
    print(f'\n[VERBINDUNGSFEHLER] {e}')
    print("Server ist moeglicherweise nicht gestartet")
except Exception as e:
    elapsed = time.time() - start
    print(f'\n[FEHLER] Nach {elapsed:.1f} Sekunden: {e}')
    import traceback
    print(f"Traceback: {traceback.format_exc()}")

print("\n" + "=" * 60)




