"""Test speziell für Brot und Spüle"""
import requests
import json

print("Testing: Wer ist Brot und Spüle?")
print("=" * 60)

response = requests.post(
    'http://127.0.0.1:8000/chat',
    json={
        'message': 'Wer ist Brot und Spüle? Suche nach der Website. Gib mir den Link.',
        'conversation_id': None,
        'max_length': 512,
        'temperature': 0.1
    },
    timeout=180
)

if response.status_code == 200:
    data = response.json()
    answer = data.get('response', '')
    
    print("\nAntwort:")
    print(answer)
    print("\n" + "=" * 60)
    
    # Check
    if 'brotundspuele.de' in answer.lower():
        print("[SUCCESS] URL korrekt gefunden!")
    else:
        print("[PROBLEM] URL nicht korrekt")
        print("Erwartet: brotundspuele.de")
        print(f"Gefunden: {answer.lower()}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

