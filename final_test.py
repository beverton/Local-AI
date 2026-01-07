"""
Finaler Test - ChatAgent mit Web-Search und minimaler Bereinigung
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

# Test-Fragen
test_questions = [
    "Wer ist aktuell Bundeskanzler von Deutschland?",
    "Was ist Künstliche Intelligenz?",
    "Erkläre mir Python in einem Satz"
]

print("=" * 70)
print("FINAL TEST - ChatAgent mit Web-Search + Minimaler Bereinigung")
print("=" * 70)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}: {question}")
    print("=" * 70)
    
    try:
        # Erstelle Conversation
        conv_response = requests.post(
            f"{BASE_URL}/conversations",
            json={"title": f"Test {i}", "conversation_type": "chat"},
            timeout=5
        )
        conversation_id = conv_response.json()["conversation_id"]
        print(f"[OK] Conversation: {conversation_id}")
        
        # Sende Frage (Streaming)
        print(f"[...] Warte auf Antwort...\n")
        chat_response = requests.post(
            f"{BASE_URL}/chat/stream",
            json={
                "conversation_id": conversation_id,
                "message": question
            },
            stream=True,
            timeout=120
        )
        
        # Sammle Antwort
        full_response = ""
        for line in chat_response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]' or 'done' in data_str.lower():
                        break
                    try:
                        data = json.loads(data_str)
                        chunk = data.get('chunk') or (data.get('content') if data.get('type') == 'content' else None)
                        if chunk:
                            full_response += chunk
                            print(chunk, end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        
        print("\n")
        print("-" * 70)
        print(f"LÄNGE: {len(full_response)} Zeichen")
        
        # Analyse
        has_url = "http" in full_response or "www" in full_response
        has_google_fallback = "google.com/search" in full_response
        is_empty = len(full_response.strip()) == 0
        
        if is_empty:
            print("[FEHLER] Antwort ist leer!")
        elif has_google_fallback:
            print("[WARNUNG] Nur Google-Fallback (keine echten Quellen)")
        elif has_url:
            print("[OK] Enthält URLs")
        else:
            print("[OK] Antwort ohne URLs (vermutlich kein Web-Search nötig)")
        
        time.sleep(1)  # Kurze Pause zwischen Tests
        
    except Exception as e:
        print(f"\n[FEHLER] {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("TESTS ABGESCHLOSSEN")
print("=" * 70)
