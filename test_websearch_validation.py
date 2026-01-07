"""Test Web Search mit validierbaren Fragen - Anti-Halluzination Test"""
import requests
import time
import json
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("Web Search Anti-Halluzination Test")
print("=" * 60)

# Health-Check
print("\n[Check] Server-Erreichbarkeit...")
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=10)  # Längerer Timeout während Modell lädt
    if r.status_code == 200:
        print("[OK] Server ist erreichbar")
    else:
        print(f"[FEHLER] Status {r.status_code}")
        exit(1)
except Exception as e:
    print(f"[FEHLER] Server nicht erreichbar: {e}")
    print("Versuche trotzdem fortzufahren...")

# Validierbare Test-Fragen mit erwarteten Antworten
test_cases = [
    {
        "question": "Was ist die Hauptstadt von Deutschland? Suche nach der Antwort.",
        "expected_keywords": ["berlin"],
        "expected_url_patterns": ["berlin", "germany", "deutschland", "wikipedia"],
        "should_not_contain": ["münchen", "hamburg", "köln", "paris", "london"]
    },
    {
        "question": "Wer ist Brot und Spüle? Suche nach der Website.",
        "expected_keywords": ["brotundspuele.de", "brot und spüle"],
        "expected_url_patterns": ["brotundspuele.de"],
        "should_not_contain": ["kartoffel", "waschbecken", "küche"]  # Halluzination-Keywords
    },
    {
        "question": "Was ist Python? Suche nach Informationen.",
        "expected_keywords": ["programmiersprache", "programming", "python"],
        "expected_url_patterns": ["python.org", "wikipedia"],
        "should_not_contain": ["schlange essen", "zoo", " tier "]  # Sollte nicht über die Schlange reden (mit Leerzeichen um "interpretierte" zu vermeiden)
    }
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'=' * 60}")
    print(f"[Test {i}/{len(test_cases)}] {test['question']}")
    print("=" * 60)
    
    start = time.time()
    
    try:
        r = requests.post(
            'http://127.0.0.1:8000/chat',
            json={
                'message': test['question'],
                'conversation_id': None,
                'max_length': 512,
                'temperature': 0.1  # Niedrige Temperature für präzisere Antworten
            },
            timeout=180  # Längerer Timeout für Mistral-7B
        )
        
        elapsed = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            response = data.get('response', '').lower()
            
            print(f"Antwortzeit: {elapsed:.1f} Sekunden")
            print(f"\nAntwort:\n{data.get('response', '')}\n")
            
            # Validierung
            passed = True
            issues = []
            
            # Check: Erwartete Keywords vorhanden?
            found_keywords = []
            for keyword in test['expected_keywords']:
                if keyword.lower() in response:
                    found_keywords.append(keyword)
            
            if not found_keywords:
                passed = False
                issues.append(f"Keine erwarteten Keywords gefunden: {test['expected_keywords']}")
            else:
                print(f"[OK] Gefundene Keywords: {found_keywords}")
            
            # Check: Erwartete URL-Muster?
            found_urls = []
            for pattern in test['expected_url_patterns']:
                if pattern.lower() in response:
                    found_urls.append(pattern)
            
            if not found_urls:
                issues.append(f"Keine erwarteten URLs gefunden: {test['expected_url_patterns']}")
                print(f"[WARN] Keine URLs gefunden")
            else:
                print(f"[OK] Gefundene URL-Muster: {found_urls}")
            
            # Check: Halluzinations-Keywords?
            found_hallucinations = []
            for keyword in test['should_not_contain']:
                if keyword.lower() in response:
                    found_hallucinations.append(keyword)
            
            if found_hallucinations:
                passed = False
                issues.append(f"Halluzination detected: {found_hallucinations}")
                print(f"[ERROR] HALLUZINATION: {found_hallucinations}")
            else:
                print(f"[OK] Keine Halluzinationen erkannt")
            
            # Ergebnis
            result = {
                "test": test['question'],
                "passed": passed,
                "issues": issues,
                "response": data.get('response', '')
            }
            results.append(result)
            
            if passed:
                print(f"\n[SUCCESS] Test bestanden!")
            else:
                print(f"\n[FAILED] Test nicht bestanden:")
                for issue in issues:
                    print(f"  - {issue}")
        else:
            print(f"[FEHLER] Status {r.status_code}")
            if r.status_code == 202:
                print("Modell wird geladen...")
            results.append({
                "test": test['question'],
                "passed": False,
                "issues": [f"HTTP {r.status_code}"],
                "response": ""
            })
            
    except Exception as e:
        print(f"[FEHLER] {e}")
        results.append({
            "test": test['question'],
            "passed": False,
            "issues": [str(e)],
            "response": ""
        })
    
    time.sleep(2)  # Kurze Pause zwischen Tests

# Zusammenfassung
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)

passed_count = sum(1 for r in results if r['passed'])
total_count = len(results)

print(f"\nErgebnis: {passed_count}/{total_count} Tests bestanden")

for i, result in enumerate(results, 1):
    status = "[PASS]" if result['passed'] else "[FAIL]"
    print(f"\n{i}. {status}: {result['test'][:50]}...")
    if result['issues']:
        for issue in result['issues']:
            print(f"   - {issue}")

if passed_count == total_count:
    print("\n[SUCCESS] Alle Tests bestanden - Kein Halluzinations-Problem!")
else:
    print(f"\n[PROBLEM] {total_count - passed_count} Test(s) fehlgeschlagen")
    print("Debug-Logs: g:\\04-CODING\\Local Ai\\.cursor\\debug.log")

print("=" * 60)

