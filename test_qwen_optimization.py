"""
Test-Script für Qwen-Modell-Optimierung
Führt 10 verschiedene Tests durch und analysiert die Ergebnisse
"""
import requests
import time
import json
from datetime import datetime

API_BASE = "http://127.0.0.1:8001"

# Test-Fragen (verschiedene Schwierigkeitsgrade)
TEST_QUESTIONS = [
    "was ist 2+2?",
    "was ist 10-5?",
    "was ist 3*4?",
    "was ist 20/4?",
    "was ist die Hauptstadt von Deutschland?",
    "wie viele Tage hat eine Woche?",
    "was ist 15+25?",
    "was ist die Quadratwurzel von 16?",
    "was ist 100-50?",
    "was ist 7*8?"
]

def test_chat(question, profile="default", timeout=60):
    """Führt einen Chat-Test durch"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": question,
                "profile": profile
            },
            timeout=timeout
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "question": question,
                "response": result.get("response", ""),
                "duration": duration,
                "response_length": len(result.get("response", "")),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "question": question,
                "error": f"Status {response.status_code}: {response.text}",
                "duration": duration,
                "status_code": response.status_code
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "question": question,
            "error": f"Timeout nach {timeout}s",
            "duration": time.time() - start_time,
            "status_code": None
        }
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "duration": time.time() - start_time,
            "status_code": None
        }

def analyze_results(results):
    """Analysiert die Testergebnisse"""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if not successful:
        return {
            "total": len(results),
            "successful": 0,
            "failed": len(failed),
            "success_rate": 0.0,
            "avg_duration": None,
            "avg_response_length": None,
            "errors": [r["error"] for r in failed]
        }
    
    durations = [r["duration"] for r in successful]
    lengths = [r["response_length"] for r in successful]
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100,
        "avg_duration": sum(durations) / len(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "avg_response_length": sum(lengths) / len(lengths),
        "min_response_length": min(lengths),
        "max_response_length": max(lengths),
        "errors": [r["error"] for r in failed] if failed else []
    }

def main():
    print("=" * 60)
    print("Qwen-Modell Optimierungs-Test")
    print("=" * 60)
    print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Anzahl Tests: {len(TEST_QUESTIONS)}")
    print("=" * 60)
    print()
    
    # Prüfe ob Service läuft
    try:
        status = requests.get(f"{API_BASE}/status", timeout=5)
        if status.status_code != 200:
            print(f"FEHLER: Service antwortet nicht (Status {status.status_code})")
            return
        print("[OK] Service ist erreichbar")
    except Exception as e:
        print(f"FEHLER: Service nicht erreichbar: {e}")
        return
    
    print()
    print("Starte Tests...")
    print("-" * 60)
    
    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Teste: {question}")
        result = test_chat(question)
        results.append(result)
        
        if result["success"]:
            print(f"  [OK] Erfolg ({result['duration']:.2f}s, {result['response_length']} Zeichen)")
            try:
                response_preview = result['response'][:100].encode('ascii', 'ignore').decode('ascii')
                print(f"  Antwort: {response_preview}...")
            except:
                print(f"  Antwort: [Enthält Sonderzeichen, Länge: {result['response_length']} Zeichen]")
        else:
            print(f"  [FEHLER] {result.get('error', 'Unknown error')}")
        print()
        
        # Kurze Pause zwischen Tests
        time.sleep(1)
    
    print("-" * 60)
    print("Analyse der Ergebnisse:")
    print("-" * 60)
    
    analysis = analyze_results(results)
    print(f"Gesamt: {analysis['total']} Tests")
    print(f"Erfolgreich: {analysis['successful']} ({analysis['success_rate']:.1f}%)")
    print(f"Fehlgeschlagen: {analysis['failed']}")
    
    if analysis['successful'] > 0:
        print()
        print("Dauer-Statistiken:")
        print(f"  Durchschnitt: {analysis['avg_duration']:.2f}s")
        print(f"  Minimum: {analysis['min_duration']:.2f}s")
        print(f"  Maximum: {analysis['max_duration']:.2f}s")
        print()
        print("Antwort-Längen:")
        print(f"  Durchschnitt: {analysis['avg_response_length']:.0f} Zeichen")
        print(f"  Minimum: {analysis['min_response_length']} Zeichen")
        print(f"  Maximum: {analysis['max_response_length']} Zeichen")
    
    if analysis['errors']:
        print()
        print("Fehler:")
        for error in set(analysis['errors']):
            print(f"  - {error}")
    
    # Speichere Ergebnisse
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"Ergebnisse gespeichert in: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
