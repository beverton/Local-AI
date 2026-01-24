"""
Umfassender Test für:
A: Qualität der Antworten
B: Performance
C: Streaming-Implementierung
"""
import os
import sys
import time
import json
import requests
from typing import Dict, Any, List
import threading

# Pfad zum Backend hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# Test-Konfiguration
MAIN_SERVER_URL = "http://localhost:8000"
TEST_CONVERSATION_ID = "test-quality-perf-stream"

def test_response_quality():
    """Test A: Qualität der Antworten"""
    print("\n" + "="*60)
    print("TEST A: Qualität der Antworten")
    print("="*60)
    
    test_cases = [
        {
            "message": "Erkläre mir den Wasserkreislauf in 3 Sätzen.",
            "expected_min_length": 50,
            "expected_max_length": 500,
            "should_be_complete": True
        },
        {
            "message": "Was ist Python?",
            "expected_min_length": 20,
            "expected_max_length": 1000,
            "should_be_complete": True
        },
        {
            "message": "Schreibe ein Python-Programm das 'Hello World' ausgibt.",
            "expected_min_length": 30,
            "expected_max_length": 2000,
            "should_be_complete": True,
            "should_contain_code": True
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['message'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{MAIN_SERVER_URL}/chat",
                json={
                    "message": test_case["message"],
                    "conversation_id": TEST_CONVERSATION_ID,
                    "max_length": 2048,
                    "temperature": 0.3
                },
                timeout=60
            )
            duration = time.time() - start_time
            
            if response.status_code != 200:
                print(f"[FEHLER] HTTP {response.status_code}: {response.text}")
                results.append({"test": i+1, "passed": False, "error": f"HTTP {response.status_code}"})
                continue
            
            data = response.json()
            answer = data.get("response", "")
            
            # Qualitätsprüfungen
            length = len(answer)
            word_count = len(answer.split())
            
            print(f"  Dauer: {duration:.2f}s")
            print(f"  Länge: {length} Zeichen, {word_count} Wörter")
            
            # Prüfe ob Antwort vollständig ist (nicht mitten im Wort abgeschnitten)
            is_complete = True
            if answer and answer[-1] not in ['.', '!', '?', '\n']:
                # Prüfe ob letztes Wort vollständig ist
                last_word = answer.split()[-1] if answer.split() else ""
                if len(last_word) < 3:  # Sehr kurzes Wort könnte abgeschnitten sein
                    is_complete = False
                elif any(c.isalnum() for c in last_word[-1]):  # Endet mit Buchstabe/Zahl
                    # Prüfe ob es ein vollständiges Wort sein könnte
                    if len(answer) > test_case.get("expected_min_length", 0) * 0.8:
                        is_complete = True  # Wahrscheinlich vollständig
                    else:
                        is_complete = False
            
            # Prüfe Länge
            length_ok = (test_case.get("expected_min_length", 0) <= length <= test_case.get("expected_max_length", 10000))
            
            # Prüfe Code-Block wenn erwartet
            has_code = "```" in answer or "def " in answer or "print(" in answer
            code_ok = not test_case.get("should_contain_code", False) or has_code
            
            passed = is_complete and length_ok and code_ok
            
            if passed:
                print(f"  [OK] Qualität akzeptabel")
            else:
                print(f"  [WARN] Qualitätsprobleme:")
                if not is_complete:
                    print(f"    - Antwort scheint unvollständig (endet mit: '{answer[-20:]}')")
                if not length_ok:
                    print(f"    - Länge außerhalb erwartetem Bereich ({length} Zeichen)")
                if not code_ok:
                    print(f"    - Code-Block fehlt")
            
            results.append({
                "test": i+1,
                "passed": passed,
                "duration": duration,
                "length": length,
                "word_count": word_count,
                "is_complete": is_complete,
                "length_ok": length_ok,
                "code_ok": code_ok
            })
            
        except Exception as e:
            print(f"  [FEHLER] {e}")
            results.append({"test": i+1, "passed": False, "error": str(e)})
    
    return results


def test_performance():
    """Test B: Performance"""
    print("\n" + "="*60)
    print("TEST B: Performance")
    print("="*60)
    
    # Test 1: Mehrere Requests hintereinander (sollte durch Caching schneller werden)
    print("\nTest 1: Mehrere Requests (Caching-Test)")
    
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response = requests.post(
                f"{MAIN_SERVER_URL}/chat",
                json={
                    "message": "Was ist 2+2?",
                    "conversation_id": f"{TEST_CONVERSATION_ID}-perf-{i}",
                    "max_length": 512,
                    "temperature": 0.3
                },
                timeout=60
            )
            duration = time.time() - start_time
            times.append(duration)
            print(f"  Request {i+1}: {duration:.2f}s")
        except Exception as e:
            print(f"  Request {i+1}: FEHLER - {e}")
            times.append(None)
    
    if times and all(t is not None for t in times):
        avg_time = sum(times) / len(times)
        print(f"  Durchschnitt: {avg_time:.2f}s")
        
        # Prüfe ob spätere Requests schneller sind (durch Caching)
        if len(times) >= 3:
            first_half = sum(times[:2]) / 2
            second_half = sum(times[2:]) / len(times[2:])
            if second_half < first_half * 0.9:
                print(f"  [OK] Performance-Verbesserung durch Caching erkannt")
            else:
                print(f"  [INFO] Keine deutliche Performance-Verbesserung durch Caching")
    
    # Test 2: Status-Check Performance
    print("\nTest 2: Status-Check Performance")
    status_times = []
    for i in range(10):
        start_time = time.time()
        try:
            requests.get(f"{MAIN_SERVER_URL}/status", timeout=5)
            status_times.append(time.time() - start_time)
        except:
            pass
    
    if status_times:
        avg_status_time = sum(status_times) / len(status_times)
        print(f"  Durchschnitt: {avg_status_time*1000:.2f}ms")
        if avg_status_time < 0.1:
            print(f"  [OK] Status-Checks schnell genug")
        else:
            print(f"  [WARN] Status-Checks könnten schneller sein")
    
    return {"request_times": times, "status_times": status_times}


def test_streaming():
    """Test C: Streaming-Implementierung"""
    print("\n" + "="*60)
    print("TEST C: Streaming-Implementierung")
    print("="*60)
    
    print("\nTest: Streaming-Endpoint")
    
    try:
        chunks_received = []
        chunk_times = []
        full_response = ""
        
        start_time = time.time()
        
        response = requests.post(
            f"{MAIN_SERVER_URL}/chat/stream",
            json={
                "message": "Erkläre mir was Python ist in 5 Sätzen.",
                "conversation_id": f"{TEST_CONVERSATION_ID}-stream",
                "max_length": 1024,
                "temperature": 0.3
            },
            stream=True,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"  [FEHLER] HTTP {response.status_code}: {response.text}")
            return {"passed": False, "error": f"HTTP {response.status_code}"}
        
        # Parse Server-Sent Events
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Entferne "data: "
                    try:
                        data = json.loads(data_str)
                        
                        if 'chunk' in data:
                            chunk = data['chunk']
                            chunks_received.append(chunk)
                            full_response += chunk
                            chunk_times.append(time.time() - start_time)
                            print(f"  Chunk {len(chunks_received)}: '{chunk[:30]}...' (nach {chunk_times[-1]:.2f}s)")
                        
                        if 'done' in data and data['done']:
                            print(f"  [OK] Streaming abgeschlossen")
                            break
                        
                        if 'error' in data:
                            print(f"  [FEHLER] Streaming-Fehler: {data['error']}")
                            return {"passed": False, "error": data['error']}
                    except json.JSONDecodeError:
                        pass
        
        duration = time.time() - start_time
        
        print(f"\n  Gesamt-Dauer: {duration:.2f}s")
        print(f"  Chunks erhalten: {len(chunks_received)}")
        print(f"  Vollständige Antwort: {len(full_response)} Zeichen")
        
        # Prüfe ob Streaming funktioniert hat
        if len(chunks_received) == 0:
            print(f"  [FEHLER] Keine Chunks erhalten!")
            return {"passed": False, "error": "Keine Chunks erhalten"}
        
        if len(chunks_received) == 1 and len(full_response) > 100:
            print(f"  [WARN] Nur ein Chunk erhalten - möglicherweise kein echtes Streaming")
            return {"passed": False, "error": "Nur ein Chunk (kein echtes Streaming)"}
        
        # Prüfe Timing (Chunks sollten über Zeit verteilt sein)
        if len(chunk_times) > 1:
            time_diffs = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
            avg_diff = sum(time_diffs) / len(time_diffs)
            print(f"  Durchschnittliche Zeit zwischen Chunks: {avg_diff:.3f}s")
            
            if avg_diff < 0.01:
                print(f"  [WARN] Chunks kommen zu schnell (simuliertes Streaming?)")
            else:
                print(f"  [OK] Chunks kommen zeitlich verteilt")
        
        # Prüfe ob Antwort vollständig ist
        if len(full_response) < 20:
            print(f"  [FEHLER] Antwort zu kurz")
            return {"passed": False, "error": "Antwort zu kurz"}
        
        print(f"  [OK] Streaming funktioniert")
        return {
            "passed": True,
            "chunks": len(chunks_received),
            "duration": duration,
            "response_length": len(full_response)
        }
        
    except Exception as e:
        print(f"  [FEHLER] {e}")
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("QUALITÄT, PERFORMANCE & STREAMING TESTS")
    print("="*60)
    
    # Prüfe ob Server läuft
    try:
        status = requests.get(f"{MAIN_SERVER_URL}/status", timeout=5)
        if status.status_code != 200:
            print("[FEHLER] Server antwortet nicht korrekt")
            return 1
    except:
        print("[FEHLER] Server nicht erreichbar. Bitte starte den Server zuerst.")
        return 1
    
    results = {}
    
    # Test A: Qualität
    results["quality"] = test_response_quality()
    
    # Test B: Performance
    results["performance"] = test_performance()
    
    # Test C: Streaming
    results["streaming"] = test_streaming()
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    # Qualität
    quality_passed = sum(1 for r in results["quality"] if r.get("passed", False))
    quality_total = len(results["quality"])
    print(f"\nQualität: {quality_passed}/{quality_total} Tests bestanden")
    
    # Performance
    print(f"\nPerformance: Tests durchgeführt (siehe Details oben)")
    
    # Streaming
    streaming_passed = results["streaming"].get("passed", False)
    print(f"\nStreaming: {'[OK]' if streaming_passed else '[FEHLER]'}")
    
    if quality_passed == quality_total and streaming_passed:
        print("\n[OK] Alle kritischen Tests bestanden!")
        return 0
    else:
        print(f"\n[WARN] Einige Tests fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())
