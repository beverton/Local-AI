"""
Performance-Test: Verschiedene max_length Werte mit gleichem Prompt
Messen: Tokens pro Sekunde, Gesamtzeit, Qualität der Antwort

Basierend auf Online-Recherche:
- Qwen 2.5-7B: ~97.90 tokens/sec Durchschnitt (variiert je nach Hardware)
- Optimal: Balance zwischen Response-Länge und Generierungszeit
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Model Service URL
MODEL_SERVICE_URL = "http://127.0.0.1:8001"

# Test-Prompt (sollte für verschiedene Längen geeignet sein)
TEST_PROMPT = "Schreibe ein Gedicht mit mindestens 5 Strophen zum Thema: Menschen mit K.I. landen in einer Utopie. Das Gedicht sollte kreativ, inspirierend und detailliert sein."

# Verschiedene max_length Werte zum Testen
# Basierend auf Recherche: 100-4000 Tokens ist realistischer Bereich
MAX_LENGTH_TESTS = [
    200,   # Sehr kurz
    500,   # Aktueller Default
    800,   # Realistisch für Gedichte
    1000,  # Mittlere Länge
    1500,  # Längere Antworten
    2000,  # Sehr lang
    3000,  # Extrem lang
    4000   # Maximum für Code
]

# Temperature für konsistente Tests
TEMPERATURE = 0.7

def check_model_service() -> bool:
    """Prüft ob Model Service läuft"""
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def count_tokens_approx(text: str) -> int:
    """Grobe Token-Schätzung: ~1 Token = 4 Zeichen für deutsche Texte"""
    return len(text) // 4

def test_max_length(max_length: int, prompt: str) -> Dict[str, Any]:
    """Testet eine max_length Einstellung"""
    print(f"\n{'='*70}")
    print(f"TEST: max_length={max_length}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Sende Request an Model Service
        payload = {
            "message": prompt,
            "max_length": max_length,
            "temperature": TEMPERATURE,
            "profile": "creative"  # Für kreative Aufgaben
        }
        
        response = requests.post(
            f"{MODEL_SERVICE_URL}/chat",
            json=payload,
            timeout=300  # 5 Minuten Timeout für sehr lange Generierungen
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ FEHLER: Status {response.status_code}")
            print(f"   Fehler: {error_text[:200]}")
            return {
                "max_length": max_length,
                "success": False,
                "error": error_text[:200],
                "elapsed_time": elapsed_time
            }
        
        result = response.json()
        response_text = result.get("response", "")
        
        # Berechne Metriken
        response_length = len(response_text)
        estimated_tokens = count_tokens_approx(response_text)
        tokens_per_second = estimated_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Prüfe ob Antwort vollständig ist (nicht abgeschnitten)
        is_complete = not any(indicator in response_text[-50:].lower() 
                             for indicator in ["...", "unvollständig", "abgeschnitten"])
        
        print(f"✅ SUCCESS nach {elapsed_time:.2f}s")
        print(f"   Response-Länge: {response_length} Zeichen")
        print(f"   Geschätzte Tokens: ~{estimated_tokens}")
        print(f"   Tokens/Sekunde: {tokens_per_second:.2f}")
        print(f"   Vollständig: {'✅' if is_complete else '⚠️ Möglicherweise abgeschnitten'}")
        print(f"   Antwort (erste 150 Zeichen): {response_text[:150]}...")
        
        return {
            "max_length": max_length,
            "success": True,
            "elapsed_time": elapsed_time,
            "response_length": response_length,
            "estimated_tokens": estimated_tokens,
            "tokens_per_second": tokens_per_second,
            "is_complete": is_complete,
            "response_preview": response_text[:200]
        }
        
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"⏱️ TIMEOUT nach {elapsed_time:.2f}s")
        return {
            "max_length": max_length,
            "success": False,
            "error": "Timeout",
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ FEHLER nach {elapsed_time:.2f}s: {e}")
        return {
            "max_length": max_length,
            "success": False,
            "error": str(e)[:200],
            "elapsed_time": elapsed_time
        }

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analysiert Testergebnisse und findet optimale Einstellungen"""
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        return {"error": "Keine erfolgreichen Tests"}
    
    # Finde beste Performance (höchste Tokens/Sekunde)
    best_performance = max(successful_results, key=lambda x: x.get("tokens_per_second", 0))
    
    # Finde beste Balance (gute Tokens/Sekunde + vollständige Antworten)
    complete_results = [r for r in successful_results if r.get("is_complete", False)]
    if complete_results:
        best_balance = max(complete_results, 
                          key=lambda x: x.get("tokens_per_second", 0) * (1 if x.get("is_complete") else 0.5))
    else:
        best_balance = best_performance
    
    # Statistiken
    tokens_per_second_values = [r.get("tokens_per_second", 0) for r in successful_results]
    avg_tokens_per_second = statistics.mean(tokens_per_second_values) if tokens_per_second_values else 0
    
    return {
        "total_tests": len(results),
        "successful_tests": len(successful_results),
        "best_performance": {
            "max_length": best_performance.get("max_length"),
            "tokens_per_second": best_performance.get("tokens_per_second"),
            "elapsed_time": best_performance.get("elapsed_time")
        },
        "best_balance": {
            "max_length": best_balance.get("max_length"),
            "tokens_per_second": best_balance.get("tokens_per_second"),
            "elapsed_time": best_balance.get("elapsed_time"),
            "is_complete": best_balance.get("is_complete")
        },
        "average_tokens_per_second": avg_tokens_per_second,
        "recommendation": {
            "for_short_answers": min([r for r in successful_results if r.get("max_length", 0) <= 500], 
                                    key=lambda x: x.get("elapsed_time", float('inf')), 
                                    default={}).get("max_length", 500),
            "for_medium_answers": best_balance.get("max_length", 1000),
            "for_long_answers": max([r for r in successful_results], 
                                   key=lambda x: x.get("max_length", 0)).get("max_length", 2000)
        }
    }

def main():
    """Hauptfunktion"""
    print("="*70)
    print("MAX_LENGTH PERFORMANCE TEST")
    print("="*70)
    print(f"Test-Prompt: {TEST_PROMPT[:80]}...")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Test-Werte: {MAX_LENGTH_TESTS}")
    print("="*70)
    
    # Prüfe Model Service
    print("\nPrüfe Model Service...")
    if not check_model_service():
        print("❌ FEHLER: Model Service läuft nicht!")
        print("   Bitte starte den Model Service mit: python backend/model_service.py")
        return
    
    print("✅ Model Service läuft")
    
    # Führe Tests durch
    results = []
    start_time = time.time()
    
    for max_length in MAX_LENGTH_TESTS:
        result = test_max_length(max_length, TEST_PROMPT)
        results.append(result)
        
        # Kurze Pause zwischen Tests (GPU abkühlen lassen)
        if max_length != MAX_LENGTH_TESTS[-1]:  # Nicht nach letztem Test
            print("\n⏸️  Pause 5 Sekunden...")
            time.sleep(5)
    
    total_time = time.time() - start_time
    
    # Analysiere Ergebnisse
    print("\n" + "="*70)
    print("ANALYSE")
    print("="*70)
    
    analysis = analyze_results(results)
    
    print(f"\nGesamt-Tests: {analysis.get('total_tests', 0)}")
    print(f"Erfolgreich: {analysis.get('successful_tests', 0)}")
    print(f"Durchschnittliche Tokens/Sekunde: {analysis.get('average_tokens_per_second', 0):.2f}")
    
    print(f"\n🏆 Beste Performance:")
    best_perf = analysis.get("best_performance", {})
    print(f"   max_length: {best_perf.get('max_length')}")
    print(f"   Tokens/Sekunde: {best_perf.get('tokens_per_second', 0):.2f}")
    print(f"   Zeit: {best_perf.get('elapsed_time', 0):.2f}s")
    
    print(f"\n⚖️  Beste Balance (Performance + Vollständigkeit):")
    best_balance = analysis.get("best_balance", {})
    print(f"   max_length: {best_balance.get('max_length')}")
    print(f"   Tokens/Sekunde: {best_balance.get('tokens_per_second', 0):.2f}")
    print(f"   Zeit: {best_balance.get('elapsed_time', 0):.2f}s")
    print(f"   Vollständig: {'✅' if best_balance.get('is_complete') else '⚠️'}")
    
    print(f"\n💡 Empfehlungen:")
    rec = analysis.get("recommendation", {})
    print(f"   Kurze Antworten: max_length={rec.get('for_short_answers', 500)}")
    print(f"   Mittlere Antworten: max_length={rec.get('for_medium_answers', 1000)}")
    print(f"   Lange Antworten: max_length={rec.get('for_long_answers', 2000)}")
    
    # Speichere Ergebnisse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_max_length_performance_{timestamp}.json"
    
    output_data = {
        "test_info": {
            "timestamp": timestamp,
            "prompt": TEST_PROMPT,
            "temperature": TEMPERATURE,
            "test_values": MAX_LENGTH_TESTS
        },
        "results": results,
        "analysis": analysis,
        "total_test_time": total_time
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Ergebnisse gespeichert: {output_file}")
    print("\n" + "="*70)
    print("TEST ABGESCHLOSSEN")
    print("="*70)

if __name__ == "__main__":
    main()
