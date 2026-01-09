"""
Systematisches Testing aller Quality Management Optionen
Testet jede Option einzeln und in Kombination
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json
import time
import os

API_BASE = "http://127.0.0.1:8000"
QUALITY_SETTINGS_PATH = "data/quality_settings.json"

def wait_for_server(max_wait=30):
    """Wartet bis Server verfügbar ist"""
    print("Warte auf Server...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/status", timeout=2)
            if response.status_code == 200:
                print("Server ist verfügbar!")
                return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_model_load(max_wait=120):
    """Wartet bis Modell geladen ist"""
    print("Warte auf Modell-Laden...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("text_model_loaded", False):
                    print("Modell ist geladen!")
                    return True
        except:
            pass
        time.sleep(2)
        if i % 10 == 0:
            print(f"  Warte... ({i*2}s)")
    return False

def set_quality_settings(settings):
    """Setzt Quality Settings"""
    try:
        response = requests.post(
            f"{API_BASE}/quality/settings",
            json=settings,
            timeout=5
        )
        if response.status_code == 200:
            print(f"[OK] Settings gesetzt")
            return True
        else:
            print(f"[ERROR] Fehler: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Fehler: {e}")
        return False

def get_quality_settings():
    """Holt aktuelle Quality Settings"""
    try:
        response = requests.get(f"{API_BASE}/quality/settings", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def test_chat(question, timeout=120):
    """Testet eine Chat-Anfrage"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": question,
                "conversation_id": None,
                "max_length": 2048,
                "temperature": 0.3
            },
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            sources = data.get("sources", [])
            quality_info = data.get("quality_info", {})
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "quality_info": quality_info,
                "elapsed": elapsed,
                "length": len(answer)
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}",
                "elapsed": elapsed
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout",
            "elapsed": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": 0
        }

def test_option(option_name, test_questions, description=""):
    """Testet eine einzelne Quality-Option"""
    print(f"\n{'='*60}")
    print(f"TEST: {option_name}")
    if description:
        print(f"Beschreibung: {description}")
    print(f"{'='*60}")
    
    # Setze Option auf True, alle anderen auf False
    settings = {
        "web_validation": False,
        "contradiction_check": False,
        "hallucination_check": False,
        "actuality_check": False,
        "source_quality_check": False,
        "completeness_check": False,
        "auto_web_search": False
    }
    settings[option_name] = True
    
    if not set_quality_settings(settings):
        print(f"[SKIP] Konnte Settings nicht setzen")
        return False
    
    # Warte kurz damit Settings übernommen werden
    time.sleep(1)
    
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Frage {i}/{len(test_questions)}: {question[:50]}...")
        result = test_chat(question)
        results.append(result)
        
        if result["success"]:
            print(f"[OK] Antwort erhalten ({result['elapsed']:.2f}s, {result['length']} Zeichen)")
            if result.get("sources"):
                print(f"  Quellen: {len(result['sources'])}")
            if result.get("quality_info"):
                issues = result["quality_info"].get("issues", [])
                if issues:
                    print(f"  Issues gefunden: {len(issues)}")
        else:
            print(f"[FAIL] {result.get('error', 'Unbekannter Fehler')}")
    
    # Zusammenfassung
    success_count = sum(1 for r in results if r["success"])
    print(f"\n--- Zusammenfassung: {success_count}/{len(results)} erfolgreich")
    
    return success_count == len(results)

def test_combination(combination_name, settings, test_questions):
    """Testet eine Kombination von Quality-Optionen"""
    print(f"\n{'='*60}")
    print(f"TEST KOMBINATION: {combination_name}")
    print(f"{'='*60}")
    
    if not set_quality_settings(settings):
        print(f"[SKIP] Konnte Settings nicht setzen")
        return False
    
    time.sleep(1)
    
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Frage {i}/{len(test_questions)}: {question[:50]}...")
        result = test_chat(question)
        results.append(result)
        
        if result["success"]:
            print(f"[OK] Antwort erhalten ({result['elapsed']:.2f}s, {result['length']} Zeichen)")
        else:
            print(f"[FAIL] {result.get('error', 'Unbekannter Fehler')}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n--- Zusammenfassung: {success_count}/{len(results)} erfolgreich")
    
    return success_count == len(results)

def main():
    """Hauptfunktion"""
    print("="*60)
    print("Quality Management - Systematisches Testing")
    print("="*60)
    
    # Warte auf Server
    if not wait_for_server():
        print("[ERROR] Server nicht verfügbar!")
        return False
    
    # Warte auf Modell
    if not wait_for_model_load():
        print("[WARN] Modell konnte nicht geladen werden - teste trotzdem...")
    
    # Teste jede Option einzeln
    print("\n" + "="*60)
    print("PHASE 1: Einzelne Optionen testen")
    print("="*60)
    
    test_results = {}
    
    # Option 1: auto_web_search
    test_results["auto_web_search"] = test_option(
        "auto_web_search",
        [
            "Was ist die aktuelle Python-Version?",
            "Wann wurde Qwen 2.5 veröffentlicht?"
        ],
        "Web-Search vor Generierung, Quellen in Antwort"
    )
    
    # Option 2: web_validation
    test_results["web_validation"] = test_option(
        "web_validation",
        [
            "Was ist die Hauptstadt von Deutschland?",
            "Wie funktioniert Quantum Computing?"
        ],
        "Response-Validierung und Retry-Logik"
    )
    
    # Option 3: hallucination_check
    test_results["hallucination_check"] = test_option(
        "hallucination_check",
        [
            "Was ist die URL zu Qwen auf HuggingFace?",
            "Wie viele Parameter hat Qwen-2.5-7B?"
        ],
        "URL- und Zahlen-Validierung"
    )
    
    # Option 4: contradiction_check
    test_results["contradiction_check"] = test_option(
        "contradiction_check",
        [
            "Ist Python eine interpretierte oder kompilierte Sprache?",
            "Ist Wasser bei 0°C flüssig oder fest?"
        ],
        "Widerspruchsprüfung"
    )
    
    # Option 5: actuality_check
    test_results["actuality_check"] = test_option(
        "actuality_check",
        [
            "Was ist die neueste Python-Version?",
            "Was sind die aktuellen AI-Trends 2024?"
        ],
        "Aktualitätsprüfung"
    )
    
    # Option 6: source_quality_check
    test_results["source_quality_check"] = test_option(
        "source_quality_check",
        [
            "Was ist Machine Learning?",
            "Erkläre Neural Networks"
        ],
        "Quellen-Qualitätsbewertung"
    )
    
    # Option 7: completeness_check
    test_results["completeness_check"] = test_option(
        "completeness_check",
        [
            "Erkläre vollständig wie Python Decorators funktionieren",
            "Beschreibe den kompletten Prozess von Machine Learning"
        ],
        "Vollständigkeitsprüfung"
    )
    
    # Teste Kombinationen
    print("\n" + "="*60)
    print("PHASE 2: Kombinationen testen")
    print("="*60)
    
    # Kombination 1: auto_web_search + web_validation
    test_results["combo_rag_validation"] = test_combination(
        "RAG + Validation",
        {
            "auto_web_search": True,
            "web_validation": True,
            "contradiction_check": False,
            "hallucination_check": False,
            "actuality_check": False,
            "source_quality_check": False,
            "completeness_check": False
        },
        [
            "Was ist die aktuelle Python-Version?",
            "Erkläre Machine Learning"
        ]
    )
    
    # Kombination 2: auto_web_search + hallucination_check
    test_results["combo_rag_hallucination"] = test_combination(
        "RAG + Hallucination Check",
        {
            "auto_web_search": True,
            "hallucination_check": True,
            "web_validation": False,
            "contradiction_check": False,
            "actuality_check": False,
            "source_quality_check": False,
            "completeness_check": False
        },
        [
            "Was ist die URL zu Qwen auf HuggingFace?",
            "Wie viele Parameter hat Qwen-2.5-7B?"
        ]
    )
    
    # Kombination 3: Alle aktiviert
    test_results["combo_all"] = test_combination(
        "Alle Optionen aktiviert",
        {
            "auto_web_search": True,
            "web_validation": True,
            "contradiction_check": True,
            "hallucination_check": True,
            "actuality_check": True,
            "source_quality_check": True,
            "completeness_check": True
        },
        [
            "Was ist die aktuelle Python-Version?",
            "Erkläre Machine Learning vollständig"
        ]
    )
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("GESAMT-ZUSAMMENFASSUNG")
    print("="*60)
    
    for option, success in test_results.items():
        status = "✓" if success else "❌"
        print(f"{status} {option}")
    
    all_ok = all(test_results.values())
    print(f"\nGesamt: {'✓ Alle Tests bestanden' if all_ok else '❌ Einige Tests fehlgeschlagen'}")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
