#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Qwen2.5-7B-Instruct Web Search Accuracy"""

import requests
import json
import time
import sys

# Stelle sicher, dass stdout UTF-8 nutzt
sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://localhost:8000"

def test_web_search(query: str, expected_url: str = None, expected_keywords: list = None):
    """Test web search with Qwen2.5-7B-Instruct"""
    print(f"\n{'='*80}")
    print(f"TEST: {query}")
    print(f"{'='*80}")
    
    # API Call
    payload = {
        "message": query,
        "model": "qwen-2.5-7b-instruct"  # Nutze das neue Modell
    }
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        
        ai_response = result.get("response", "")
        print(f"\n✅ AI Response:\n{ai_response}\n")
        
        # Validierung
        passed = True
        
        if expected_url:
            if expected_url.lower() in ai_response.lower():
                print(f"✅ URL gefunden: {expected_url}")
            else:
                print(f"❌ URL NICHT gefunden: {expected_url}")
                passed = False
        
        if expected_keywords:
            for keyword in expected_keywords:
                if keyword.lower() in ai_response.lower():
                    print(f"✅ Keyword gefunden: {keyword}")
                else:
                    print(f"⚠️ Keyword nicht gefunden: {keyword}")
        
        return passed
        
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Server antwortet nicht")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("="*80)
    print("Qwen2.5-7B-Instruct Web Search Accuracy Test")
    print("="*80)
    
    # Test 1: Brot und Spüle (Bekannt problematischer Test)
    test1_passed = test_web_search(
        query="Wer ist Brot und Spüle? Suche nach einer Website.",
        expected_url="https://www.brotundspuele.de",
        expected_keywords=["Brot und Spüle", "Website"]
    )
    
    # Test 2: Aktuelle Faktenfrage
    test2_passed = test_web_search(
        query="Wer ist der aktuelle Bundeskanzler von Deutschland?",
        expected_keywords=["Olaf Scholz"]
    )
    
    # Test 3: Technische Frage
    test3_passed = test_web_search(
        query="Was ist Python? Suche nach Informationen.",
        expected_keywords=["Python", "Programmiersprache"]
    )
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    print(f"Test 1 (Brot und Spüle): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Bundeskanzler): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (Python): {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n{'✅ ALLE TESTS BESTANDEN' if all_passed else '❌ EINIGE TESTS FEHLGESCHLAGEN'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

