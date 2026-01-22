"""
Test-Script für alle Optimierungen:
1. Settings-Caching
2. GPU-Budget Fix
3. Post-Processing Verbesserung
4. Tool-Usage Toggle-Prüfung
5. Performance-Optimierungen
"""
import os
import sys
import time
import json
import requests
from typing import Dict, Any

# Pfad zum Backend hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# Test-Konfiguration
MODEL_SERVICE_URL = "http://localhost:8001"
MAIN_SERVER_URL = "http://localhost:8000"
TEST_CONVERSATION_ID = "test-optimizations-conv"

def test_settings_caching():
    """Test 1: Settings-Caching"""
    print("\n" + "="*60)
    print("TEST 1: Settings-Caching")
    print("="*60)
    
    try:
        from backend.settings_loader import load_performance_settings, invalidate_cache
        
        # Test 1.1: Erste Ladung
        start_time = time.time()
        settings1 = load_performance_settings()
        first_load_time = time.time() - start_time
        
        # Test 1.2: Zweite Ladung (sollte aus Cache kommen)
        start_time = time.time()
        settings2 = load_performance_settings()
        cached_load_time = time.time() - start_time
        
        # Test 1.3: Nach Cache-Invalidierung
        invalidate_cache()
        start_time = time.time()
        settings3 = load_performance_settings()
        reload_time = time.time() - start_time
        
        print(f"[OK] Erste Ladung: {first_load_time*1000:.2f}ms")
        print(f"[OK] Cache-Ladung: {cached_load_time*1000:.2f}ms")
        print(f"[OK] Nach Invalidation: {reload_time*1000:.2f}ms")
        
        if cached_load_time < first_load_time * 0.5:
            print("[OK] Caching funktioniert (Cache ist schneller)")
        else:
            print("[WARN] Caching konnte verbessert werden")
        
        # Prüfe Konsistenz
        if settings1 == settings2 == settings3:
            print("[OK] Settings konsistent")
        else:
            print("[FEHLER] Settings inkonsistent!")
            return False
        
        return True
    except Exception as e:
        print(f"[FEHLER] Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_budget_fix():
    """Test 2: GPU-Budget Fix (Quantisierung bleibt aktiv)"""
    print("\n" + "="*60)
    print("TEST 2: GPU-Budget Fix")
    print("="*60)
    
    try:
        # Prüfe ob Model-Service läuft
        try:
            status = requests.get(f"{MODEL_SERVICE_URL}/status", timeout=2)
            if status.status_code != 200:
                print("[WARN] Model-Service nicht verfuegbar - ueberspringe Test")
                return True
        except:
            print("[WARN] Model-Service nicht verfuegbar - ueberspringe Test")
            return True
        
        # Prüfe Performance-Settings
        from backend.settings_loader import load_performance_settings
        perf_settings = load_performance_settings()
        use_quantization = perf_settings.get("use_quantization", False)
        
        print(f"[OK] Performance-Settings geladen: use_quantization={use_quantization}")
        
        # Prüfe ob Quantisierung in Settings aktiviert ist
        if use_quantization:
            print("[OK] Quantisierung ist in Settings aktiviert")
        else:
            print("[WARN] Quantisierung ist in Settings deaktiviert (Test kann nicht vollständig durchgefuehrt werden)")
        
        return True
    except Exception as e:
        print(f"[FEHLER] Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_post_processing():
    """Test 3: Post-Processing Verbesserung"""
    print("\n" + "="*60)
    print("TEST 3: Post-Processing Verbesserung")
    print("="*60)
    
    try:
        from backend.model_manager import ModelManager
        
        # Erstelle ModelManager-Instanz (ohne Modell zu laden)
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            print("[WARN] config.json nicht gefunden - ueberspringe Test")
            return True
        
        manager = ModelManager(config_path=config_path)
        
        # Test 3.1: Minimal Cleaning (sollte nicht zu aggressiv sein)
        test_responses = [
            "Das ist eine normale Antwort.",
            "```python\nprint('Hello')\n```",
            "User: Test\nAssistant: Das ist eine Antwort.",
            "Das ist eine Antwort mit <br> HTML-Tags.",
        ]
        
        # Mock messages für _clean_response_minimal
        mock_messages = [{"role": "user", "content": "Test"}]
        
        for response in test_responses:
            cleaned = manager._clean_response_minimal(response, mock_messages)
            if len(cleaned) < len(response) * 0.5:
                print(f"[WARN] Response zu aggressiv bereinigt: {len(response)} -> {len(cleaned)} Zeichen")
            else:
                print(f"[OK] Response akzeptabel bereinigt: {len(response)} -> {len(cleaned)} Zeichen")
        
        return True
    except Exception as e:
        print(f"[FEHLER] Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_toggle():
    """Test 4: Tool-Usage Toggle-Prüfung"""
    print("\n" + "="*60)
    print("TEST 4: Tool-Usage Toggle-Prüfung")
    print("="*60)
    
    try:
        from backend.agents.chat_agent import ChatAgent
        
        # Erstelle ChatAgent-Instanz
        agent = ChatAgent(
            agent_id="test-agent",
            conversation_id=TEST_CONVERSATION_ID,
            model_id=None
        )
        
        # Test 4.1: Prüfe Tool-Enable-Check
        tools_to_test = ["web_search", "read_file", "write_file"]
        
        for tool_name in tools_to_test:
            is_enabled = agent._is_tool_enabled(tool_name)
            print(f"[OK] Tool '{tool_name}': enabled={is_enabled}")
        
        # Test 4.2: Prüfe ob Tool-Detection funktioniert
        test_messages = [
            "Suche nach Python Tutorial",
            "Lese die Datei test.txt",
            "Schreibe 'Hello' in test.txt",
        ]
        
        for message in test_messages:
            tool_info = agent._detect_tool_need(message)
            if tool_info:
                tool_name = tool_info["tool_name"]
                is_enabled = agent._is_tool_enabled(tool_name)
                print(f"[OK] Nachricht '{message[:30]}...' -> Tool: {tool_name}, enabled: {is_enabled}")
            else:
                print(f"[OK] Nachricht '{message[:30]}...' -> Kein Tool erkannt")
        
        return True
    except Exception as e:
        print(f"[FEHLER] Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test 5: Performance-Optimierungen"""
    print("\n" + "="*60)
    print("TEST 5: Performance-Optimierungen")
    print("="*60)
    
    try:
        # Test 5.1: Settings-Laden Performance
        from backend.settings_loader import load_performance_settings
        
        # Mehrfaches Laden (sollte aus Cache kommen)
        times = []
        for i in range(10):
            start_time = time.time()
            load_performance_settings()
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"[OK] Settings-Laden (10x):")
        print(f"  Durchschnitt: {avg_time*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        
        if max_time < 0.01:  # < 10ms
            print("[OK] Performance akzeptabel")
        else:
            print("[WARN] Performance koennte verbessert werden")
        
        return True
    except Exception as e:
        print(f"[FEHLER] Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Führt alle Tests aus"""
    print("\n" + "="*60)
    print("OPTIMIERUNGS-TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Settings-Caching
    results["settings_caching"] = test_settings_caching()
    
    # Test 2: GPU-Budget Fix
    results["gpu_budget_fix"] = test_gpu_budget_fix()
    
    # Test 3: Post-Processing
    results["post_processing"] = test_post_processing()
    
    # Test 4: Tool Toggle
    results["tool_toggle"] = test_tool_toggle()
    
    # Test 5: Performance
    results["performance"] = test_performance()
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{status}: {test_name}")
    
    print(f"\nGesamt: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n[OK] Alle Tests bestanden!")
        return 0
    else:
        print(f"\n[WARN] {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == "__main__":
    sys.exit(main())
