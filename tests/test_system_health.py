"""
Einfacher, robuster System-Test für Local AI
Prüft alle kritischen Komponenten
"""
import sys
import requests
import time
from pathlib import Path

# API URLs
API_BASE = "http://127.0.0.1:8000"
MODEL_SERVICE_BASE = "http://127.0.0.1:8001"

def test_server_reachable():
    """Prüft ob Server erreichbar ist"""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=5)
        assert response.status_code == 200, f"Server antwortet mit {response.status_code}"
        print("[OK] Server erreichbar")
        return True
    except Exception as e:
        print(f"[FEHLER] Server nicht erreichbar: {e}")
        return False

def test_model_service_reachable():
    """Prüft ob Model-Service erreichbar ist"""
    try:
        response = requests.get(f"{MODEL_SERVICE_BASE}/status", timeout=5)
        assert response.status_code == 200, f"Model-Service antwortet mit {response.status_code}"
        print("[OK] Model-Service erreichbar")
        return True
    except Exception as e:
        print(f"[FEHLER] Model-Service nicht erreichbar: {e}")
        return False

def test_health_endpoint():
    """Prüft ob Health-Endpoint funktioniert"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        assert response.status_code == 200, f"Health-Endpoint antwortet mit {response.status_code}"
        health = response.json()
        assert "status" in health, "Health-Response enthält kein 'status' Feld"
        assert "models" in health, "Health-Response enthält kein 'models' Feld"
        print(f"[OK] Health-Endpoint funktioniert (Status: {health.get('status')})")
        return True
    except Exception as e:
        print(f"[FEHLER] Health-Endpoint fehlgeschlagen: {e}")
        return False

def test_text_model_health():
    """Prüft ob Text-Modell funktioniert"""
    try:
        # Health-Check
        response = requests.get(f"{API_BASE}/health", timeout=10)
        assert response.status_code == 200, f"Health-Endpoint antwortet mit {response.status_code}"
        health = response.json()
        
        if health["models"]["text"]["loaded"]:
            if health["models"]["text"]["healthy"]:
                response_time = health["models"]["text"].get("response_time_ms", 0)
                print(f"[OK] Text-Modell gesund ({response_time:.0f}ms)")
                return True
            else:
                error = health["models"]["text"].get("error", "Unbekannter Fehler")
                print(f"[FEHLER] Text-Modell nicht gesund: {error}")
                return False
        else:
            print("[WARNUNG] Text-Modell nicht geladen")
            return False
    except Exception as e:
        print(f"[FEHLER] Text-Modell Health-Check fehlgeschlagen: {e}")
        return False

def test_audio_model_health():
    """Prüft ob Audio-Modell funktioniert"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        assert response.status_code == 200
        health = response.json()
        
        if health["models"]["audio"]["loaded"]:
            if health["models"]["audio"]["healthy"]:
                print("[OK] Audio-Modell gesund")
                return True
            else:
                error = health["models"]["audio"].get("error", "Unbekannter Fehler")
                print(f"[FEHLER] Audio-Modell nicht gesund: {error}")
                return False
        else:
            print("[WARNUNG] Audio-Modell nicht geladen")
            return False
    except Exception as e:
        print(f"[FEHLER] Audio-Modell Health-Check fehlgeschlagen: {e}")
        return False

def test_basic_chat():
    """Prüft ob Chat-Endpoint funktioniert (einfacher Test)"""
    try:
        # Erstelle eine einfache Chat-Anfrage
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "message": "Test",
                "conversation_id": None,
                "max_length": 50,
                "temperature": 0.3
            },
            timeout=30
        )
        assert response.status_code == 200, f"Chat-Endpoint antwortet mit {response.status_code}"
        data = response.json()
        assert "response" in data, "Chat-Response enthält kein 'response' Feld"
        assert len(data["response"]) > 0, "Chat-Response ist leer"
        print(f"[OK] Chat-Endpoint funktioniert (Response-Länge: {len(data['response'])})")
        return True
    except Exception as e:
        print(f"[FEHLER] Chat-Endpoint fehlgeschlagen: {e}")
        return False

def main():
    """Führt alle Tests aus"""
    print("=" * 60)
    print("System Health Test")
    print("=" * 60)
    print()
    
    results = []
    results.append(("Server erreichbar", test_server_reachable()))
    results.append(("Model-Service erreichbar", test_model_service_reachable()))
    results.append(("Health-Endpoint", test_health_endpoint()))
    results.append(("Text-Modell Health", test_text_model_health()))
    results.append(("Audio-Modell Health", test_audio_model_health()))
    results.append(("Chat-Endpoint", test_basic_chat()))
    
    print()
    print("=" * 60)
    print("Zusammenfassung")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[OK]" if result else "[FEHLER]"
        print(f"{status} {name}")
    
    print()
    print(f"Ergebnis: {passed}/{total} Tests bestanden")
    
    # Exit-Code: 0 bei Erfolg, 1 bei Fehler
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()




