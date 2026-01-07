"""
Integration Tests für API-Endpunkte mit Log-Analyse
"""
import sys
import os
import time
from pathlib import Path

# Füge backend zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))

from log_manager import LogManager
from system_checker import SystemChecker
from test_runner import generate_debug_report


def test_text_model_load_endpoint():
    """Test: /models/load Endpunkt"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_text_model_load_endpoint", "start", "start", {})
    
    try:
        # Prüfe ob Endpunkt erreichbar ist
        endpoint_check = system_checker.check_api_endpoint("/models", method="GET")
        log_manager.write_test_log("test_text_model_load_endpoint", "endpoint_check", "info", endpoint_check)
        
        if endpoint_check["status"] != "ok":
            log_manager.write_test_log("test_text_model_load_endpoint", "end", "skipped", {
                "reason": "API nicht erreichbar"
            })
            return
        
        # Hole verfügbare Modelle
        models_response = system_checker.check_api_endpoint("/models", method="GET")
        if models_response["status"] != "ok":
            raise AssertionError(f"Konnte Modelle nicht abrufen: {models_response['error']}")
        
        available_models = models_response["response"].get("models", {})
        text_models = {k: v for k, v in available_models.items() 
                      if v.get("type") != "image" and v.get("type") != "audio"}
        
        if not text_models:
            log_manager.write_test_log("test_text_model_load_endpoint", "end", "skipped", {
                "reason": "No text models available"
            })
            return
        
        test_model_id = list(text_models.keys())[0]
        log_manager.write_test_log("test_text_model_load_endpoint", "load_request", "info", {
            "model_id": test_model_id
        })
        
        # Teste Load-Endpunkt
        load_response = system_checker.check_api_endpoint("/models/load", method="POST", 
                                                          data={"model_id": test_model_id})
        log_manager.write_test_log("test_text_model_load_endpoint", "load_response", "info", load_response)
        
        # Prüfe Status
        time.sleep(1)  # Warte kurz
        status_response = system_checker.check_api_endpoint("/models/load/status", method="GET")
        log_manager.write_test_log("test_text_model_load_endpoint", "status_check", "info", status_response)
        
        assert load_response["status"] == "ok" or load_response["status_code"] in [200, 202], \
            f"Load-Endpunkt sollte funktionieren: {load_response['error']}"
        
        log_manager.write_test_log("test_text_model_load_endpoint", "end", "success", {})
    
    except Exception as e:
        log_manager.write_test_log("test_text_model_load_endpoint", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_text_model_load_endpoint",
            "exception": str(e)
        })
        
        debug_report = generate_debug_report("test_text_model_load_endpoint", problems)
        log_manager.write_test_log("test_text_model_load_endpoint", "debug_report", "info", debug_report)
        
        raise


def test_image_model_load_endpoint():
    """Test: /image/models/load Endpunkt"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_image_model_load_endpoint", "start", "start", {})
    
    try:
        # Prüfe ob Endpunkt erreichbar ist
        endpoint_check = system_checker.check_api_endpoint("/image/models", method="GET")
        log_manager.write_test_log("test_image_model_load_endpoint", "endpoint_check", "info", endpoint_check)
        
        if endpoint_check["status"] != "ok":
            log_manager.write_test_log("test_image_model_load_endpoint", "end", "skipped", {
                "reason": "Image API nicht verfügbar"
            })
            return
        
        # Hole verfügbare Modelle
        models_response = system_checker.check_api_endpoint("/image/models", method="GET")
        if models_response["status"] != "ok":
            log_manager.write_test_log("test_image_model_load_endpoint", "end", "skipped", {
                "reason": "Konnte Image-Modelle nicht abrufen"
            })
            return
        
        available_models = models_response["response"].get("models", {})
        if not available_models:
            log_manager.write_test_log("test_image_model_load_endpoint", "end", "skipped", {
                "reason": "No image models available"
            })
            return
        
        test_model_id = list(available_models.keys())[0]
        log_manager.write_test_log("test_image_model_load_endpoint", "load_request", "info", {
            "model_id": test_model_id
        })
        
        # Teste Load-Endpunkt
        load_response = system_checker.check_api_endpoint("/image/models/load", method="POST",
                                                          data={"model_id": test_model_id})
        log_manager.write_test_log("test_image_model_load_endpoint", "load_response", "info", load_response)
        
        # Prüfe Status
        time.sleep(1)
        status_response = system_checker.check_api_endpoint("/image/models/load/status", method="GET")
        log_manager.write_test_log("test_image_model_load_endpoint", "status_check", "info", status_response)
        
        assert load_response["status"] == "ok" or load_response["status_code"] in [200, 202], \
            f"Load-Endpunkt sollte funktionieren: {load_response['error']}"
        
        log_manager.write_test_log("test_image_model_load_endpoint", "end", "success", {})
    
    except Exception as e:
        log_manager.write_test_log("test_image_model_load_endpoint", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_image_model_load_endpoint",
            "exception": str(e),
            "model_type": "image"
        })
        
        debug_report = generate_debug_report("test_image_model_load_endpoint", problems)
        log_manager.write_test_log("test_image_model_load_endpoint", "debug_report", "info", debug_report)
        
        raise


def test_audio_model_load_endpoint():
    """Test: /audio/models/load Endpunkt"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_audio_model_load_endpoint", "start", "start", {})
    
    try:
        # Prüfe ob Endpunkt erreichbar ist
        endpoint_check = system_checker.check_api_endpoint("/audio/models", method="GET")
        log_manager.write_test_log("test_audio_model_load_endpoint", "endpoint_check", "info", endpoint_check)
        
        if endpoint_check["status"] != "ok":
            log_manager.write_test_log("test_audio_model_load_endpoint", "end", "skipped", {
                "reason": "Audio API nicht verfügbar"
            })
            return
        
        # Hole verfügbare Modelle
        models_response = system_checker.check_api_endpoint("/audio/models", method="GET")
        if models_response["status"] != "ok":
            log_manager.write_test_log("test_audio_model_load_endpoint", "end", "skipped", {
                "reason": "Konnte Audio-Modelle nicht abrufen"
            })
            return
        
        available_models = models_response["response"].get("models", {})
        if not available_models:
            log_manager.write_test_log("test_audio_model_load_endpoint", "end", "skipped", {
                "reason": "No audio models available"
            })
            return
        
        test_model_id = list(available_models.keys())[0]
        log_manager.write_test_log("test_audio_model_load_endpoint", "load_request", "info", {
            "model_id": test_model_id
        })
        
        # Teste Load-Endpunkt
        load_response = system_checker.check_api_endpoint("/audio/models/load", method="POST",
                                                          data={"model_id": test_model_id})
        log_manager.write_test_log("test_audio_model_load_endpoint", "load_response", "info", load_response)
        
        # Prüfe Status
        time.sleep(1)
        status_response = system_checker.check_api_endpoint("/audio/models/load/status", method="GET")
        log_manager.write_test_log("test_audio_model_load_endpoint", "status_check", "info", status_response)
        
        assert load_response["status"] == "ok" or load_response["status_code"] in [200, 202], \
            f"Load-Endpunkt sollte funktionieren: {load_response['error']}"
        
        log_manager.write_test_log("test_audio_model_load_endpoint", "end", "success", {})
    
    except Exception as e:
        log_manager.write_test_log("test_audio_model_load_endpoint", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_audio_model_load_endpoint",
            "exception": str(e),
            "model_type": "audio"
        })
        
        debug_report = generate_debug_report("test_audio_model_load_endpoint", problems)
        log_manager.write_test_log("test_audio_model_load_endpoint", "debug_report", "info", debug_report)
        
        raise


def test_ensure_text_model_loaded_logic():
    """Test: ensure_text_model_loaded Logik über API"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_ensure_text_model_loaded", "start", "start", {})
    
    try:
        # Prüfe Modell-Status
        state = system_checker.check_model_state("text")
        log_manager.write_test_log("test_ensure_text_model_loaded", "initial_state", "info", state)
        
        # Teste Chat-Endpunkt (sollte ensure_text_model_loaded aufrufen)
        # Hole verfügbare Modelle
        models_response = system_checker.check_api_endpoint("/models", method="GET")
        if models_response["status"] != "ok":
            log_manager.write_test_log("test_ensure_text_model_loaded", "end", "skipped", {
                "reason": "Konnte Modelle nicht abrufen"
            })
            return
        
        available_models = models_response["response"].get("models", {})
        text_models = {k: v for k, v in available_models.items() 
                      if v.get("type") != "image" and v.get("type") != "audio"}
        
        if not text_models:
            log_manager.write_test_log("test_ensure_text_model_loaded", "end", "skipped", {
                "reason": "No text models available"
            })
            return
        
        test_model_id = list(text_models.keys())[0]
        
        # Teste Chat-Endpunkt (sollte Modell automatisch laden)
        chat_response = system_checker.check_api_endpoint("/chat", method="POST", data={
            "message": "Test",
            "conversation_id": None
        })
        
        log_manager.write_test_log("test_ensure_text_model_loaded", "chat_response", "info", chat_response)
        
        # Prüfe ob HTTP 202 (Modell lädt) oder 200 (Modell geladen)
        assert chat_response["status_code"] in [200, 202], \
            f"Chat-Endpunkt sollte 200 oder 202 zurückgeben, bekam {chat_response['status_code']}"
        
        log_manager.write_test_log("test_ensure_text_model_loaded", "end", "success", {})
    
    except Exception as e:
        log_manager.write_test_log("test_ensure_text_model_loaded", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_ensure_text_model_loaded",
            "exception": str(e),
            "model_type": "text"
        })
        
        debug_report = generate_debug_report("test_ensure_text_model_loaded", problems)
        log_manager.write_test_log("test_ensure_text_model_loaded", "debug_report", "info", debug_report)
        
        raise










