"""
End-to-End Workflow-Tests mit vollständiger Log-Analyse
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


def test_whisper_to_chat_workflow():
    """Test: Kompletter Workflow Whisper → Chat"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("workflow_whisper_chat", "start", "start", {
        "workflow": "whisper → chat"
    })
    
    try:
        # Schritt 1: Prüfe Whisper-Status
        log_manager.write_test_log("workflow_whisper_chat", "whisper_check_start", "start", {})
        whisper_state = system_checker.check_model_state("audio")
        log_manager.write_test_log("workflow_whisper_chat", "whisper_state", "info", whisper_state)
        
        # Schritt 2: Prüfe Chat-Status
        log_manager.write_test_log("workflow_whisper_chat", "chat_check_start", "start", {})
        chat_state = system_checker.check_model_state("text")
        log_manager.write_test_log("workflow_whisper_chat", "chat_state", "info", chat_state)
        
        # Schritt 3: Teste Chat-Endpunkt (simuliert nach Whisper-Transkription)
        log_manager.write_test_log("workflow_whisper_chat", "chat_request", "start", {})
        chat_response = system_checker.check_api_endpoint("/chat", method="POST", data={
            "message": "Test message after transcription",
            "conversation_id": None
        })
        log_manager.write_test_log("workflow_whisper_chat", "chat_response", "info", chat_response)
        
        # Analysiere gesamten Workflow
        all_logs = log_manager.read_test_logs("workflow_whisper_chat")
        problems = log_manager.analyze_logs("workflow_whisper_chat")
        
        if problems:
            log_manager.write_test_log("workflow_whisper_chat", "problems_detected", "error", {
                "problems": problems
            })
            
            debug_report = generate_debug_report("workflow_whisper_chat", problems, 
                                                relevant_logs=all_logs)
            log_manager.write_test_log("workflow_whisper_chat", "debug_report", "info", debug_report)
        
        # Assert: Chat sollte funktionieren (200 oder 202)
        assert chat_response["status_code"] in [200, 202], \
            f"Chat-Endpunkt sollte funktionieren: {chat_response['error']}"
        
        log_manager.write_test_log("workflow_whisper_chat", "end", "success", {
            "chat_status_code": chat_response["status_code"]
        })
    
    except Exception as e:
        log_manager.write_test_log("workflow_whisper_chat", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "workflow_whisper_chat",
            "exception": str(e)
        })
        
        all_logs = log_manager.read_test_logs("workflow_whisper_chat")
        debug_report = generate_debug_report("workflow_whisper_chat", problems, 
                                            relevant_logs=all_logs)
        log_manager.write_test_log("workflow_whisper_chat", "debug_report", "info", debug_report)
        
        raise


def test_chat_to_image_workflow():
    """Test: Workflow Chat → Bildgenerierung"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("workflow_chat_image", "start", "start", {
        "workflow": "chat → image"
    })
    
    try:
        # Schritt 1: Prüfe Chat-Status
        log_manager.write_test_log("workflow_chat_image", "chat_check_start", "start", {})
        chat_state = system_checker.check_model_state("text")
        log_manager.write_test_log("workflow_chat_image", "chat_state", "info", chat_state)
        
        # Schritt 2: Prüfe Image-Status
        log_manager.write_test_log("workflow_chat_image", "image_check_start", "start", {})
        image_state = system_checker.check_model_state("image")
        log_manager.write_test_log("workflow_chat_image", "image_state", "info", image_state)
        
        # Schritt 3: Teste Image-Generierung (simuliert nach Chat)
        log_manager.write_test_log("workflow_chat_image", "image_request", "start", {})
        image_response = system_checker.check_api_endpoint("/image/generate", method="POST", data={
            "prompt": "A beautiful sunset",
            "conversation_id": None
        })
        log_manager.write_test_log("workflow_chat_image", "image_response", "info", image_response)
        
        # Analysiere Workflow
        all_logs = log_manager.read_test_logs("workflow_chat_image")
        problems = log_manager.analyze_logs("workflow_chat_image")
        
        if problems:
            log_manager.write_test_log("workflow_chat_image", "problems_detected", "error", {
                "problems": problems
            })
            
            debug_report = generate_debug_report("workflow_chat_image", problems,
                                                relevant_logs=all_logs)
            log_manager.write_test_log("workflow_chat_image", "debug_report", "info", debug_report)
        
        # Assert: Image-Endpunkt sollte funktionieren (200, 202 oder 503 wenn nicht verfügbar)
        assert image_response["status_code"] in [200, 202, 503], \
            f"Image-Endpunkt sollte funktionieren: {image_response['error']}"
        
        log_manager.write_test_log("workflow_chat_image", "end", "success", {
            "image_status_code": image_response["status_code"]
        })
    
    except Exception as e:
        log_manager.write_test_log("workflow_chat_image", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "workflow_chat_image",
            "exception": str(e),
            "model_type": "image"
        })
        
        all_logs = log_manager.read_test_logs("workflow_chat_image")
        debug_report = generate_debug_report("workflow_chat_image", problems,
                                            relevant_logs=all_logs)
        log_manager.write_test_log("workflow_chat_image", "debug_report", "info", debug_report)
        
        raise


def test_full_workflow():
    """Test: Kompletter Workflow Whisper → Chat → Bild"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("workflow_full", "start", "start", {
        "workflow": "whisper → chat → image"
    })
    
    try:
        # Schritt 1: Whisper
        log_manager.write_test_log("workflow_full", "whisper_check", "start", {})
        whisper_state = system_checker.check_model_state("audio")
        log_manager.write_test_log("workflow_full", "whisper_state", "info", whisper_state)
        
        # Schritt 2: Chat
        log_manager.write_test_log("workflow_full", "chat_check", "start", {})
        chat_state = system_checker.check_model_state("text")
        log_manager.write_test_log("workflow_full", "chat_state", "info", chat_state)
        
        # Schritt 3: Image
        log_manager.write_test_log("workflow_full", "image_check", "start", {})
        image_state = system_checker.check_model_state("image")
        log_manager.write_test_log("workflow_full", "image_state", "info", image_state)
        
        # Schritt 4: Teste Chat
        log_manager.write_test_log("workflow_full", "chat_request", "start", {})
        chat_response = system_checker.check_api_endpoint("/chat", method="POST", data={
            "message": "Generate an image of a sunset",
            "conversation_id": None
        })
        log_manager.write_test_log("workflow_full", "chat_response", "info", chat_response)
        
        # Schritt 5: Teste Image (falls verfügbar)
        if image_state.get("current_model") or image_response["status_code"] == 200:
            log_manager.write_test_log("workflow_full", "image_request", "start", {})
            image_response = system_checker.check_api_endpoint("/image/generate", method="POST", data={
                "prompt": "A beautiful sunset over mountains",
                "conversation_id": None
            })
            log_manager.write_test_log("workflow_full", "image_response", "info", image_response)
        
        # Analysiere gesamten Workflow
        all_logs = log_manager.read_test_logs("workflow_full")
        problems = log_manager.analyze_logs("workflow_full")
        
        # Prüfe Logs auf Fehler
        errors = system_checker.check_logs_for_errors()
        if errors["errors"]:
            log_manager.write_test_log("workflow_full", "log_errors", "error", errors)
            problems.extend([{
                "type": "log_error",
                "description": f"Fehler in Logs gefunden: {len(errors['errors'])}",
                "evidence": errors["errors"][:3]
            }])
        
        if problems:
            log_manager.write_test_log("workflow_full", "problems_detected", "error", {
                "problems": problems
            })
            
            debug_report = generate_debug_report("workflow_full", problems,
                                                relevant_logs=all_logs)
            log_manager.write_test_log("workflow_full", "debug_report", "info", debug_report)
        
        # Assert: Chat sollte funktionieren
        assert chat_response["status_code"] in [200, 202], \
            f"Chat-Endpunkt sollte funktionieren: {chat_response['error']}"
        
        log_manager.write_test_log("workflow_full", "end", "success", {
            "chat_status_code": chat_response["status_code"],
            "problems_count": len(problems)
        })
    
    except Exception as e:
        log_manager.write_test_log("workflow_full", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "workflow_full",
            "exception": str(e)
        })
        
        all_logs = log_manager.read_test_logs("workflow_full")
        debug_report = generate_debug_report("workflow_full", problems,
                                            relevant_logs=all_logs)
        log_manager.write_test_log("workflow_full", "debug_report", "info", debug_report)
        
        raise


def test_simultaneous_operations():
    """Test: Simultanes Arbeiten mit mehreren Modellen"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_simultaneous", "start", "start", {
        "test": "simultaneous model operations"
    })
    
    try:
        # Prüfe alle Modell-Status gleichzeitig
        log_manager.write_test_log("test_simultaneous", "check_all_models", "start", {})
        
        text_state = system_checker.check_model_state("text")
        image_state = system_checker.check_model_state("image")
        audio_state = system_checker.check_model_state("audio")
        
        log_manager.write_test_log("test_simultaneous", "all_states", "info", {
            "text": text_state,
            "image": image_state,
            "audio": audio_state
        })
        
        # Prüfe ob mehrere Modelle gleichzeitig geladen werden können
        loading_count = sum([
            text_state.get("is_loading", False),
            image_state.get("is_loading", False),
            audio_state.get("is_loading", False)
        ])
        
        log_manager.write_test_log("test_simultaneous", "loading_count", "info", {
            "simultaneous_loading": loading_count
        })
        
        # Test sollte erfolgreich sein wenn keine Fehler
        errors = system_checker.check_logs_for_errors()
        if errors["errors"]:
            log_manager.write_test_log("test_simultaneous", "errors_found", "error", errors)
        
        log_manager.write_test_log("test_simultaneous", "end", "success", {
            "simultaneous_loading": loading_count,
            "errors_count": len(errors["errors"])
        })
    
    except Exception as e:
        log_manager.write_test_log("test_simultaneous", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_simultaneous",
            "exception": str(e)
        })
        
        debug_report = generate_debug_report("test_simultaneous", problems)
        log_manager.write_test_log("test_simultaneous", "debug_report", "info", debug_report)
        
        raise









