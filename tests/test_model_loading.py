"""
Unit Tests für Modell-Laden mit automatischem Debugging
"""
import sys
import os
from pathlib import Path

# Füge backend zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))

from log_manager import LogManager
from system_checker import SystemChecker
from test_runner import generate_debug_report

# Importiere Manager (werden in Tests initialisiert)
model_manager = None
image_manager = None
whisper_manager = None


def setUp():
    """Setup für Tests"""
    global model_manager, image_manager, whisper_manager
    
    try:
        from model_manager import ModelManager
        from whisper_manager import WhisperManager
        
        config_path = workspace_root / "config.json"
        model_manager = ModelManager(config_path=str(config_path))
        whisper_manager = WhisperManager(config_path=str(config_path))
        
        # ImageManager optional (kann fehlen wenn diffusers nicht installiert)
        try:
            from image_manager import ImageManager
            image_manager = ImageManager(config_path=str(config_path))
        except ImportError:
            image_manager = None
    
    except Exception as e:
        print(f"Fehler beim Setup: {e}")


def tearDown():
    """Cleanup nach Tests"""
    global model_manager, image_manager, whisper_manager
    
    # Entlade Modelle
    if model_manager and model_manager.is_model_loaded():
        try:
            model_manager.model = None
            model_manager.tokenizer = None
            model_manager.current_model_id = None
        except:
            pass
    
    if image_manager and image_manager.is_model_loaded():
        try:
            image_manager.pipeline = None
            image_manager.current_model_id = None
        except:
            pass
    
    if whisper_manager and whisper_manager.is_model_loaded():
        try:
            whisper_manager.model = None
            whisper_manager.processor = None
            whisper_manager.current_model_id = None
        except:
            pass


def test_model_manager_load_success():
    """Test: ModelManager lädt Modell erfolgreich"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_model_manager_load", "start", "start", {
        "expected_behavior": "ModelManager should load a model successfully"
    })
    
    try:
        # Prüfe verfügbare Modelle
        available_models = model_manager.get_available_models()
        log_manager.write_test_log("test_model_manager_load", "check_available", "info", {
            "available_models": list(available_models.keys())
        })
        
        if not available_models:
            log_manager.write_test_log("test_model_manager_load", "end", "skipped", {
                "reason": "No models available in config"
            })
            return  # Skip test if no models
        
        # Wähle erstes Text-Modell
        text_models = {k: v for k, v in available_models.items() if v.get("type") != "image" and v.get("type") != "audio"}
        if not text_models:
            log_manager.write_test_log("test_model_manager_load", "end", "skipped", {
                "reason": "No text models available"
            })
            return
        
        test_model_id = list(text_models.keys())[0]
        log_manager.write_test_log("test_model_manager_load", "load_model", "info", {
            "model_id": test_model_id
        })
        
        # Lade Modell
        result = model_manager.load_model(test_model_id)
        log_manager.write_test_log("test_model_manager_load", "load_result", "info", {
            "result": result
        })
        
        # Prüfe System-State
        state = {
            "is_loaded": model_manager.is_model_loaded(),
            "current_model": model_manager.get_current_model(),
            "expected_model": test_model_id
        }
        log_manager.write_test_log("test_model_manager_load", "state_check", "info", {
            **state,
            "expected_loaded": True
        })
        
        # Prüfe Logs auf Fehler
        errors = system_checker.check_logs_for_errors()
        if errors["errors"]:
            log_manager.write_test_log("test_model_manager_load", "error_detected", "error", errors)
        
        # Assert
        assert result == True, f"load_model() sollte True zurückgeben, bekam {result}"
        assert state["is_loaded"] == True, f"Modell sollte geladen sein, is_model_loaded()={state['is_loaded']}"
        assert state["current_model"] == test_model_id, f"current_model sollte {test_model_id} sein, ist {state['current_model']}"
        
        log_manager.write_test_log("test_model_manager_load", "end", "success", {
            "result": result,
            "model_id": test_model_id
        })
    
    except Exception as e:
        log_manager.write_test_log("test_model_manager_load", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_model_manager_load",
            "exception": str(e),
            "model_type": "text"
        })
        
        debug_report = generate_debug_report("test_model_manager_load", problems)
        log_manager.write_test_log("test_model_manager_load", "debug_report", "info", debug_report)
        
        raise


def test_model_manager_is_model_loaded():
    """Test: is_model_loaded() funktioniert korrekt"""
    log_manager = LogManager()
    
    log_manager.write_test_log("test_is_model_loaded", "start", "start", {})
    
    try:
        # Prüfe initialen Status
        initial_state = model_manager.is_model_loaded()
        log_manager.write_test_log("test_is_model_loaded", "initial_check", "info", {
            "is_loaded": initial_state
        })
        
        # Entlade falls geladen
        if initial_state:
            model_manager.model = None
            model_manager.tokenizer = None
            model_manager.current_model_id = None
        
        # Prüfe nach Entladen
        after_unload = model_manager.is_model_loaded()
        assert after_unload == False, f"Nach Entladen sollte is_model_loaded() False sein, ist {after_unload}"
        
        log_manager.write_test_log("test_is_model_loaded", "end", "success", {})
    
    except Exception as e:
        log_manager.write_test_log("test_is_model_loaded", "exception", "error", {
            "error": str(e)
        })
        raise


def test_image_manager_load_if_available():
    """Test: ImageManager lädt Modell (falls verfügbar)"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    if image_manager is None:
        log_manager.write_test_log("test_image_manager_load", "end", "skipped", {
            "reason": "ImageManager not available (diffusers not installed)"
        })
        return
    
    log_manager.write_test_log("test_image_manager_load", "start", "start", {})
    
    try:
        available_models = image_manager.get_available_models()
        image_models = {k: v for k, v in available_models.items() if v.get("type") == "image"}
        
        if not image_models:
            log_manager.write_test_log("test_image_manager_load", "end", "skipped", {
                "reason": "No image models available"
            })
            return
        
        test_model_id = list(image_models.keys())[0]
        log_manager.write_test_log("test_image_manager_load", "load_model", "info", {
            "model_id": test_model_id
        })
        
        result = image_manager.load_model(test_model_id)
        log_manager.write_test_log("test_image_manager_load", "load_result", "info", {
            "result": result
        })
        
        state = {
            "is_loaded": image_manager.is_model_loaded(),
            "current_model": image_manager.get_current_model()
        }
        log_manager.write_test_log("test_image_manager_load", "state_check", "info", {
            **state,
            "expected_loaded": True
        })
        
        if result:
            assert state["is_loaded"] == True, "Modell sollte geladen sein"
            assert state["current_model"] == test_model_id, f"current_model sollte {test_model_id} sein"
        
        log_manager.write_test_log("test_image_manager_load", "end", "success", {
            "result": result
        })
    
    except Exception as e:
        log_manager.write_test_log("test_image_manager_load", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_image_manager_load",
            "exception": str(e),
            "model_type": "image"
        })
        
        debug_report = generate_debug_report("test_image_manager_load", problems)
        log_manager.write_test_log("test_image_manager_load", "debug_report", "info", debug_report)
        
        raise


def test_whisper_manager_load():
    """Test: WhisperManager lädt Modell"""
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    log_manager.write_test_log("test_whisper_manager_load", "start", "start", {})
    
    try:
        available_models = whisper_manager.get_available_models()
        log_manager.write_test_log("test_whisper_manager_load", "check_available", "info", {
            "available_models": list(available_models.keys())
        })
        
        if not available_models:
            log_manager.write_test_log("test_whisper_manager_load", "end", "skipped", {
                "reason": "No audio models available"
            })
            return
        
        test_model_id = list(available_models.keys())[0]
        log_manager.write_test_log("test_whisper_manager_load", "load_model", "info", {
            "model_id": test_model_id
        })
        
        result = whisper_manager.load_model(test_model_id)
        log_manager.write_test_log("test_whisper_manager_load", "load_result", "info", {
            "result": result
        })
        
        state = {
            "is_loaded": whisper_manager.is_model_loaded(),
            "current_model": whisper_manager.get_current_model()
        }
        log_manager.write_test_log("test_whisper_manager_load", "state_check", "info", {
            **state,
            "expected_loaded": True
        })
        
        assert result == True, f"load_model() sollte True zurückgeben"
        assert state["is_loaded"] == True, "Modell sollte geladen sein"
        assert state["current_model"] == test_model_id, f"current_model sollte {test_model_id} sein"
        
        log_manager.write_test_log("test_whisper_manager_load", "end", "success", {
            "result": result
        })
    
    except Exception as e:
        log_manager.write_test_log("test_whisper_manager_load", "exception", "error", {
            "error": str(e)
        })
        
        problems = system_checker.identify_problems({
            "test": "test_whisper_manager_load",
            "exception": str(e),
            "model_type": "audio"
        })
        
        debug_report = generate_debug_report("test_whisper_manager_load", problems)
        log_manager.write_test_log("test_whisper_manager_load", "debug_report", "info", debug_report)
        
        raise








