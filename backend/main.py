"""
FastAPI Server - Hauptserver für den lokalen AI-Dienst
"""
import json
import sys
import os
import logging
import subprocess
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import csv
import io
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager
from conversation_manager import ConversationManager
from preference_learner import PreferenceLearner
from quality_manager import QualityManager

# Logger muss vor dem try-except verfügbar sein
from logging_utils import get_logger
_temp_logger = get_logger(__name__, log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "main_server.log"))

# ImageManager optional importieren (kann ohne diffusers fehlschlagen)
try:
    from image_manager import ImageManager
except Exception as e:
    _temp_logger.warning(f"ImageManager konnte nicht geladen werden: {e}")
    _temp_logger.warning("Bildgenerierung wird nicht verfügbar sein")
    ImageManager = None

from whisper_manager import WhisperManager
from agent_manager import AgentManager
from agent_tools import (
    initialize_tools, read_file, write_file, execute_code, generate_image, 
    describe_image, call_agent, web_search, list_directory, delete_file, file_exists
)
from model_service_client import ModelServiceClient
import psutil
import torch
import numpy as np
from scipy.io import wavfile
from PIL import Image

logger = get_logger(__name__, log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "main_server.log"))

app = FastAPI(title="Local AI Service")

# CORS für Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manager initialisieren (mit korrekten Pfaden)
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
workspace_root = os.path.dirname(os.path.dirname(__file__))

# Lade Config für Model-Service
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    model_service_config = config.get("model_service", {})
    model_service_host = model_service_config.get("host", "127.0.0.1")
    model_service_port = model_service_config.get("port", 8001)
except:
    model_service_host = "127.0.0.1"
    model_service_port = 8001
    config = {}

# Helper-Funktion für Temp-Verzeichnis
def get_temp_directory() -> str:
    """
    Gibt das konfigurierte Temp-Verzeichnis zurück.
    Erstellt das Verzeichnis falls es nicht existiert.
    Fallback auf System-Temp bei Fehlern.
    
    Returns:
        Pfad zum Temp-Verzeichnis
    """
    default_temp_dir = "G:\\KI Modelle\\KI-Temp"
    
    try:
        # Lade aus Config
        temp_dir = config.get("temp_directory", default_temp_dir)
        
        # Erstelle Verzeichnis falls nicht vorhanden
        if temp_dir and not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Temp-Verzeichnis erstellt: {temp_dir}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Konnte Temp-Verzeichnis nicht erstellen: {temp_dir} - {e}. Verwende System-Temp.")
                import tempfile
                return tempfile.gettempdir()
        
        return temp_dir
    except Exception as e:
        logger.warning(f"Fehler beim Laden des Temp-Verzeichnisses aus Config: {e}. Verwende System-Temp.")
        import tempfile
        return tempfile.gettempdir()

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Konvertiert Text zu einem gültigen Dateinamen.
    Entfernt Sonderzeichen und begrenzt die Länge.
    
    Args:
        text: Der zu konvertierende Text
        max_length: Maximale Länge des Dateinamens (ohne Extension)
        
    Returns:
        Sanitized Dateiname
    """
    import re
    # Entferne Sonderzeichen, behalte nur Buchstaben, Zahlen, Leerzeichen, Bindestriche, Unterstriche
    sanitized = re.sub(r'[^\w\s-]', '', text)
    # Ersetze Leerzeichen durch Unterstriche
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Entferne mehrfache Unterstriche
    sanitized = re.sub(r'_+', '_', sanitized)
    # Entferne führende/abschließende Unterstriche
    sanitized = sanitized.strip('_')
    # Begrenze Länge
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    return sanitized if sanitized else "image"

def save_generated_image(image: Image.Image, prompt: str, temp_dir: Optional[str] = None) -> Optional[str]:
    """
    Speichert ein generiertes Bild mit Timestamp und Prompt im Dateinamen.
    Verwendet Output Manager für konfigurierbare Pfade und Dateinamen.
    
    Args:
        image: Das PIL Image
        prompt: Der Prompt-Text für den Dateinamen
        temp_dir: Optionales Temp-Verzeichnis (deprecated, wird ignoriert)
        
    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        # Verwende Output Manager für organisierte Speicherung
        from output_manager import get_output_manager
        output_mgr = get_output_manager()
        
        # Generiere Output-Pfad mit automatischem Titel und Datum
        filepath = output_mgr.get_image_output_path(prompt=prompt, extension="png")
        
        # Speichere Bild
        image.save(str(filepath), format="PNG", optimize=True)
        logger.info(f"Bild gespeichert: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.warning(f"Fehler beim Speichern des Bildes (Output Manager): {e}")
        # Fallback zum alten Verhalten
        try:
            if temp_dir is None:
                temp_dir = get_temp_directory()
            images_dir = os.path.join(temp_dir, "generated_images")
            os.makedirs(images_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_prompt = sanitize_filename(prompt, max_length=50)
            filename = f"{timestamp}_{sanitized_prompt}.png"
            file_path = os.path.join(images_dir, filename)
            image.save(file_path, format="PNG")
            logger.info(f"Bild gespeichert (Fallback): {file_path}")
            return file_path
        except Exception as fallback_error:
            logger.warning(f"Auch Fallback fehlgeschlagen: {fallback_error}")
            return None

# Model-Service-Client initialisieren
model_service_client = ModelServiceClient(host=model_service_host, port=model_service_port)

# Lokale Manager (für Fallback oder direkte Nutzung)
model_manager = ModelManager(config_path=config_path)
conversation_manager = ConversationManager()
preference_learner = PreferenceLearner()

# Initialisiere Quality Manager mit Web-Search (global) - für ALLE Chat-Modelle
quality_manager = QualityManager(web_search_function=web_search)

# ImageManager initialisieren (mit Fehlerbehandlung)

image_manager = None
if ImageManager:
    try:
        
        image_manager = ImageManager(config_path=config_path)
        
        logger.info("ImageManager erfolgreich initialisiert")
    except Exception as e:
        
        logger.error(f"Fehler bei ImageManager-Initialisierung: {e}")
        import traceback
        logger.error(traceback.format_exc())
        image_manager = None
else:
    
    logger.warning("ImageManager-Klasse nicht verfügbar (Import fehlgeschlagen)")


whisper_manager = WhisperManager(config_path=config_path)
agent_manager = AgentManager()

# Prüfe ob Model-Service verfügbar ist
USE_MODEL_SERVICE = model_service_client.is_available()
if USE_MODEL_SERVICE:
    logger.info("Model-Service ist verfügbar - nutze Model-Service für Modell-Operationen")
else:
    logger.warning("Model-Service ist nicht verfügbar - nutze lokale Manager (Fallback)")

def check_model_service_available() -> bool:
    """Prüft dynamisch ob Model-Service verfügbar ist (kann sich zur Laufzeit ändern)"""
    return model_service_client.is_available()

# Thread Pool für Modell-Laden (damit UI nicht blockiert)
model_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_loader")
image_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image_loader")
audio_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio_loader")
# Thread Pool für Web-Search (damit Server nicht blockiert)
web_search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="web_search")
# Thread Pool für Modell-Generierung (damit Server nicht blockiert)
model_generation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_generation")

# Lade-Status für alle Manager
loading_status = {
    "text_model": {"loading": False, "model_id": None, "error": None, "conversation_id": None, "start_time": None},
    "image_model": {"loading": False, "model_id": None, "error": None, "conversation_id": None, "start_time": None},
    "audio_model": {"loading": False, "model_id": None, "error": None, "conversation_id": None, "start_time": None}
}

# Tools initialisieren
if image_manager:
    initialize_tools(model_manager, image_manager, agent_manager, workspace_root)
else:
    logger.warning("ImageManager nicht verfügbar - Tools werden ohne Bildgenerierung initialisiert")
    # Initialisiere Tools ohne image_manager (wird in agent_tools.py gehandhabt)
    initialize_tools(model_manager, None, agent_manager, workspace_root)

# Tools registrieren
agent_manager.register_tool("read_file", read_file)
agent_manager.register_tool("write_file", write_file)
agent_manager.register_tool("execute_code", execute_code)
agent_manager.register_tool("generate_image", generate_image)
agent_manager.register_tool("describe_image", describe_image)
agent_manager.register_tool("call_agent", call_agent)
agent_manager.register_tool("web_search", web_search)
agent_manager.register_tool("list_directory", list_directory)
agent_manager.register_tool("delete_file", delete_file)
agent_manager.register_tool("file_exists", file_exists)

# Agent-Typen registrieren
from agents import PromptAgent, ImageAgent, VisionAgent, ChatAgent
from pipeline_manager import PipelineManager

agent_manager.register_agent_type("prompt_agent", "Prompt Agent", 
                                  "Erstellt detaillierte Bildbeschreibungen/Prompts aus Text", 
                                  "text", PromptAgent)
agent_manager.register_agent_type("image_agent", "Image Agent", 
                                  "Generiert Bilder basierend auf Prompts", 
                                  "image", ImageAgent)
agent_manager.register_agent_type("vision_agent", "Vision Agent", 
                                  "Beschreibt generierte Bilder", 
                                  "text", VisionAgent)
agent_manager.register_agent_type("chat_agent", "Chat Agent", 
                                  "Normaler Chat mit automatischer Tool-Unterstützung (WebSearch, Dateimanipulation)", 
                                  "text", ChatAgent)

# Pipeline Manager
pipeline_manager = PipelineManager(agent_manager)

# Helper-Funktion um Agent-Manager zu setzen
def set_agent_managers(agent_instance):
    """Setzt die Manager für einen Agent"""
    agent_instance.set_model_manager(model_manager)
    agent_instance.set_agent_manager(agent_manager)
    agent_instance.set_model_service_client(model_service_client)  # Model Service Client hinzufügen
    if image_manager:
        agent_instance.set_image_manager(image_manager)

# Helper-Funktion zum sicheren Löschen von Dateien (mit Retry)
def _safe_delete_file(file_path: str, max_retries: int = 3, delay: float = 0.1):
    """
    Löscht eine Datei sicher mit Retry-Mechanismus.
    Wichtig für Windows, wo Dateien manchmal noch geöffnet sind.
    
    Args:
        file_path: Pfad zur zu löschenden Datei
        max_retries: Maximale Anzahl von Versuchen
        delay: Wartezeit zwischen Versuchen in Sekunden
    """
    if not file_path or not os.path.exists(file_path):
        return
    
    import time
    for attempt in range(max_retries):
        try:
            os.unlink(file_path)
            return  # Erfolgreich gelöscht
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponentielles Backoff
            else:
                # Letzter Versuch fehlgeschlagen - logge Warnung aber wirf keinen Fehler
                logger.warning(f"Konnte temporäre Datei nicht löschen nach {max_retries} Versuchen: {file_path} - {e}")
        except Exception as e:
            # Unerwarteter Fehler - logge aber wirf keinen Fehler
            logger.warning(f"Unerwarteter Fehler beim Löschen von {file_path}: {e}")
            return

# Setze Manager-Funktion für Pipeline Manager
pipeline_manager.set_managers_func(set_agent_managers)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_length: int = 2048
    temperature: float = 0.3  # Default: 0.3 für bessere Qualität (konsistent mit Frontend)
    language: Optional[str] = None  # Sprache für Antwort (z.B. "de", "en") - wenn None, wird aus speech_input_app/config.json gelesen


class SetConversationModelRequest(BaseModel):
    model_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class LoadModelRequest(BaseModel):
    model_id: str


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    aspect_ratio: Optional[str] = None  # z.B. "16:9", "custom:2.5:1"
    model_id: Optional[str] = None
    conversation_id: Optional[str] = None


class GenerateImageResponse(BaseModel):
    image_base64: str
    model_id: str


class PerformanceSettingsRequest(BaseModel):
    cpu_threads: Optional[int] = None
    gpu_optimization: str = "balanced"  # balanced, speed, memory
    disable_cpu_offload: bool = False
    use_torch_compile: Optional[bool] = None
    use_quantization: Optional[bool] = None
    quantization_bits: Optional[int] = None
    use_flash_attention: Optional[bool] = None
    enable_tf32: Optional[bool] = None
    enable_cudnn_benchmark: Optional[bool] = None


class PerformanceSettingsResponse(BaseModel):
    cpu_threads: Optional[int]
    gpu_optimization: str
    disable_cpu_offload: bool
    use_torch_compile: bool
    use_quantization: bool
    quantization_bits: int
    use_flash_attention: bool
    enable_tf32: bool
    enable_cudnn_benchmark: bool


class AudioSettingsRequest(BaseModel):
    transcription_language: Optional[str] = ""  # Leerer String = Auto-Erkennung


class AudioSettingsResponse(BaseModel):
    transcription_language: Optional[str]


class GenerateImageResponse(BaseModel):
    image_base64: str
    model_id: str


# API Endpoints

@app.get("/")
async def root():
    return {"message": "Local AI Service API", "status": "running"}


@app.get("/status")
async def get_status():
    """Gibt den aktuellen Status zurück"""
    return {
        "model_loaded": model_manager.is_model_loaded(),
        "current_model": model_manager.get_current_model(),
        "preference_learning_enabled": preference_learner.is_enabled()
    }


@app.get("/health")
async def health_check():
    """Health-Check für alle geladenen Modelle"""
    import time
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "models": {}
    }
    
    # Text-Modell
    if model_manager.is_model_loaded():
        text_health = model_manager.health_check()
        health_status["models"]["text"] = {
            "loaded": True,
            "healthy": text_health["healthy"],
            "last_check": text_health["last_check"],
            "response_time_ms": text_health["response_time_ms"],
            "error": text_health.get("error")
        }
        if not text_health["healthy"]:
            health_status["status"] = "degraded"
    else:
        health_status["models"]["text"] = {
            "loaded": False,
            "healthy": False
        }
    
    # Audio-Modell (Whisper - vorerst nur Basis-Status, kein Health-Check)
    if whisper_manager.is_model_loaded():
        health_status["models"]["audio"] = {
            "loaded": True,
            "healthy": True,  # Whisper funktioniert zuverlässig, daher immer healthy
            "note": "Whisper Health-Check vorerst ausgelassen - funktioniert bereits sehr gut"
        }
    else:
        health_status["models"]["audio"] = {
            "loaded": False,
            "healthy": False
        }
    
    return health_status


@app.get("/system/stats")
async def get_system_stats():
    """Gibt Systemressourcen zurück: CPU, RAM, GPU"""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_memory_percent": None,
        "gpu_utilization": None
    }
    
    # GPU-Informationen
    if torch.cuda.is_available():
        try:
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            
            # Nutze nvidia-smi für EXAKTE GPU-Speicher-Messung
            # (torch.cuda.memory_allocated zeigt nur PyTorch-Allokationen, nicht tatsächlichen VRAM)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(',')
                    stats["gpu_memory_used_mb"] = float(values[0].strip())
                    stats["gpu_memory_total_mb"] = float(values[1].strip())
                    stats["gpu_memory_percent"] = round((stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100, 1) if stats["gpu_memory_total_mb"] > 0 else 0
                    stats["gpu_utilization"] = int(values[2].strip())
                else:
                    # Fallback auf PyTorch (ungenau)
                    stats["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 0)
                    stats["gpu_memory_used_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 0)  # memory_reserved statt memory_allocated
                    stats["gpu_memory_percent"] = round((stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100, 1) if stats["gpu_memory_total_mb"] > 0 else 0
            except Exception as smi_error:
                # Fallback auf PyTorch (ungenau)
                logger.debug(f"nvidia-smi nicht verfügbar, verwende PyTorch-Messung: {smi_error}")
                stats["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 0)
                stats["gpu_memory_used_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 0)  # memory_reserved statt memory_allocated
                stats["gpu_memory_percent"] = round((stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100, 1) if stats["gpu_memory_total_mb"] > 0 else 0
        except Exception as e:
            logger.warning(f"Fehler beim Abrufen der GPU-Informationen: {e}")
    
    return stats


@app.get("/models")
async def get_models():
    """Gibt alle verfügbaren Modelle zurück"""
    return {
        "models": model_manager.get_available_models(),
        "current_model": model_manager.get_current_model()
    }


def _load_text_model_async(model_id: str, conversation_id: Optional[str] = None):
    """Lädt ein Text-Modell im Hintergrund"""
    try:
        loading_status["text_model"]["loading"] = True
        loading_status["text_model"]["model_id"] = model_id
        loading_status["text_model"]["error"] = None
        loading_status["text_model"]["conversation_id"] = conversation_id
        
        success = model_manager.load_model(model_id)
        if not success:
            loading_status["text_model"]["error"] = f"Fehler beim Laden des Modells: {model_id}"
        else:
            loading_status["text_model"]["error"] = None
    except Exception as e:
        loading_status["text_model"]["error"] = str(e)
        logger.error(f"Fehler beim asynchronen Laden des Modells: {e}")
    finally:
        loading_status["text_model"]["loading"] = False
        loading_status["text_model"]["conversation_id"] = None

@app.post("/models/load")
async def load_model(request: LoadModelRequest):
    """Lädt ein Modell im Hintergrund (blockiert UI nicht)"""
    # Prüfe ob bereits ein Modell geladen wird
    if loading_status["text_model"]["loading"]:
        raise HTTPException(status_code=400, detail="Ein Modell wird bereits geladen. Bitte warten Sie.")
    
    # Starte Ladevorgang im Hintergrund
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(model_load_executor, _load_text_model_async, request.model_id, None)
    
    return {
        "message": f"Modell wird geladen: {request.model_id}",
        "model_id": request.model_id,
        "status": "loading"
    }

@app.get("/models/load/status")
async def get_model_load_status():
    """Gibt den Status des Modell-Ladevorgangs zurück"""
    
    return loading_status["text_model"]


async def ensure_image_model_loaded(model_id: str, conversation_id: Optional[str] = None) -> bool:
    """
    Stellt sicher dass ein Image-Modell geladen ist.
    Lädt es asynchron falls nötig.
    
    Returns:
        True wenn Modell bereits geladen oder sofort geladen werden konnte
        False wenn Modell im Hintergrund geladen wird
    """
    if not image_manager:
        raise HTTPException(status_code=503, detail="Bildgenerierung nicht verfügbar")
    
    # Prüfe dynamisch ob Model-Service verfügbar ist (kann sich zur Laufzeit ändern)
    use_model_service = check_model_service_available()
    
    
    
    # Prüfe ob Modell bereits geladen ist (abhängig von Model-Service-Verfügbarkeit)
    if use_model_service:
        # Prüfe Model-Service Status
        status = model_service_client.get_image_model_status()
        
        if status and status.get("loaded") and status.get("model_id") == model_id:
            
            return True  # Bereits im Model Service geladen
    else:
        # Prüfe lokalen Manager
        current_model = image_manager.get_current_model()
        if current_model == model_id and image_manager.is_model_loaded():
            
            return True  # Bereits geladen
    
    # Prüfe ob bereits ein Modell geladen wird
    if loading_status["image_model"]["loading"]:
        # Prüfe ob es das gleiche Modell ist
        if loading_status["image_model"]["model_id"] == model_id:
            
            return False  # Wird bereits geladen
        else:
            # Anderes Modell wird geladen - prüfe ob es hängt (länger als 10 Minuten)
            import time
            # Wenn loading länger als 10 Minuten dauert, könnte es hängen
            # Aber wir starten trotzdem kein neues Laden, um Konflikte zu vermeiden
            logger.warning(f"Ein anderes Modell ({loading_status['image_model']['model_id']}) wird bereits geladen. Warte auf Abschluss.")
            return False  # Warte auf aktuelles Laden
    
    
    
    # Prüfe dynamisch ob Model-Service verfügbar ist
    use_model_service = check_model_service_available()
    
    # Starte asynchrones Laden
    if use_model_service:
        # Nutze Model Service - lade direkt (nicht asynchron, da Model Service bereits asynchron lädt)
        logger.info(f"Lade Bildmodell über Model Service: {model_id}")
        success = model_service_client.load_image_model(model_id)
        if success:
            # Prüfe Status nach kurzer Wartezeit
            import time
            time.sleep(2)
            status = model_service_client.get_image_model_status()
            if status and status.get("loaded") and status.get("model_id") == model_id:
                return True  # Bereits geladen
            else:
                return False  # Wird noch geladen
        else:
            return False  # Fehler beim Laden
    else:
        # Fallback: Nutze lokalen Manager asynchron
        loop = asyncio.get_event_loop()
        loop.run_in_executor(image_load_executor, _load_image_model_async, model_id, conversation_id)
        return False  # Wird geladen

async def ensure_text_model_loaded(model_id: str, conversation_id: Optional[str] = None) -> bool:
    """
    Stellt sicher dass ein Text-Modell geladen ist.
    Lädt es asynchron falls nötig.
    
    Returns:
        True wenn Modell bereits geladen oder sofort geladen werden konnte
        False wenn Modell im Hintergrund geladen wird
    """
    
    
    # Prüfe ob Modell bereits geladen ist (abhängig von USE_MODEL_SERVICE)
    if USE_MODEL_SERVICE:
        # Prüfe Model-Service Status
        status = model_service_client.get_text_model_status()
        if status and status.get("loaded") and status.get("model_id") == model_id:
            return True  # Bereits im Model Service geladen
    else:
        # Prüfe lokalen Manager
        if model_manager.get_current_model() == model_id:
            return True  # Bereits geladen
    
    # Prüfe ob bereits ein Modell geladen wird
    if loading_status["text_model"]["loading"]:
        # Prüfe ob es das gleiche Modell ist
        if loading_status["text_model"]["model_id"] == model_id:
            return False  # Wird bereits geladen
        else:
            # Anderes Modell wird geladen - warte nicht, starte neues Laden
            pass
    
    
    # Starte asynchrones Laden
    loop = asyncio.get_event_loop()
    loop.run_in_executor(model_load_executor, _load_text_model_async, model_id, conversation_id)
    
    return False  # Wird geladen

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat-Endpunkt mit Streaming - gibt Antworten schrittweise zurück
    """
    
    # Conversation ID prüfen/erstellen
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = conversation_manager.create_conversation(conversation_type="chat")
    
    # Prüfe ob Conversation ein Modell zugewiesen hat
    conversation_model_id = conversation_manager.get_conversation_model(conversation_id)
    
    # Bestimme welches Modell verwendet werden soll
    model_to_use = conversation_model_id
    
    if not model_to_use:
        
        if USE_MODEL_SERVICE:
            status = model_service_client.get_text_model_status()
            
            if status and status.get("loaded"):
                model_to_use = status.get("model_id")
            else:
                default_model = model_manager.config.get("default_model")
                if default_model:
                    model_to_use = default_model
                else:
                    raise HTTPException(status_code=400, detail="Kein Modell geladen")
        else:
            
            if model_manager.is_model_loaded():
                model_to_use = model_manager.get_current_model()
            else:
                default_model = model_manager.config.get("default_model")
                if default_model:
                    model_to_use = default_model
                else:
                    raise HTTPException(status_code=400, detail="Kein Modell geladen")
    
    # Lade Conversation History
    history = conversation_manager.get_conversation_history(conversation_id)
    
    # Bestimme Antwort-Sprache (flexibel basierend auf Audiosprache)
    stream_response_language = request.language
    if not stream_response_language:
        # Lese Sprache aus speech_input_app/config.json
        try:
            speech_config_path = os.path.join(workspace_root, "speech_input_app", "config.json")
            if os.path.exists(speech_config_path):
                with open(speech_config_path, 'r', encoding='utf-8') as f:
                    speech_config = json.load(f)
                    stream_response_language = speech_config.get("language", "de")
            else:
                stream_response_language = "de"  # Default: Deutsch
        except Exception as e:
            logger.warning(f"Fehler beim Lesen der Sprach-Konfiguration: {e}, verwende Deutsch")
            stream_response_language = "de"
    
    # Erstelle Messages-Liste
    messages = []
    current_model = model_manager.get_current_model() if not USE_MODEL_SERVICE else model_to_use
    
    # Generiere System-Prompt basierend auf Sprache
    if current_model and "phi-3" in current_model.lower():
        system_prompt = "Du bist ein hilfreicher AI-Assistent." if stream_response_language != "en" else "You are a helpful AI assistant."
    elif current_model and "mistral" in current_model.lower():
        if stream_response_language == "en":
            system_prompt = "You are a helpful AI assistant. Answer briefly, precisely and directly in English. Keep answers under 200 words."
        else:
            system_prompt = "Du bist ein hilfreicher AI-Assistent. Antworte kurz, präzise und direkt auf Deutsch. Halte Antworten unter 200 Wörtern."
    else:
        if stream_response_language == "en":
            system_prompt = "You are a helpful, precise and friendly AI assistant. Answer clearly and directly in English."
        else:
            system_prompt = "Du bist ein hilfreicher, präziser und freundlicher AI-Assistent. Antworte klar und direkt auf Deutsch."
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Filtere History: Nur user/assistant Rollen, keine agent_* Rollen
    # Entferne auch die aktuelle User-Nachricht falls sie bereits in der History ist
    filtered_history = []
    for msg in history[-10:]:  # Letzte 10 Nachrichten
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Überspringe die aktuelle User-Nachricht falls sie bereits in der History ist
        if role == "user" and content == request.message:
            continue
        
        # Konvertiere agent_* Rollen zu user
        if role.startswith("agent_"):
            role = "user"
        # Nur user/assistant Rollen erlauben
        if role in ["user", "assistant"]:
            filtered_history.append({"role": role, "content": content})
    
    # Stelle sicher, dass Rollen alternieren (keine zwei aufeinanderfolgenden gleichen Rollen)
    cleaned_history = []
    last_role = None
    for msg in filtered_history:
        current_role = msg.get("role")
        # Überspringe wenn die Rolle gleich der vorherigen ist
        if current_role == last_role:
            continue
        cleaned_history.append(msg)
        last_role = current_role
    
    # Stelle sicher, dass nach System-Prompt die erste Nachricht "user" ist
    # Wenn die History mit "assistant" beginnt, entferne sie
    if cleaned_history and cleaned_history[0].get("role") == "assistant":
        cleaned_history = cleaned_history[1:]  # Entferne erste "assistant" Nachricht
    
    # Füge gefilterte History hinzu
    messages.extend(cleaned_history)
    
    # Aktuelle Nachricht hinzufügen - nur wenn die letzte Nachricht nicht bereits "user" ist
    if not cleaned_history or cleaned_history[-1].get("role") != "user":
        messages.append({"role": "user", "content": request.message})
    else:
        # Wenn die letzte History-Nachricht bereits "user" ist, ersetze sie mit der aktuellen
        if messages and messages[-1].get("role") == "user":
            messages[-1] = {"role": "user", "content": request.message}
        else:
            messages.append({"role": "user", "content": request.message})
    
    # Speichere User-Nachricht
    conversation_manager.add_message(conversation_id, "user", request.message)
    
    # 🔥 CHATANGENT INTEGRATION - Nutze ChatAgent für Tool-Support (Web-Search etc.)
    use_chat_agent = True  # Flag um ChatAgent zu aktivieren
    
    # Streaming-Generator
    async def generate():
        try:
            # 🔥 CHATANGENT INTEGRATION - Versuche ChatAgent zu nutzen
            if use_chat_agent:
                try:
                    # Erstelle oder hole ChatAgent für diese Conversation
                    conversation_agents = agent_manager.get_conversation_agents(conversation_id)
                    chat_agent = None
                    
                    # Suche nach existierendem ChatAgent
                    for agent_info in conversation_agents:
                        if agent_info.get("type") == "chat_agent":
                            chat_agent = agent_manager.get_agent(conversation_id, agent_info["id"])
                            break
                    
                    # Erstelle neuen ChatAgent falls nicht vorhanden
                    if not chat_agent:
                        agent_id = agent_manager.create_agent(
                            conversation_id=conversation_id,
                            agent_type="chat_agent",
                            model_id=model_to_use,
                            set_managers_func=set_agent_managers
                        )
                        chat_agent = agent_manager.get_agent(conversation_id, agent_id)
                    
                    # Nutze ChatAgent für Antwort-Generierung (mit Tools/Web-Search)
                    response = chat_agent.process_message(request.message)
                    
                    # 🔧 FIX: Simuliere Streaming durch Chunks im richtigen Format
                    # Frontend erwartet {'chunk': '...'}, nicht {'type': 'content', 'content': '...'}
                    chunk_size = 15  # Zeichen pro Chunk
                    for i in range(0, len(response), chunk_size):
                        chunk = response[i:i+chunk_size]
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
                    # Fertig - Frontend prüft auf {'done': True}
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                    # Speichere Assistant-Nachricht
                    conversation_manager.add_message(conversation_id, "assistant", response)
                    
                    # Erfolg - return ohne Fallback
                    return
                    
                except Exception as e:
                    logger.warning(f"ChatAgent fehlgeschlagen, fallback auf normales Streaming: {e}")
                    # Fallback auf normalen Streaming-Modus unten
            
            effective_temperature = request.temperature if request.temperature > 0 else 0.3
            
            full_response = ""
            
            if USE_MODEL_SERVICE:
                # Model-Service unterstützt noch kein echtes Streaming
                # Verwende lokales Streaming als Fallback, wenn Modell lokal verfügbar ist
                try:
                    # Versuche lokales Streaming zu verwenden, wenn Modell lokal geladen ist
                    if model_manager.is_model_loaded():
                        # Lokales Streaming verwenden
                        for chunk in model_manager.generate_stream(
                            messages,
                            max_length=request.max_length,
                            temperature=effective_temperature
                        ):
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    else:
                        # Fallback: Normale Methode und simuliere Streaming
                        result = model_service_client.chat(
                            message=request.message,
                            messages=messages,
                            conversation_id=conversation_id,
                            max_length=request.max_length,
                            temperature=effective_temperature,
                            language=stream_response_language
                        )
                        if result:
                            response = result.get("response", "")
                            full_response = response
                            # Simuliere Streaming durch Chunks (kleinere Chunks für bessere UX)
                            chunk_size = 10
                            for i in range(0, len(response), chunk_size):
                                chunk = response[i:i+chunk_size]
                                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                                # Kleine Verzögerung für bessere UX
                                await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Fehler bei Streaming mit Model-Service: {e}")
                    # Fallback auf normale Methode
                    result = model_service_client.chat(
                        message=request.message,
                        messages=messages,
                        conversation_id=conversation_id,
                        max_length=request.max_length,
                        temperature=effective_temperature,
                        language=stream_response_language
                    )
                    if result:
                        response = result.get("response", "")
                        full_response = response
                        # Simuliere Streaming durch Chunks
                        chunk_size = 10
                        for i in range(0, len(response), chunk_size):
                            chunk = response[i:i+chunk_size]
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                            await asyncio.sleep(0.01)
            else:
                # Lokales Streaming
                for chunk in model_manager.generate_stream(
                    messages,
                    max_length=request.max_length,
                    temperature=effective_temperature
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Speichere vollständige Antwort
            if full_response:
                conversation_manager.add_message(conversation_id, "assistant", full_response)
        except Exception as e:
            logger.error(f"Fehler bei Streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat-Endpunkt - verarbeitet eine Nachricht und gibt eine Antwort zurück
    """
    ## Conversation ID prüfen/erstellen
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = conversation_manager.create_conversation(conversation_type="chat")
    
    # Prüfe ob Conversation ein Modell zugewiesen hat
    conversation_model_id = conversation_manager.get_conversation_model(conversation_id)
    
    # Bestimme welches Modell verwendet werden soll
    model_to_use = conversation_model_id  # Conversation-Modell hat Priorität
    
    # Wenn kein Conversation-Modell, prüfe ob globales Modell geladen ist
    if not model_to_use:
        if USE_MODEL_SERVICE:
            # Prüfe Model-Service Status
            status = model_service_client.get_text_model_status()
            if status and status.get("loaded"):
                model_to_use = status.get("model_id")
            else:
                # Lade Default-Modell
                default_model = model_manager.config.get("default_model")
                if default_model:
                    model_to_use = default_model
                else:
                    raise HTTPException(status_code=400, detail="Kein Modell geladen und kein Default-Modell konfiguriert")
        else:
            # Fallback auf lokale Manager
            if model_manager.is_model_loaded():
                model_to_use = model_manager.get_current_model()
            else:
                # Lade Default-Modell
                default_model = model_manager.config.get("default_model")
                if default_model:
                    model_to_use = default_model
                else:
                    raise HTTPException(status_code=400, detail="Kein Modell geladen und kein Default-Modell konfiguriert")
    
    
    
    # Stelle sicher dass Modell geladen ist
    if USE_MODEL_SERVICE:
        # Nutze Model-Service
        status = model_service_client.get_text_model_status()
        if status and status.get("loaded") and status.get("model_id") == model_to_use:
            # Modell ist bereits geladen
            model_ready = True
        else:
            # Prüfe ob Modell bereits geladen wird (im Model Service)
            # Der Model Service prüft intern, ob das Modell bereits geladen ist
            # Modell muss geladen werden
            if not model_service_client.load_text_model(model_to_use):
                raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Modells: {model_to_use}")
            model_ready = True
    else:
        # Fallback: Nutze lokale Manager
        model_ready = await ensure_text_model_loaded(model_to_use, conversation_id)
    
    # Wenn Modell noch nicht geladen ist, gib Status zurück
    if not model_ready:
        raise HTTPException(
            status_code=202,  # Accepted - wird verarbeitet
            detail={
                "status": "model_loading",
                "message": f"Modell {model_to_use} wird geladen. Bitte warten Sie.",
                "model_id": model_to_use,
                "conversation_id": conversation_id
            }
        )
    
    # IMMER ChatAgent verwenden - alle Chats sind jetzt Agenten mit Tool-Unterstützung
    # ChatAgent erkennt automatisch Tool-Bedarf (WebSearch, Datei-Operationen, etc.)
    try:
        # Stelle sicher dass Modell geladen ist
        if USE_MODEL_SERVICE:
            status = model_service_client.get_text_model_status()
            if not (status and status.get("loaded") and status.get("model_id") == model_to_use):
                if not model_service_client.load_text_model(model_to_use):
                    raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Modells: {model_to_use}")
        else:
            model_ready = await ensure_text_model_loaded(model_to_use, conversation_id)
            if not model_ready:
                raise HTTPException(
                    status_code=202,
                    detail={
                        "status": "model_loading",
                        "message": f"Modell {model_to_use} wird geladen. Bitte warten Sie.",
                        "model_id": model_to_use,
                        "conversation_id": conversation_id
                    }
                )
        
        # ==========================================
        # PHASE 1: RAG (wenn auto_web_search aktiv) - auch für ChatAgent
        # ==========================================
        sources = []
        sources_context = ""
        enhanced_message = request.message
        
        if quality_manager.settings.get("auto_web_search", False):
            # Prüfe ob Frage Web-Search benötigt (Fakten, aktuelle Infos, etc.)
            if quality_manager._needs_web_search(request.message):
                try:
                    logger.quality(f"RAG aktiviert: Starte Web-Search für '{request.message[:100]}...'")
                    # Web-Search durchführen (SYNCHRON - vor Generierung!)
                    # WICHTIG: Expliziter Timeout von 5 Sekunden um Blockierung zu vermeiden
                    search_results = quality_manager.web_search(request.message, max_results=5, timeout=5.0)
                    
                    if search_results and "results" in search_results:
                        sources = search_results["results"]
                        
                        # Formatiere Quellen als Kontext für LLM
                        sources_context = quality_manager.format_sources_for_context(sources)
                        
                        # Füge Kontext zur User-Nachricht hinzu (ChatAgent nutzt diese)
                        enhanced_message = f"{request.message}\n\nRelevante Informationen aus verifizierten Webquellen:\n{sources_context}\n\nNutze diese Informationen in deiner Antwort und referenziere die Quellen mit [1], [2], [3], etc."
                        
                        logger.quality(f"RAG erfolgreich: {len(sources)} Quellen als Kontext hinzugefügt")
                except Exception as e:
                    logger.quality(f"RAG Web-Search fehlgeschlagen: {e}, fahre ohne Kontext fort", level="warning")
        
        # Erstelle oder hole ChatAgent für diese Conversation
        conversation_agents = agent_manager.get_conversation_agents(conversation_id)
        chat_agent = None
        
        # Suche nach existierendem ChatAgent
        for agent_info in conversation_agents:
            if agent_info.get("type") == "chat_agent":
                chat_agent = agent_manager.get_agent(conversation_id, agent_info["id"])
                break
        
        # Erstelle neuen ChatAgent falls nicht vorhanden
        if not chat_agent:
            agent_id = agent_manager.create_agent(
                conversation_id=conversation_id,
                agent_type="chat_agent",
                model_id=model_to_use,
                set_managers_func=set_agent_managers
            )
            chat_agent = agent_manager.get_agent(conversation_id, agent_id)
        
        # Nutze ChatAgent für Antwort-Generierung (mit RAG-Kontext wenn vorhanden)
        logger.chat(f"Starte Antwort-Generierung mit ChatAgent (enhanced_message_length={len(enhanced_message)})")
        response = chat_agent.process_message(enhanced_message)
        logger.chat(f"Antwort-Generierung abgeschlossen: response_length={len(response)}")
        
        # ==========================================
        # PHASE 2: Post-Processing (Validation + Retry) - auch für ChatAgent
        # ==========================================
        # FIX: Retries aktivieren wenn EITHER web_validation ODER hallucination_check aktiv ist
        has_quality_checks = (
            quality_manager.settings.get("web_validation", False) or 
            quality_manager.settings.get("hallucination_check", False)
        )
        max_retries = 2 if has_quality_checks else 0
        best_response = response
        best_score = 1.0
        
        for attempt in range(max_retries + 1):
            # POST-PROCESSING: Sammle alle Check-Ergebnisse
            all_issues = []
            
            # CHECK 1: Hallucination-Check (unabhängig von web_validation!)
            if quality_manager.settings.get("hallucination_check", False):
                hallucination_issues = quality_manager._check_hallucinations(response)
                all_issues.extend(hallucination_issues)
                if hallucination_issues:
                    logger.info(f"Hallucination-Check (Versuch {attempt + 1}): {len(hallucination_issues)} Issues gefunden")
            
            # CHECK 2: Web-Validation (Vollständigkeit, Struktur, etc.)
            # FIX: Web-Validation kann parallel zu Hallucination-Check laufen
            if quality_manager.settings.get("web_validation", False):
                validation = quality_manager.validate_response(
                    response=response,
                    question=request.message,
                    auto_search=False  # Kein Web-Search (hatten wir schon in RAG)
                )
                
                # Füge Validation-Issues hinzu
                all_issues.extend(validation.get("issues", []))
                
                score = validation.get("confidence", 1.0)
                if score > best_score:
                    best_response = response
                    best_score = score
            else:
                validation = {
                    "valid": True,
                    "confidence": 1.0,
                    "issues": [],
                    "sources": sources,
                    "suggestions": []
                }
            
            # ENTSCHEIDUNG: Valid oder Retry?
            is_valid = len(all_issues) == 0
            
            if is_valid:
                logger.info(f"Response valide (Versuch {attempt + 1})")
                break  # Erfolgreich!
            else:
                logger.warning(f"Response hat Issues (Versuch {attempt + 1}): {all_issues}")
                
                # FIX: Retry wenn EITHER web_validation ODER hallucination_check aktiv ist UND Retries übrig
                if has_quality_checks and attempt < max_retries:
                    # Retry mit Feedback - füge Feedback zur Nachricht hinzu
                    feedback = quality_manager.generate_retry_prompt(
                        request.message, response, all_issues
                    )
                    enhanced_message_with_feedback = f"{enhanced_message}\n\n{feedback}"
                    
                    logger.info(f"Retry {attempt + 2}/{max_retries + 1} mit Feedback...")
                    response = chat_agent.process_message(enhanced_message_with_feedback)
                    continue  # Prüfe erneut
                else:
                    # Kein Retry möglich - nutze beste verfügbare Antwort
                    if best_response != response and has_quality_checks:
                        response = best_response
                        logger.info("Nutze beste verfügbare Antwort nach max Retries")
                    break
        
        # Quellen-Header für Response
        sources_header = ""
        if sources:
            source_links = []
            for i, source in enumerate(sources[:5], 1):
                url = source.get("url", "")
                title = source.get("title", f"Quelle {i}")
                if url:
                    source_links.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">[{i}] {title}</a>')
            
            if source_links:
                sources_header = f"<div style='margin-bottom: 1em; font-size: 0.9em; color: #666;'><strong>Quellen:</strong> {' | '.join(source_links)}</div>\n\n"
        
        if sources_header:
            response = sources_header + response
        
        # Speichere Nachrichten
        conversation_manager.add_message(conversation_id, "user", request.message)
        conversation_manager.add_message(conversation_id, "assistant", response)
        
        # Lerne aus Conversation (wenn aktiviert)
        if preference_learner.is_enabled():
            all_messages = conversation_manager.get_conversation_history(conversation_id)
            preference_learner.learn_from_conversation(all_messages)
        
        return ChatResponse(response=response, conversation_id=conversation_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler bei ChatAgent: {e}")
        logger.exception("Detaillierter Fehler:")
        ## Fallback auf normalen Chat-Modus bei Fehler
        logger.warning("Fallback auf normalen Chat-Modus (ohne Agent)")
        # Weiter mit normalem Flow unten
    
    # Lade Conversation History
    history = conversation_manager.get_conversation_history(conversation_id)
    
    # Erstelle Messages-Liste für Chat-Format
    messages = []
    
    # Bestimme Antwort-Sprache (flexibel basierend auf Audiosprache)
    response_language = request.language
    if not response_language:
        # Lese Sprache aus speech_input_app/config.json
        try:
            speech_config_path = os.path.join(workspace_root, "speech_input_app", "config.json")
            if os.path.exists(speech_config_path):
                with open(speech_config_path, 'r', encoding='utf-8') as f:
                    speech_config = json.load(f)
                    response_language = speech_config.get("language", "de")
                    logger.info(f"Sprache aus config.json gelesen: {response_language}")
            else:
                response_language = "de"  # Default: Deutsch
                logger.warning("speech_input_app/config.json nicht gefunden, verwende Deutsch")
        except Exception as e:
            logger.warning(f"Fehler beim Lesen der Sprach-Konfiguration: {e}, verwende Deutsch")
            response_language = "de"
    
    ## System-Prompt (mit Präferenzen wenn aktiviert)
    # WICHTIG: Verwende model_to_use (bereits bestimmt oben), nicht model_manager.get_current_model()
    # da bei USE_MODEL_SERVICE das Modell im Model-Service läuft, nicht lokal
    current_model = model_to_use  # Verwende bereits bestimmtes Modell
    
    # Prüfe ob Frage Coding-bezogen ist
    is_coding = quality_manager.is_coding_question(request.message)
    logger.debug(f"[UseCase] Coding Detection: is_coding={is_coding} für Nachricht: '{request.message[:100]}...'")
    
    # Generiere System-Prompt basierend auf Sprache und Modell
    if current_model and "phi-3" in current_model.lower():
        # Phi-3 verwendet ChatML-Format, kein separater System-Prompt nötig
        system_prompt = None
    elif current_model and "mistral" in current_model.lower():
        # Mistral: Sprachspezifischer Prompt
        if response_language == "en":
            system_prompt = "You are a helpful AI assistant. Answer briefly, precisely and directly in English. Keep answers under 200 words. Answer ONLY the asked question, no additional explanations or technical details."
        else:  # Deutsch (de) oder andere
            system_prompt = "Du bist ein hilfreicher AI-Assistent. Antworte kurz, präzise und direkt auf Deutsch. Halte Antworten unter 200 Wörtern. Antworte NUR auf die gestellte Frage, keine zusätzlichen Erklärungen oder technischen Details."
        system_prompt = preference_learner.get_system_prompt(system_prompt)
    elif current_model and "qwen" in current_model.lower():
        # Qwen: Hybrid-Prompt (kann chatten UND coden)
        if response_language == "en":
            system_prompt = """You are a helpful AI assistant who can both answer questions and write code.
- For questions: Answer clearly and directly
- For code requests: Use Markdown code blocks with language tags (```python, ```javascript, etc.)
- Only use code blocks when code is requested
- Include helpful comments in code when appropriate
- Explain code briefly if needed"""
        else:  # Deutsch (de) oder andere
            system_prompt = """Du bist ein hilfreicher AI-Assistent, der sowohl Fragen beantworten als auch Code schreiben kann.
- Bei Fragen: Antworte klar und direkt
- Bei Code-Anfragen: Verwende Markdown Code-Blocks mit Sprach-Tags (```python, ```javascript, etc.)
- Verwende Code-Blocks nur wenn Code gefragt ist
- Füge hilfreiche Kommentare in Code hinzu wenn angemessen
- Erkläre Code kurz wenn nötig"""
        system_prompt = preference_learner.get_system_prompt(system_prompt)
    else:
        # Andere Modelle: Sprachspezifischer Prompt
        if response_language == "en":
            system_prompt = "You are a helpful, precise and friendly AI assistant. Answer clearly and directly in English. IMPORTANT: Answer ONLY with your response, do NOT repeat the system prompt or user messages. Do NOT generate additional user or assistant messages."
        else:  # Deutsch (de) oder andere
            system_prompt = "Du bist ein hilfreicher, präziser und freundlicher AI-Assistent. Antworte klar und direkt auf Deutsch. WICHTIG: Antworte NUR mit deiner Antwort, wiederhole NICHT den System-Prompt oder User-Nachrichten. Generiere KEINE weiteren User- oder Assistant-Nachrichten."
        system_prompt = preference_learner.get_system_prompt(system_prompt)
    # ==========================================
    # PHASE 1: RAG (wenn auto_web_search aktiv)
    # ==========================================
    sources = []
    sources_context = ""
    
    if quality_manager.settings.get("auto_web_search", False):
        # Prüfe ob Frage Web-Search benötigt (Fakten, aktuelle Infos, etc.)
        if quality_manager._needs_web_search(request.message):
            try:
                # Web-Search durchführen (SYNCHRON - vor Generierung!)
                # WICHTIG: Expliziter Timeout von 5 Sekunden um Blockierung zu vermeiden
                search_results = quality_manager.web_search(request.message, max_results=5, timeout=5.0)
                
                if search_results and "results" in search_results:
                    sources = search_results["results"]
                    
                    # Formatiere Quellen als Kontext für LLM
                    sources_context = quality_manager.format_sources_for_context(sources)
                    
                    # Füge Kontext zum System-Prompt hinzu
                    if system_prompt:
                        system_prompt += f"\n\nRelevante Informationen aus verifizierten Webquellen:\n{sources_context}\n\nNutze diese Informationen in deiner Antwort und referenziere die Quellen mit [1], [2], [3], etc."
                    else:
                        # Falls kein System-Prompt, erstelle einen
                        system_prompt = f"Relevante Informationen aus verifizierten Webquellen:\n{sources_context}\n\nNutze diese Informationen in deiner Antwort und referenziere die Quellen mit [1], [2], [3], etc."
                    
                    logger.info(f"RAG aktiviert: {len(sources)} Quellen als Kontext hinzugefügt")
            except Exception as e:
                logger.warning(f"RAG Web-Search fehlgeschlagen: {e}, fahre ohne Kontext fort")
                sources = []  # Sicherstellen dass sources leer ist bei Fehler
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Conversation History hinzufügen
    
    
    # Filtere History: Nur user/assistant Rollen, keine agent_* Rollen
    # Entferne auch die aktuelle User-Nachricht falls sie bereits in der History ist
    filtered_history = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Überspringe die aktuelle User-Nachricht falls sie bereits in der History ist
        if role == "user" and content == request.message:
            continue
        
        # Konvertiere agent_* Rollen zu user
        if role.startswith("agent_"):
            role = "user"
        # Nur user/assistant Rollen erlauben
        if role in ["user", "assistant"]:
            filtered_history.append({"role": role, "content": content})
    
    # Stelle sicher, dass Rollen alternieren (keine zwei aufeinanderfolgenden gleichen Rollen)
    cleaned_history = []
    last_role = None
    for msg in filtered_history:
        current_role = msg.get("role")
        # Überspringe wenn die Rolle gleich der vorherigen ist
        if current_role == last_role:
            continue
        cleaned_history.append(msg)
        last_role = current_role
    
    # Stelle sicher, dass nach System-Prompt die erste Nachricht "user" ist
    # Wenn die History mit "assistant" beginnt, entferne sie
    if cleaned_history and cleaned_history[0].get("role") == "assistant":
        cleaned_history = cleaned_history[1:]  # Entferne erste "assistant" Nachricht
    
    filtered_history = cleaned_history
    
    
    
    messages.extend(filtered_history)
    
    # Aktuelle Nachricht hinzufügen - nur wenn die letzte Nachricht nicht bereits "user" ist
    # (um sicherzustellen, dass Rollen alternieren)
    if not filtered_history or filtered_history[-1].get("role") != "user":
        messages.append({"role": "user", "content": request.message})
    else:
        # Wenn die letzte History-Nachricht bereits "user" ist, ersetze die letzte Message
        # (die bereits in messages ist, da wir filtered_history hinzugefügt haben)
        if messages and len(messages) > 0 and messages[-1].get("role") == "user":
            messages[-1] = {"role": "user", "content": request.message}
        else:
            messages.append({"role": "user", "content": request.message})
    
    
    
    # Logge Messages (Memory 4468610)
    logger.info(f"Messages: {len(messages)} Nachrichten")
    
    ## Generiere Antwort
    try:
        # Coding-Optimierungen für Qwen: niedrigere Temperature, höhere max_length
        if current_model and "qwen" in current_model.lower() and is_coding:
            effective_temperature = request.temperature if request.temperature > 0 else 0.2  # Niedriger für präziseren Code
            effective_max_length = max(request.max_length, 4096)  # Höher für längeren Code
        else:
            effective_temperature = request.temperature if request.temperature > 0 else 0.3
            effective_max_length = request.max_length
        
        if USE_MODEL_SERVICE:
            # Nutze Model-Service
            # WICHTIG: Chat-Request in Thread Pool ausführen (blockiert Event Loop nicht)
            def chat_in_thread():
                """Chat-Request in separatem Thread"""
                try:
                    return model_service_client.chat(
                        message=request.message,
                        messages=messages,  # Sende vollständige Messages-Liste
                        conversation_id=conversation_id,
                        max_length=effective_max_length,
                        temperature=effective_temperature,
                        language=response_language  # Sende Sprache mit
                    )
                except Exception as e:
                    logger.error(f"Fehler bei Model-Service Chat im Thread: {e}")
                    raise
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                model_generation_executor,
                chat_in_thread
            )
            if result:
                response = result.get("response", "")
                logger.info(f"Model-Service Response erhalten: Länge={len(response)} Zeichen")
                if not response:
                    logger.warning("Model-Service hat leere Response zurückgegeben!")
                    raise HTTPException(status_code=500, detail="Model-Service hat leere Response zurückgegeben")
            else:
                logger.error("Model-Service hat kein Ergebnis zurückgegeben!")
                raise HTTPException(status_code=500, detail="Model-Service hat kein Ergebnis zurückgegeben")
        else:
            # Fallback: Nutze lokale Manager
            # WICHTIG: Generierung in Thread Pool ausführen (blockiert Event Loop nicht)
            def generate_in_thread():
                """Generiert Antwort in separatem Thread"""
                try:
                    return model_manager.generate(
                        messages,
                        max_length=effective_max_length,
                        temperature=effective_temperature
                    )
                except Exception as e:
                    logger.error(f"Fehler bei Modell-Generierung im Thread: {e}")
                    raise
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                model_generation_executor,
                generate_in_thread
            )
        
        # Logge Response
        if response:
            logger.info(f"Response erhalten: Länge={len(response)} Zeichen, Vorschau: {response[:200]}...")
        else:
            logger.error("Response ist leer oder None!")
            raise HTTPException(status_code=500, detail="Response ist leer")
        
        # HINWEIS: Response-Bereinigung erfolgt bereits in model_manager.generate()
        # Keine doppelte Bereinigung hier - das würde gültige Antworten beschädigen
        # Die Response ist bereits vollständig bereinigt und bereit für Quality Management
        
        # ==========================================
        # PHASE 2: Post-Processing (Validation + Retry)
        # ==========================================
        # FIX: Retries aktivieren wenn EITHER web_validation ODER hallucination_check aktiv ist
        has_quality_checks = (
            quality_manager.settings.get("web_validation", False) or 
            quality_manager.settings.get("hallucination_check", False)
        )
        max_retries = 2 if has_quality_checks else 0
        best_response = response
        best_score = 1.0
        original_messages = messages.copy()  # Backup für Retries
        
        for attempt in range(max_retries + 1):
            # POST-PROCESSING: Sammle alle Check-Ergebnisse
            all_issues = []
            
            # CHECK 1: Hallucination-Check (unabhängig von web_validation!)
            if quality_manager.settings.get("hallucination_check", False):
                hallucination_issues = quality_manager._check_hallucinations(response)
                all_issues.extend(hallucination_issues)
                if hallucination_issues:
                    logger.info(f"Hallucination-Check (Versuch {attempt + 1}): {len(hallucination_issues)} Issues gefunden")
            
            # CHECK 2: Web-Validation (Vollständigkeit, Struktur, etc.)
            # FIX: Web-Validation kann parallel zu Hallucination-Check laufen
            if quality_manager.settings.get("web_validation", False):
                validation = quality_manager.validate_response(
                    response=response,
                    question=request.message,
                    auto_search=False  # Kein Web-Search (hatten wir schon in RAG)
                )
                
                # Füge Validation-Issues hinzu
                all_issues.extend(validation.get("issues", []))
                
                score = validation.get("confidence", 1.0)
                if score > best_score:
                    best_response = response
                    best_score = score
            else:
                # Kein Validation - setze default
                validation = {
                    "valid": True,
                    "confidence": 1.0,
                    "issues": [],
                    "sources": sources,
                    "suggestions": []
                }
            
            # ENTSCHEIDUNG: Valid oder Retry?
            is_valid = len(all_issues) == 0
            
            if is_valid:
                logger.info(f"Response valide (Versuch {attempt + 1})")
                break  # Erfolgreich!
            else:
                logger.warning(f"Response hat Issues (Versuch {attempt + 1}): {all_issues}")
                
                # FIX: Retry wenn EITHER web_validation ODER hallucination_check aktiv ist UND Retries übrig
                if has_quality_checks and attempt < max_retries:
                    # Retry mit Feedback
                    feedback = quality_manager.generate_retry_prompt(
                        request.message, response, all_issues
                    )
                    # Erstelle neue Messages-Liste mit Feedback
                    messages = original_messages.copy()
                    messages.append({"role": "system", "content": feedback})
                    
                    # Regeneriere Response
                    logger.info(f"Retry {attempt + 2}/{max_retries + 1} mit Feedback...")
                    
                    if USE_MODEL_SERVICE:
                        def chat_retry_in_thread():
                            return model_service_client.chat(
                                message=request.message,
                                messages=messages,
                                conversation_id=conversation_id,
                                max_length=effective_max_length,
                                temperature=effective_temperature,
                                language=response_language
                            )
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            model_generation_executor,
                            chat_retry_in_thread
                        )
                        if result:
                            response = result.get("response", "")
                    else:
                        def generate_retry_in_thread():
                            return model_manager.generate(
                                messages,
                                max_length=effective_max_length,
                                temperature=effective_temperature,
                                is_coding=is_coding
                            )
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            model_generation_executor,
                            generate_retry_in_thread
                        )
                    continue  # Prüfe erneut
                else:
                    # Kein Retry möglich - nutze beste verfügbare Antwort
                    if best_response != response and has_quality_checks:
                        response = best_response
                        logger.info("Nutze beste verfügbare Antwort nach max Retries")
                    break
        
        # Finale Validation für Response-Formatierung
        validation = {
            "valid": len(all_issues) == 0,
            "confidence": best_score,
            "issues": all_issues,
            "sources": sources,
            "suggestions": validation.get("suggestions", []) if quality_manager.settings.get("web_validation", False) else []
        }
        
        # Quellen im Fließtext als klickbare Links im Header der Antwort
        sources_header = ""
        if validation["sources"]:
            source_links = []
            for i, source in enumerate(validation["sources"][:5], 1):  # Max 5 Quellen
                url = source.get("url", "")
                title = source.get("title", f"Quelle {i}")
                if url:
                    source_links.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">[{i}] {title}</a>')
            
            if source_links:
                sources_header = f"<div style='margin-bottom: 1em; font-size: 0.9em; color: #666;'><strong>Quellen:</strong> {' | '.join(source_links)}</div>\n\n"
        
        # Füge Quellen-Header zur Response hinzu
        if sources_header:
            response = sources_header + response
        
        # Füge Quality-Info zur Response hinzu (optional, nur bei niedriger Konfidenz)
        if validation["confidence"] < 0.7:
            response += f"\n\n<div style='margin-top: 1em; padding: 0.5em; background: #fff3cd; border-left: 3px solid #ffc107;'><strong>Warnung:</strong> Diese Antwort hat eine niedrige Konfidenz ({validation['confidence']:.0%}). Bitte prüfen Sie die Quellen.</div>"
        
        # Speichere Nachrichten mit Quality-Info und Quellen (nur einmal)
        response_id = str(uuid.uuid4())
        conversation_manager.add_message(conversation_id, "user", request.message)
        conversation_manager.add_message(conversation_id, "assistant", response, metadata={
            "quality": validation,
            "response_id": response_id,
            "sources": validation["sources"]  # Quellen für spätere Referenz
        })
        
        # Lerne aus Conversation (wenn aktiviert)
        if preference_learner.is_enabled():
            all_messages = conversation_manager.get_conversation_history(conversation_id)
            preference_learner.learn_from_conversation(all_messages)
        
        # Finale Prüfung vor Return
        if not response:
            logger.error("Response ist leer vor Return!")
            raise HTTPException(status_code=500, detail="Response ist leer")
        
        logger.info(f"Return ChatResponse: Länge={len(response)} Zeichen, conversation_id={conversation_id}")
        return ChatResponse(response=response, conversation_id=conversation_id)
        
    except HTTPException:
        # HTTPExceptions direkt weiterwerfen
        raise
    except Exception as e:
        logger.error(f"Fehler bei der Generierung: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Generierung: {str(e)}")


@app.get("/conversations")
async def get_conversations():
    """Gibt alle Conversations zurück"""
    return {"conversations": conversation_manager.get_all_conversations()}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Lädt eine spezifische Conversation"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    return conversation


@app.post("/conversations")
async def create_conversation():
    """Erstellt eine neue Chat-Conversation"""
    conversation_id = conversation_manager.create_conversation(conversation_type="chat")
    return {"conversation_id": conversation_id}

@app.post("/conversations/image")
async def create_image_conversation():
    """Erstellt eine neue Image-Conversation"""
    conversation_id = conversation_manager.create_conversation(conversation_type="image")
    return {"conversation_id": conversation_id}


@app.get("/conversations/{conversation_id}/file-mode")
async def get_file_mode(conversation_id: str):
    """Gibt den File-Mode-Status einer Conversation zurück (nur Datei-Operationen)"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    
    file_mode = conversation_manager.get_file_mode(conversation_id)
    return {"conversation_id": conversation_id, "file_mode": file_mode}


# Alias für Rückwärtskompatibilität
@app.get("/conversations/{conversation_id}/agent-mode")
async def get_agent_mode(conversation_id: str):
    """Alias für get_file_mode (Rückwärtskompatibilität)"""
    return await get_file_mode(conversation_id)


class SetFileModeRequest(BaseModel):
    enabled: bool


@app.post("/conversations/{conversation_id}/file-mode")
async def set_file_mode(conversation_id: str, request: SetFileModeRequest):
    """Aktiviert oder deaktiviert den File-Mode für eine Conversation (nur Datei-Operationen)"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    
    success = conversation_manager.set_file_mode(conversation_id, request.enabled)
    if not success:
        raise HTTPException(status_code=500, detail="Fehler beim Setzen des File-Modus")
    
    return {
        "conversation_id": conversation_id,
        "file_mode": request.enabled,
        "message": f"File-Mode {'aktiviert' if request.enabled else 'deaktiviert'} (nur Datei-Operationen, Web-Search ist immer aktiv)"
    }


# Alias für Rückwärtskompatibilität
class SetAgentModeRequest(BaseModel):
    enabled: bool


@app.post("/conversations/{conversation_id}/agent-mode")
async def set_agent_mode(conversation_id: str, request: SetAgentModeRequest):
    """Alias für set_file_mode (Rückwärtskompatibilität)"""
    return await set_file_mode(conversation_id, SetFileModeRequest(enabled=request.enabled))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Löscht eine Conversation"""
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    return {"message": "Conversation gelöscht"}


@app.post("/conversations/{conversation_id}/model")
async def set_conversation_model(conversation_id: str, request: SetConversationModelRequest):
    """Setzt das Modell für eine Conversation"""
    # Prüfe ob Conversation existiert
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    
    # Prüfe ob Modell existiert (wenn angegeben)
    if request.model_id:
        available_models = model_manager.get_available_models()
        if request.model_id not in available_models:
            raise HTTPException(status_code=400, detail=f"Modell nicht gefunden: {request.model_id}")
    
    # Setze Modell
    success = conversation_manager.set_conversation_model(conversation_id, request.model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Fehler beim Setzen des Modells")
    
    return {"message": f"Modell für Conversation gesetzt: {request.model_id or 'Kein Modell'}", "model_id": request.model_id}


# Agent API Endpoints

class CreateAgentRequest(BaseModel):
    agent_type: str
    model_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class AgentChatRequest(BaseModel):
    message: str


class CreatePipelineRequest(BaseModel):
    name: str
    steps: List[Dict[str, Any]]
    initial_input: str


@app.get("/agents/types")
async def get_agent_types():
    """Gibt alle verfügbaren Agent-Typen zurück"""
    return {"agent_types": agent_manager.get_available_agent_types()}


@app.get("/agents")
async def get_agents(conversation_id: Optional[str] = None):
    """Gibt Agenten zurück (optional gefiltert nach Conversation)"""
    if conversation_id:
        agents = agent_manager.get_conversation_agents(conversation_id)
        return {"agents": agents, "conversation_id": conversation_id}
    else:
        # Gibt alle Agenten aller Conversations zurück
        all_agents = []
        for conv_id in agent_manager.agent_instances.keys():
            agents = agent_manager.get_conversation_agents(conv_id)
            for agent in agents:
                agent["conversation_id"] = conv_id
            all_agents.extend(agents)
        return {"agents": all_agents}


@app.post("/agents/create")
async def create_agent(conversation_id: str, request: CreateAgentRequest):
    """Erstellt einen neuen Agent für eine Conversation"""
    # Prüfe ob Conversation existiert
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    
    try:
        agent_id = agent_manager.create_agent(
            conversation_id=conversation_id,
            agent_type=request.agent_type,
            model_id=request.model_id,
            config=request.config,
            set_managers_func=set_agent_managers
        )
        
        agent = agent_manager.get_agent(conversation_id, agent_id)
        return {
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "agent_info": agent.get_info() if agent else None
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Agenten: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Erstellen des Agenten: {str(e)}")


@app.get("/agents/{conversation_id}/{agent_id}")
async def get_agent(conversation_id: str, agent_id: str):
    """Gibt einen Agent zurück"""
    agent = agent_manager.get_agent(conversation_id, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    return {"agent": agent.get_info()}


@app.post("/agents/{conversation_id}/{agent_id}/chat")
async def agent_chat(conversation_id: str, agent_id: str, request: AgentChatRequest):
    """Chat mit einem Agent"""
    agent = agent_manager.get_agent(conversation_id, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    
    try:
        response = agent.process_message(request.message)
        return {"response": response, "agent_id": agent_id}
    except Exception as e:
        logger.error(f"Fehler beim Agent-Chat: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Agent-Chat: {str(e)}")


@app.delete("/agents/{conversation_id}/{agent_id}")
async def delete_agent(conversation_id: str, agent_id: str):
    """Löscht einen Agent"""
    success = agent_manager.delete_agent(conversation_id, agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent nicht gefunden")
    return {"message": "Agent gelöscht"}


@app.post("/restart")
async def restart_server():
    """Startet den Server neu (ruft start_local_ai.bat auf)"""
    try:
        import subprocess
        import sys
        
        # Pfad zum start_local_ai.bat
        project_root = os.path.dirname(os.path.dirname(__file__))
        restart_script = os.path.join(project_root, "start_local_ai.bat")
        
        if not os.path.exists(restart_script):
            raise HTTPException(status_code=404, detail="Restart-Script nicht gefunden")
        
        # Starte Restart-Script in neuem Prozess (nicht-blockierend)
        # Auf Windows: start ohne Warten
        if sys.platform == "win32":
            subprocess.Popen(
                [restart_script],
                cwd=project_root,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                ["bash", restart_script],
                cwd=project_root,
                start_new_session=True
            )
        
        logger.info("Server-Restart initiiert")
        return {
            "status": "success",
            "message": "Server wird neu gestartet..."
        }
    except Exception as e:
        logger.error(f"Fehler beim Server-Restart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/tools")
async def get_tools():
    """Gibt alle verfügbaren Tools zurück"""
    return {"tools": agent_manager.get_available_tools()}


@app.post("/agents/pipeline")
async def create_and_execute_pipeline(conversation_id: str, request: CreatePipelineRequest):
    """Erstellt und führt eine Pipeline aus"""
    # Prüfe ob Conversation existiert
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    
    try:
        # Erstelle Pipeline
        pipeline_id = pipeline_manager.create_pipeline(
            conversation_id=conversation_id,
            pipeline_name=request.name,
            steps=request.steps
        )
        
        # Führe Pipeline aus
        result = pipeline_manager.execute_pipeline(pipeline_id, request.initial_input)
        
        return result
    except Exception as e:
        logger.error(f"Fehler bei Pipeline-Ausführung: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei Pipeline-Ausführung: {str(e)}")


@app.get("/agents/pipeline/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Gibt eine Pipeline zurück"""
    pipeline = pipeline_manager.get_pipeline(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline nicht gefunden")
    return {"pipeline": pipeline}


@app.get("/agents/pipelines/{conversation_id}")
async def get_conversation_pipelines(conversation_id: str):
    """Gibt alle Pipelines einer Conversation zurück"""
    pipelines = pipeline_manager.get_conversation_pipelines(conversation_id)
    return {"pipelines": pipelines}


@app.get("/config")
async def get_config():
    """Gibt die Konfiguration zurück (für Frontend)"""
    return {
        "image_generation": config.get("image_generation", {
            "resolution_presets": {
                "s": 512,
                "m": 720,
                "l": 1024
            }
        })
    }

@app.get("/preferences")
async def get_preferences():
    """Gibt die aktuellen Präferenzen zurück"""
    return preference_learner.get_preferences()


@app.post("/preferences/toggle")
async def toggle_preferences():
    """Schaltet Preference Learning ein/aus"""
    current = preference_learner.is_enabled()
    preference_learner.set_enabled(not current)
    return {"enabled": not current}


@app.post("/preferences/reset")
async def reset_preferences():
    """Setzt Präferenzen auf Default zurück"""
    preference_learner.reset_preferences()
    return {"message": "Präferenzen zurückgesetzt"}


# Quality Management Endpoints
class FeedbackRequest(BaseModel):
    response_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    issues: Optional[List[str]] = None  # ["hallucination", "incomplete", "wrong", etc.]


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Empfängt Nutzerrückmeldung zu einer Antwort"""
    quality_manager.add_feedback(
        response_id=request.response_id,
        feedback={
            "rating": request.rating,  # 1-5
            "comment": request.comment,
            "issues": request.issues  # ["hallucination", "incomplete", "wrong", etc.]
        }
    )
    return {"status": "success"}


@app.get("/quality/settings")
async def get_quality_settings():
    """Gibt Quality Settings zurück"""
    return quality_manager.get_settings()


class QualitySettingsRequest(BaseModel):
    web_validation: Optional[bool] = None
    contradiction_check: Optional[bool] = None
    hallucination_check: Optional[bool] = None
    actuality_check: Optional[bool] = None
    source_quality_check: Optional[bool] = None
    completeness_check: Optional[bool] = None
    auto_web_search: Optional[bool] = None


@app.post("/quality/settings")
async def update_quality_settings(request: QualitySettingsRequest):
    """Aktualisiert Quality Settings"""
    for key, value in request.dict(exclude_unset=True).items():
        if key in quality_manager.settings and value is not None:
            quality_manager.update_setting(key, value)
    return quality_manager.get_settings()


# Performance-Einstellungen
PERFORMANCE_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "performance_settings.json")
AUDIO_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audio_settings.json")

def _load_performance_settings() -> Dict[str, Any]:
    """Lädt Performance-Einstellungen"""
    # Default-Werte
    default_settings = {
        "cpu_threads": None,  # None = Auto
        "gpu_optimization": "balanced",
        "disable_cpu_offload": False,
        "use_torch_compile": False,
        "use_quantization": False,
        "quantization_bits": 8,
        "use_flash_attention": True,
        "enable_tf32": True,
        "enable_cudnn_benchmark": True
    }
    
    try:
        if os.path.exists(PERFORMANCE_SETTINGS_FILE):
            with open(PERFORMANCE_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
                # Merge mit Defaults (für neue Optionen)
                default_settings.update(loaded_settings)
                return default_settings
    except Exception as e:
        logger.warning(f"Fehler beim Laden der Performance-Einstellungen: {e}")
    
    return default_settings

def _save_performance_settings(settings: Dict[str, Any]):
    """Speichert Performance-Einstellungen"""
    try:
        os.makedirs(os.path.dirname(PERFORMANCE_SETTINGS_FILE), exist_ok=True)
        with open(PERFORMANCE_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Performance-Einstellungen: {e}")

@app.get("/performance/settings", response_model=PerformanceSettingsResponse)
async def get_performance_settings():
    """Gibt die aktuellen Performance-Einstellungen zurück"""
    settings = _load_performance_settings()
    return PerformanceSettingsResponse(**settings)

@app.post("/performance/settings", response_model=PerformanceSettingsResponse)
async def set_performance_settings(request: PerformanceSettingsRequest):
    """Setzt Performance-Einstellungen"""
    # Lade aktuelle Settings
    current_settings = _load_performance_settings()
    
    # Aktualisiere nur gesetzte Werte
    settings = {
        "cpu_threads": request.cpu_threads if request.cpu_threads is not None else current_settings.get("cpu_threads"),
        "gpu_optimization": request.gpu_optimization if request.gpu_optimization else current_settings.get("gpu_optimization", "balanced"),
        "disable_cpu_offload": request.disable_cpu_offload if request.disable_cpu_offload is not None else current_settings.get("disable_cpu_offload", False),
        "use_torch_compile": request.use_torch_compile if request.use_torch_compile is not None else current_settings.get("use_torch_compile", False),
        "use_quantization": request.use_quantization if request.use_quantization is not None else current_settings.get("use_quantization", False),
        "quantization_bits": request.quantization_bits if request.quantization_bits is not None else current_settings.get("quantization_bits", 8),
        "use_flash_attention": request.use_flash_attention if request.use_flash_attention is not None else current_settings.get("use_flash_attention", True),
        "enable_tf32": request.enable_tf32 if request.enable_tf32 is not None else current_settings.get("enable_tf32", True),
        "enable_cudnn_benchmark": request.enable_cudnn_benchmark if request.enable_cudnn_benchmark is not None else current_settings.get("enable_cudnn_benchmark", True)
    }
    _save_performance_settings(settings)
    
    # Wende CPU-Threads sofort an
    if request.cpu_threads and request.cpu_threads > 0:
        torch.set_num_threads(request.cpu_threads)
        torch.set_num_interop_threads(request.cpu_threads)
        logger.info(f"CPU-Threads auf {request.cpu_threads} gesetzt")
    else:
        # Auto - verwende alle verfügbaren Threads
        import os as os_module
        num_threads = os_module.cpu_count() or 4
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        logger.info(f"CPU-Threads auf Auto ({num_threads}) gesetzt")
    
    logger.info(f"Performance-Einstellungen gespeichert: {settings}")
    return PerformanceSettingsResponse(**settings)


def _load_audio_settings() -> Dict[str, Any]:
    """Lädt Audio-Einstellungen"""
    try:
        if os.path.exists(AUDIO_SETTINGS_FILE):
            with open(AUDIO_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Fehler beim Laden der Audio-Einstellungen: {e}")
    return {
        "transcription_language": ""  # Leerer String = Auto-Erkennung
    }

def _save_audio_settings(settings: Dict[str, Any]):
    """Speichert Audio-Einstellungen"""
    try:
        os.makedirs(os.path.dirname(AUDIO_SETTINGS_FILE), exist_ok=True)
        with open(AUDIO_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Audio-Einstellungen: {e}")

@app.get("/audio/settings", response_model=AudioSettingsResponse)
async def get_audio_settings():
    """Gibt die aktuellen Audio-Einstellungen zurück"""
    settings = _load_audio_settings()
    return AudioSettingsResponse(**settings)

@app.post("/audio/settings", response_model=AudioSettingsResponse)
async def set_audio_settings(request: AudioSettingsRequest):
    """Setzt Audio-Einstellungen"""
    settings = {
        "transcription_language": request.transcription_language or ""
    }
    _save_audio_settings(settings)
    logger.info(f"Audio-Einstellungen gespeichert: {settings}")
    return AudioSettingsResponse(**settings)


# Output Settings Endpoints
class OutputSettingsRequest(BaseModel):
    base_directory: Optional[str] = None
    use_date_folders: Optional[bool] = None
    filename_format: Optional[str] = None

class OutputSettingsResponse(BaseModel):
    base_directory: str
    images_subdir: str
    conversations_subdir: str
    audio_subdir: str
    use_date_folders: bool
    filename_format: str

@app.get("/output/settings", response_model=OutputSettingsResponse)
async def get_output_settings():
    """Gibt die aktuellen Output-Einstellungen zurück"""
    try:
        from output_manager import get_output_manager
        output_mgr = get_output_manager()
        
        return OutputSettingsResponse(
            base_directory=output_mgr.base_directory,
            images_subdir=output_mgr.images_subdir,
            conversations_subdir=output_mgr.conversations_subdir,
            audio_subdir=output_mgr.audio_subdir,
            use_date_folders=output_mgr.use_date_folders,
            filename_format=output_mgr.filename_format
        )
    except Exception as e:
        logger.error(f"Fehler beim Laden der Output-Einstellungen: {e}")
        # Fallback-Defaults
        return OutputSettingsResponse(
            base_directory="G:\\KI Modelle\\Outputs",
            images_subdir="generated_images",
            conversations_subdir="conversations",
            audio_subdir="audio",
            use_date_folders=True,
            filename_format="{date}_{title}"
        )

@app.post("/output/settings", response_model=OutputSettingsResponse)
async def update_output_settings(request: OutputSettingsRequest):
    """Aktualisiert Output-Einstellungen"""
    try:
        from output_manager import get_output_manager
        output_mgr = get_output_manager()
        
        # Aktualisiere Settings
        success = output_mgr.update_config(
            base_directory=request.base_directory,
            use_date_folders=request.use_date_folders,
            filename_format=request.filename_format
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Fehler beim Speichern der Einstellungen")
        
        return OutputSettingsResponse(
            base_directory=output_mgr.base_directory,
            images_subdir=output_mgr.images_subdir,
            conversations_subdir=output_mgr.conversations_subdir,
            audio_subdir=output_mgr.audio_subdir,
            use_date_folders=output_mgr.use_date_folders,
            filename_format=output_mgr.filename_format
        )
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren der Output-Einstellungen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/models")
async def get_image_models():
    """Gibt alle verfügbaren Bildgenerierungsmodelle zurück"""
    return {
        "models": image_manager.get_available_models(),
        "current_model": image_manager.get_current_model()
    }


def _load_image_model_async(model_id: str, conversation_id: Optional[str] = None):
    """Lädt ein Bild-Modell im Hintergrund"""
    import time
    import threading
    import os
    import json
    log_path = r'g:\04-CODING\Local Ai\.cursor\debug.log'
    
    
    
    start_time = time.time()
    max_load_time = 600  # 10 Minuten Maximum für Modell-Laden
    
    try:
        loading_status["image_model"]["loading"] = True
        loading_status["image_model"]["model_id"] = model_id
        loading_status["image_model"]["error"] = None
        loading_status["image_model"]["conversation_id"] = conversation_id
        loading_status["image_model"]["start_time"] = start_time
        
        logger.info(f"Starte Laden des Bildmodells {model_id} (max. {max_load_time}s)...")
        
        # Prüfe dynamisch ob Model-Service verfügbar ist
        use_model_service = check_model_service_available()
        
        
        
        # Lade Modell - verwende Model Service wenn verfügbar, sonst lokalen Manager
        if use_model_service:
            # Nutze Model Service
            logger.info(f"Lade Bildmodell über Model Service: {model_id}")
            success = model_service_client.load_image_model(model_id)
            if success:
                # Warte kurz und prüfe Status
                time.sleep(1)
                status = model_service_client.get_image_model_status()
                if status and status.get("loaded") and status.get("model_id") == model_id:
                    logger.info(f"Bildmodell erfolgreich über Model Service geladen: {model_id}")
                else:
                    logger.warning(f"Model Service meldet Modell als geladen, aber Status-Prüfung fehlgeschlagen")
        else:
            # Fallback: Nutze lokalen Manager
            if not image_manager:
                logger.error("ImageManager nicht verfügbar und Model Service nicht verfügbar")
                success = False
            else:
                logger.info(f"Lade Bildmodell lokal: {model_id}")
                success = image_manager.load_model(model_id)
        
        
        
        elapsed_time = time.time() - start_time
        if not success:
            loading_status["image_model"]["error"] = f"Fehler beim Laden des Bildgenerierungsmodells: {model_id}"
            logger.error(f"Modell-Laden fehlgeschlagen nach {elapsed_time:.1f}s: {model_id}")
        else:
            loading_status["image_model"]["error"] = None
            logger.info(f"Modell erfolgreich geladen nach {elapsed_time:.1f}s: {model_id}")
            
    except KeyboardInterrupt:
        # Wird bei Unterbrechung aufgerufen
        loading_status["image_model"]["error"] = "Modell-Laden wurde unterbrochen"
        logger.warning(f"Modell-Laden wurde unterbrochen: {model_id}")
    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        loading_status["image_model"]["error"] = f"Fehler beim Laden: {error_msg}"
        logger.error(f"Fehler beim asynchronen Laden des Bildmodells nach {elapsed_time:.1f}s: {e}", exc_info=True)
        
        
    finally:
        
        
        # Stelle sicher dass loading-Status immer zurückgesetzt wird
        loading_status["image_model"]["loading"] = False
        loading_status["image_model"]["conversation_id"] = None
        loading_status["image_model"]["start_time"] = None
        elapsed_time = time.time() - start_time
        logger.info(f"Modell-Ladevorgang beendet nach {elapsed_time:.1f}s: {model_id}")
        
        

@app.post("/image/models/load")
async def load_image_model(request: LoadModelRequest):
    """Lädt ein Bildgenerierungsmodell im Hintergrund (blockiert UI nicht)"""
    if not image_manager:
        raise HTTPException(status_code=503, detail="Bildgenerierung nicht verfügbar (diffusers/xformers nicht installiert)")
    
    # Prüfe ob bereits ein Modell geladen wird
    if loading_status["image_model"]["loading"]:
        raise HTTPException(status_code=400, detail="Ein Bildmodell wird bereits geladen. Bitte warten Sie.")
    
    # Starte Ladevorgang im Hintergrund
    loop = asyncio.get_event_loop()
    loop.run_in_executor(image_load_executor, _load_image_model_async, request.model_id, None)
    
    return {
        "message": f"Bildgenerierungsmodell wird geladen: {request.model_id}",
        "model_id": request.model_id,
        "status": "loading"
    }

@app.get("/image/models/load/status")
async def get_image_model_load_status():
    """Gibt den Status des Bildmodell-Ladevorgangs zurück"""
    import time
    
    # Prüfe ob Loading länger als 10 Minuten dauert (möglicherweise hängt)
    status = loading_status["image_model"].copy()
    if status["loading"] and status.get("start_time"):
        elapsed = time.time() - status["start_time"]
        status["elapsed_seconds"] = elapsed
        if elapsed > 600:  # 10 Minuten
            logger.warning(f"Modell-Laden läuft bereits {elapsed:.1f}s - möglicherweise hängt es: {status['model_id']}")
            status["warning"] = f"Laden läuft bereits {elapsed:.1f}s - möglicherweise hängt"
    
    
    # Entferne start_time aus der Antwort (intern)
    status.pop("start_time", None)
    return status


@app.post("/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Lädt eine Datei hoch und parst sie (CSV, TXT, JSON)
    Gibt strukturierte Daten zurück
    """
    try:
        # Lese Dateiinhalt
        content = await file.read()
        text_content = content.decode('utf-8')
        
        file_info = {
            "filename": file.filename,
            "size": len(content),
            "type": file.filename.split('.')[-1].lower() if '.' in file.filename else 'txt',
            "content": text_content,
            "parsed_data": None
        }
        
        # Parse basierend auf Dateityp
        if file_info["type"] == "csv":
            try:
                # CSV parsen
                csv_reader = csv.DictReader(io.StringIO(text_content))
                rows = list(csv_reader)
                file_info["parsed_data"] = {
                    "type": "csv",
                    "headers": list(rows[0].keys()) if rows else [],
                    "rows": rows[:100],  # Erste 100 Zeilen für Preview
                    "total_rows": len(rows),
                    "preview": rows[:10]  # Erste 10 Zeilen für Preview
                }
            except Exception as e:
                logger.warning(f"Fehler beim CSV-Parsing: {e}")
                # Fallback: Einfaches Zeilen-basiertes Parsing
                lines = text_content.split('\n')
                file_info["parsed_data"] = {
                    "type": "csv_simple",
                    "rows": [line.split(',') for line in lines[:100]],
                    "total_rows": len(lines)
                }
        elif file_info["type"] == "json":
            try:
                import json
                json_data = json.loads(text_content)
                file_info["parsed_data"] = {
                    "type": "json",
                    "data": json_data
                }
            except Exception as e:
                logger.warning(f"Fehler beim JSON-Parsing: {e}")
        else:
            # Text-Datei
            lines = text_content.split('\n')
            file_info["parsed_data"] = {
                "type": "text",
                "lines": lines[:100],
                "total_lines": len(lines)
            }
        
        return file_info
        
    except Exception as e:
        logger.error(f"Fehler beim Hochladen der Datei: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Verarbeiten der Datei: {str(e)}")


@app.get("/audio/models")
async def get_audio_models():
    """Gibt alle verfügbaren Audio-Modelle zurück"""
    if USE_MODEL_SERVICE:
        # Hole Status vom Model-Service
        status = model_service_client.get_audio_model_status()
        current_model = status.get("model_id") if status else None
        # Verfügbare Modelle kommen aus der Config (whisper_manager hat die Liste)
        return {
            "models": whisper_manager.get_available_models(),
            "current_model": current_model
        }
    else:
        # Lokale Manager
        return {
            "models": whisper_manager.get_available_models(),
            "current_model": whisper_manager.get_current_model()
        }


def _load_audio_model_async(model_id: str):
    """Lädt ein Audio-Modell im Hintergrund"""
    try:
        loading_status["audio_model"]["loading"] = True
        loading_status["audio_model"]["model_id"] = model_id
        loading_status["audio_model"]["error"] = None
        
        success = whisper_manager.load_model(model_id)
        if not success:
            loading_status["audio_model"]["error"] = f"Fehler beim Laden des Audio-Modells: {model_id}"
        else:
            loading_status["audio_model"]["error"] = None
    except Exception as e:
        loading_status["audio_model"]["error"] = str(e)
        logger.error(f"Fehler beim asynchronen Laden des Audio-Modells: {e}")
    finally:
        loading_status["audio_model"]["loading"] = False

@app.post("/audio/models/load")
async def load_audio_model(request: LoadModelRequest):
    """Lädt ein Audio-Modell im Hintergrund (blockiert UI nicht)"""
    if USE_MODEL_SERVICE:
        # Lade über Model-Service
        if not model_service_client.load_audio_model(request.model_id):
            raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Audio-Modells über Model-Service: {request.model_id}")
        return {
            "message": f"Audio-Modell wird geladen: {request.model_id}",
            "model_id": request.model_id,
            "status": "loading"
        }
    else:
        # Lade lokal
        # Prüfe ob bereits ein Modell geladen wird
        if loading_status["audio_model"]["loading"]:
            raise HTTPException(status_code=400, detail="Ein Audio-Modell wird bereits geladen. Bitte warten Sie.")
        
        # Starte Ladevorgang im Hintergrund
        loop = asyncio.get_event_loop()
        loop.run_in_executor(audio_load_executor, _load_audio_model_async, request.model_id)
        
        return {
            "message": f"Audio-Modell wird geladen: {request.model_id}",
            "model_id": request.model_id,
            "status": "loading"
        }

@app.get("/audio/models/load/status")
async def get_audio_model_load_status():
    """Gibt den Status des Audio-Modell-Ladevorgangs zurück"""
    return loading_status["audio_model"]


class TranscribeAudioResponse(BaseModel):
    text: str
    model_id: str


@app.post("/audio/transcribe", response_model=TranscribeAudioResponse)
async def transcribe_audio(file: UploadFile = File(...), language: Optional[str] = None):
    """
    Transkribiert Audio-Datei zu Text
    """
    
    # Prüfe ob Modell geladen ist (abhängig von USE_MODEL_SERVICE)
    model_loaded = False
    if USE_MODEL_SERVICE:
        # Prüfe Model-Service Status
        status = model_service_client.get_audio_model_status()
        model_loaded = status and status.get("loaded", False)
    else:
        # Prüfe lokalen Manager
        model_loaded = whisper_manager.is_model_loaded()
    
    if not model_loaded:
        
        
        # Prüfe ob bereits ein Modell geladen wird
        if loading_status["audio_model"]["loading"]:
            
            raise HTTPException(
                status_code=202,  # Accepted - wird verarbeitet
                detail={
                    "status": "model_loading",
                    "message": f"Audio-Modell wird geladen. Bitte warten Sie.",
                    "model_id": loading_status["audio_model"]["model_id"]
                }
            )
        
        # Lade Default-Audio-Modell
        if USE_MODEL_SERVICE:
            # Lade über Model-Service
            # Hole verfügbare Modelle vom Model-Service oder aus Config
            available_models = whisper_manager.get_available_models()  # Nur für Modell-Liste
            if available_models:
                default_model_id = list(available_models.keys())[0]
                logger.info(f"Lade Audio-Modell über Model-Service: {default_model_id}")
                # Lade über Model-Service
                if not model_service_client.load_audio_model(default_model_id):
                    raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Audio-Modells über Model-Service: {default_model_id}")
                raise HTTPException(
                    status_code=202,  # Accepted - wird verarbeitet
                    detail={
                        "status": "model_loading",
                        "message": f"Audio-Modell {default_model_id} wird geladen. Bitte warten Sie.",
                        "model_id": default_model_id
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Kein Audio-Modell verfügbar")
        else:
            # Lade lokal
            available_models = whisper_manager.get_available_models()
            if available_models:
                default_model_id = list(available_models.keys())[0]
                logger.info(f"Starte asynchrones Laden des Default-Audio-Modells: {default_model_id}")
                
                # Starte asynchrones Laden
                loop = asyncio.get_event_loop()
                loop.run_in_executor(audio_load_executor, _load_audio_model_async, default_model_id)
                
                raise HTTPException(
                    status_code=202,  # Accepted - wird verarbeitet
                    detail={
                        "status": "model_loading",
                        "message": f"Audio-Modell {default_model_id} wird geladen. Bitte warten Sie.",
                        "model_id": default_model_id
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Kein Audio-Modell geladen und kein Modell verfügbar")
    
    try:
        # Lese Audio-Datei
        content = await file.read()
        
        # Speichere temporär als WAV
        import tempfile
        temp_dir = get_temp_directory()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Lade Audio-Datei (wavfile.read schließt die Datei automatisch)
            sample_rate, audio_data = wavfile.read(tmp_path)
            
            # Warte kurz, damit Datei sicher geschlossen ist (Windows-spezifisch)
            import time
            time.sleep(0.1)
            
            # Konvertiere zu float32 und normalisiere
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.float32:
                # Bereits float32, aber prüfe ob normalisiert (Werte zwischen -1 und 1)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            elif audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            else:
                # Fallback: Konvertiere zu float32
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            
            # Konvertiere zu Mono wenn Stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample auf 16kHz wenn nötig
            if sample_rate != 16000:
                from scipy import signal
                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
            
            # Transkribiere
            if USE_MODEL_SERVICE:
                # Nutze Model-Service
                import base64
                # Konvertiere Audio zu WAV und dann zu Base64
                from scipy.io import wavfile as wavfile_write
                import tempfile as tmpfile
                tmp_wav_path = None
                try:
                    temp_dir = get_temp_directory()
                    with tmpfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as tmp_wav:
                        tmp_wav_path = tmp_wav.name
                        wavfile_write.write(tmp_wav_path, 16000, (audio_data * 32768.0).astype(np.int16))
                    
                    # Warte kurz, damit Datei sicher geschlossen ist
                    time.sleep(0.1)
                    
                    # Lese Datei
                    with open(tmp_wav_path, 'rb') as f:
                        audio_bytes = f.read()
                finally:
                    # Lösche temporäre WAV-Datei mit Retry
                    if tmp_wav_path:
                        _safe_delete_file(tmp_wav_path)
                
                audio_base64 = base64.b64encode(audio_bytes).decode()
                result = model_service_client.transcribe(audio_base64, language=language)
                if result:
                    text = result.get("text", "")
                else:
                    raise HTTPException(status_code=500, detail="Fehler bei Transkription über Model-Service")
            else:
                # Fallback: Nutze lokale Manager
                text = whisper_manager.transcribe(audio_data, language=language)
            
            # Bestimme model_id
            if USE_MODEL_SERVICE:
                status = model_service_client.get_audio_model_status()
                model_id = status.get("model_id") if status else None
            else:
                model_id = whisper_manager.get_current_model()
            
            return TranscribeAudioResponse(
                text=text,
                model_id=model_id
            )
            
        finally:
            # Lösche temporäre Datei mit Retry-Mechanismus
            _safe_delete_file(tmp_path)
        
    except Exception as e:
        logger.error(f"Fehler bei der Audio-Transkription: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Audio-Transkription: {str(e)}")


@app.post("/image/generate", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generiert ein Bild basierend auf einem Prompt"""
    
    if not image_manager:
        raise HTTPException(status_code=503, detail="Bildgenerierung nicht verfügbar (diffusers/xformers nicht installiert)")
    
    # Bestimme welches Modell verwendet werden soll
    model_to_use = request.model_id
    
    if not model_to_use:
        if USE_MODEL_SERVICE:
            # Prüfe Model-Service Status
            status = model_service_client.get_image_model_status()
            if status and status.get("loaded"):
                model_to_use = status.get("model_id")
            else:
                # Lade Default-Bildmodell falls vorhanden
                if image_manager:
                    available_models = image_manager.get_available_models()
                    if available_models:
                        # Versuche zuerst default_model aus image_generation config zu verwenden
                        default_image_model = config.get("image_generation", {}).get("default_model")
                        if default_image_model and default_image_model in available_models:
                            model_to_use = default_image_model
                        else:
                            # Fallback: Erstes verfügbares Modell
                            model_to_use = list(available_models.keys())[0]
                    else:
                        raise HTTPException(status_code=400, detail="Kein Bildgenerierungsmodell geladen und kein Modell verfügbar")
                else:
                    raise HTTPException(status_code=400, detail="Kein Bildgenerierungsmodell verfügbar")
        else:
            # Fallback auf lokale Manager
            if image_manager and image_manager.is_model_loaded():
                model_to_use = image_manager.get_current_model()
            else:
                # Lade Default-Bildmodell falls vorhanden
                if image_manager:
                    available_models = image_manager.get_available_models()
                    if available_models:
                        # Versuche zuerst default_model aus image_generation config zu verwenden
                        default_image_model = config.get("image_generation", {}).get("default_model")
                        if default_image_model and default_image_model in available_models:
                            model_to_use = default_image_model
                        else:
                            # Fallback: Erstes verfügbares Modell
                            model_to_use = list(available_models.keys())[0]
                    else:
                        raise HTTPException(status_code=400, detail="Kein Bildgenerierungsmodell geladen und kein Modell verfügbar")
                else:
                    raise HTTPException(status_code=400, detail="Kein Bildgenerierungsmodell verfügbar")
    
    # Stelle sicher dass Modell geladen ist
    if USE_MODEL_SERVICE:
        # Nutze Model-Service
        status = model_service_client.get_image_model_status()
        if status and status.get("loaded") and status.get("model_id") == model_to_use:
            # Modell ist bereits geladen
            model_ready = True
        else:
            # Prüfe ob Modell bereits geladen wird (im Model Service)
            # Der Model Service prüft intern, ob das Modell bereits geladen ist
            # Modell muss geladen werden
            if not model_service_client.load_image_model(model_to_use):
                raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Bildmodells: {model_to_use}")
            model_ready = True
    else:
        # Fallback: Nutze lokale Manager
        model_ready = await ensure_image_model_loaded(model_to_use, request.conversation_id)
    
    # Wenn Modell noch nicht geladen ist, gib Status zurück
    if not model_ready:
        
        raise HTTPException(
            status_code=202,  # Accepted - wird verarbeitet
            detail={
                "status": "model_loading",
                "message": f"Bildmodell {model_to_use} wird geladen. Bitte warten Sie.",
                "model_id": model_to_use,
                "conversation_id": request.conversation_id
            }
        )
    
    try:
        
        
        # Prüfe dynamisch ob Model-Service verfügbar ist
        use_model_service = check_model_service_available()
        ## Generiere Bild
        if use_model_service:
            ## Nutze Model-Service
            result = model_service_client.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                aspect_ratio=request.aspect_ratio
            )
            if result:
                image_base64 = result.get("image_base64", "")
                model_id = result.get("model_id", model_to_use)
                actual_width = result.get("width", request.width)
                actual_height = result.get("height", request.height)
                auto_resized = result.get("auto_resized", False)
                cpu_offload_used = result.get("cpu_offload_used", False)
                
                # Speichere Bild automatisch (dekodiere Base64 zu PIL Image)
                try:
                    import base64
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    saved_path = save_generated_image(image, request.prompt)
                    if saved_path:
                        logger.info(f"Bild gespeichert: {saved_path}")
                except Exception as e:
                    logger.warning(f"Fehler beim Speichern des Bildes vom Model-Service: {e}")
                
                return GenerateImageResponse(
                    image_base64=image_base64,
                    model_id=model_id,
                    width=actual_width,
                    height=actual_height,
                    auto_resized=auto_resized,
                    cpu_offload_used=cpu_offload_used
                )
            else:
                raise HTTPException(status_code=500, detail="Fehler bei Bildgenerierung über Model-Service")
        else:
            # Fallback: Nutze lokale Manager
            if not image_manager:
                raise HTTPException(status_code=503, detail="Bildgenerierung nicht verfügbar")
            
            result = image_manager.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                aspect_ratio=request.aspect_ratio
            )
            
            if result is None or result.get("image") is None:
                raise HTTPException(status_code=500, detail="Bildgenerierung fehlgeschlagen")
            
            image = result["image"]
            actual_width = result.get("width", request.width)
            actual_height = result.get("height", request.height)
            auto_resized = result.get("auto_resized", False)
            cpu_offload_used = result.get("cpu_offload_used", False)
            
            # Konvertiere zu Base64
            image_base64 = image_manager.image_to_base64(image)
            model_id = image_manager.get_current_model()
            
            # Speichere Bild automatisch mit Timestamp und Prompt
            saved_path = save_generated_image(image, request.prompt)
            if saved_path:
                logger.info(f"Bild gespeichert: {saved_path}")
        
        
        
        # Speichere Bild in Conversation wenn conversation_id vorhanden
        if request.conversation_id:
            # Speichere User-Prompt
            conversation_manager.add_message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.prompt
            )
            # Speichere Bild als Message mit speziellem Format
            conversation = conversation_manager.get_conversation(request.conversation_id)
            if conversation:
                conversation["messages"].append({
                    "role": "assistant",
                    "content": "image",
                    "image_base64": image_base64,
                    "prompt": request.prompt,
                    "timestamp": datetime.now().isoformat()
                })
                conversation["updated_at"] = datetime.now().isoformat()
                conversation_manager._save_conversation(conversation)
        
        return GenerateImageResponse(
            image_base64=image_base64,
            model_id=model_id,
            width=actual_width,
            height=actual_height,
            auto_resized=auto_resized,
            cpu_offload_used=cpu_offload_used
        )
        
    except Exception as e:
        
        logger.error(f"Fehler bei der Bildgenerierung: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildgenerierung: {str(e)}")


# Model-Service Management Endpoints (für Frontend)

@app.get("/model-service/status")
async def get_model_service_status():
    """Gibt Status aller Modelle vom Model-Service zurück"""
    if not USE_MODEL_SERVICE:
        raise HTTPException(status_code=503, detail="Model-Service ist nicht verfügbar")
    
    status = model_service_client.get_status()
    if status:
        return status
    else:
        raise HTTPException(status_code=500, detail="Fehler beim Abrufen des Model-Service-Status")


@app.get("/model-service/models/{model_type}/status")
async def get_model_status(model_type: str):
    """Gibt Status eines bestimmten Modell-Typs zurück"""
    if not USE_MODEL_SERVICE:
        raise HTTPException(status_code=503, detail="Model-Service ist nicht verfügbar")
    
    if model_type == "text":
        status = model_service_client.get_text_model_status()
    elif model_type == "audio":
        status = model_service_client.get_audio_model_status()
    elif model_type == "image":
        status = model_service_client.get_image_model_status()
    else:
        raise HTTPException(status_code=400, detail=f"Ungültiger Modell-Typ: {model_type}")
    
    if status:
        return status
    else:
        raise HTTPException(status_code=500, detail="Fehler beim Abrufen des Modell-Status")


class LoadModelRequestModelService(BaseModel):
    model_id: str


@app.post("/model-service/models/{model_type}/load")
async def load_model_via_service(model_type: str, request: LoadModelRequestModelService):
    """Lädt ein Modell über den Model-Service"""
    if not USE_MODEL_SERVICE:
        raise HTTPException(status_code=503, detail="Model-Service ist nicht verfügbar")
    
    success = False
    if model_type == "text":
        success = model_service_client.load_text_model(request.model_id)
    elif model_type == "audio":
        success = model_service_client.load_audio_model(request.model_id)
    elif model_type == "image":
        success = model_service_client.load_image_model(request.model_id)
    else:
        raise HTTPException(status_code=400, detail=f"Ungültiger Modell-Typ: {model_type}")
    
    if success:
        return {"status": "success", "message": f"Modell {request.model_id} wird geladen"}
    else:
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Modells: {request.model_id}")


@app.post("/model-service/models/{model_type}/unload")
async def unload_model_via_service(model_type: str):
    """Entlädt ein Modell über den Model-Service"""
    if not USE_MODEL_SERVICE:
        raise HTTPException(status_code=503, detail="Model-Service ist nicht verfügbar")
    
    success = False
    if model_type == "text":
        success = model_service_client.unload_text_model()
    elif model_type == "audio":
        success = model_service_client.unload_audio_model()
    elif model_type == "image":
        success = model_service_client.unload_image_model()
    else:
        raise HTTPException(status_code=400, detail=f"Ungültiger Modell-Typ: {model_type}")
    
    if success:
        return {"status": "success", "message": f"{model_type}-Modell wurde entladen"}
    else:
        raise HTTPException(status_code=500, detail=f"Fehler beim Entladen des {model_type}-Modells")


# Statische Dateien für Frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    
    
    import uvicorn
    
    
    
    try:
        
        
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        
        raise

