"""
Model Service - Separater Service für Model-Management
Läuft auf Port 8001 und hält Modelle im Speicher, auch wenn Local AI Server neu startet
"""
import json
import os
import sys
import logging
import time
import threading
import asyncio
import torch
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from logging_utils import get_logger
from contextlib import asynccontextmanager

# Setup logging - verwende StructuredLogger
logger = get_logger(__name__, log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "model_service.log"))

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import managers
from model_manager import ModelManager
from whisper_manager import WhisperManager

# ImageManager optional importieren
try:
    
    from image_manager import ImageManager
    
except Exception as e:
    
    logger.warning(f"ImageManager konnte nicht geladen werden: {e}")
    ImageManager = None

# Lifespan-Handler wird später definiert (nach allen benötigten Definitionen)
# App wird hier initialisiert, lifespan wird später gesetzt
def _create_lifespan():
    """Erstellt den lifespan-Handler (wird später aufgerufen)"""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Event Handler - lädt Qwen Text-Modell beim Start"""
        # Startup
        logger.model_load("Starte Model Service - Lade nur Qwen Text-Modell beim Start...")
        
        # FIX: Nur ein Modell beim Start laden (Qwen) - keine gleichzeitigen Ladevorgänge
        # Suche nach Qwen-Modell in Config
        qwen_model_id = None
        for model_id in model_manager.config.get("models", {}):
            if "qwen" in model_id.lower():
                qwen_model_id = model_id
                break
        
        # Fallback: Verwende zuletzt aktiviertes Text-Modell wenn kein Qwen gefunden
        if not qwen_model_id:
            last_active = load_last_active_models()
            if "text" in last_active and last_active["text"]:
                qwen_model_id = last_active["text"]
                logger.model_load(f"Kein Qwen-Modell in Config gefunden, verwende zuletzt aktiviertes: {qwen_model_id}")
        
        # Lade nur Text-Modell (Qwen oder Fallback)
        if qwen_model_id:
            logger.model_load(f"Lade Text-Modell beim Start: {qwen_model_id}")
            try:
                # Prüfe ob Modell in Config existiert
                if qwen_model_id in model_manager.config.get("models", {}):
                    # Starte asynchrones Laden
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(model_load_executor, _load_text_model_async, qwen_model_id)
                else:
                    logger.model_load(f"Modell {qwen_model_id} nicht in Config gefunden", level="warning")
            except Exception as e:
                logger.exception(f"Fehler beim Laden des Text-Modells {qwen_model_id}: {e}", tag="MODEL_LOAD")
        else:
            logger.model_load("Kein Modell zum Laden beim Start gefunden", level="warning")
        
        # Audio- und Image-Modelle werden NICHT beim Start geladen (nur bei Bedarf)
        logger.model_load("Audio- und Image-Modelle werden nicht beim Start geladen (nur bei Bedarf)")
        
        yield
        
        # Shutdown (falls nötig)
        logger.info("Model Service wird beendet...")
    
    return lifespan

# Initialisiere App (lifespan wird später gesetzt)
app = FastAPI(title="Model Service", version="1.0.0")

# CORS für alle Clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get config path
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

# Load config for temp directory
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
except:
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

def save_generated_image(image, prompt: str, temp_dir: Optional[str] = None) -> Optional[str]:
    """
    Speichert ein generiertes Bild mit Timestamp und Prompt im Dateinamen.
    
    Args:
        image: Das PIL Image
        prompt: Der Prompt-Text für den Dateinamen
        temp_dir: Optionales Temp-Verzeichnis (wird aus Config geladen wenn None)
        
    Returns:
        Pfad zur gespeicherten Datei oder None bei Fehler
    """
    try:
        from PIL import Image as PILImage
        from datetime import datetime
        
        if temp_dir is None:
            temp_dir = get_temp_directory()
        
        # Erstelle Unterordner für generierte Bilder
        images_dir = os.path.join(temp_dir, "generated_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Erstelle Dateinamen: timestamp_sanitized_prompt.png
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_prompt = sanitize_filename(prompt, max_length=50)
        filename = f"{timestamp}_{sanitized_prompt}.png"
        
        # Vollständiger Pfad
        file_path = os.path.join(images_dir, filename)
        
        # Speichere Bild
        image.save(file_path, format="PNG")
        logger.info(f"Bild gespeichert: {file_path}")
        return file_path
        
    except Exception as e:
        logger.warning(f"Fehler beim Speichern des Bildes: {e}")
        return None

# Initialize managers
model_manager = ModelManager(config_path=config_path)
whisper_manager = WhisperManager(config_path=config_path)

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


# Client tracking: {model_type: {model_id: [{client_id, app_name, timestamp, last_used}]}}
client_tracking: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "text": {},
    "audio": {},
    "image": {}
}

# Lock for thread-safe operations
tracking_lock = threading.Lock()

# Client cleanup interval (5 minutes)
CLIENT_CLEANUP_INTERVAL = 300  # seconds

# Thread Pool für asynchrones Modell-Laden
model_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_loader")

# Lade-Status für alle Modelle
loading_status = {
    "text": {"loading": False, "model_id": None, "error": None},
    "audio": {"loading": False, "model_id": None, "error": None},
    "image": {"loading": False, "model_id": None, "error": None}
}

# Pfad für zuletzt aktivierte Modelle
LAST_ACTIVE_MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "last_active_models.json")

def save_last_active_model(model_type: str, model_id: str):
    """Speichert das zuletzt aktivierte Modell"""
    try:
        os.makedirs(os.path.dirname(LAST_ACTIVE_MODELS_PATH), exist_ok=True)
        
        # Lade bestehende Daten
        last_active = {}
        if os.path.exists(LAST_ACTIVE_MODELS_PATH):
            try:
                with open(LAST_ACTIVE_MODELS_PATH, 'r', encoding='utf-8') as f:
                    last_active = json.load(f)
            except:
                pass
        
        # Aktualisiere für diesen Typ
        last_active[model_type] = model_id
        
        # Speichere
        with open(LAST_ACTIVE_MODELS_PATH, 'w', encoding='utf-8') as f:
            json.dump(last_active, f, indent=2)
        
        logger.info(f"Zuletzt aktiviertes {model_type}-Modell gespeichert: {model_id}")
    except Exception as e:
        logger.warning(f"Fehler beim Speichern des zuletzt aktivierten Modells: {e}")

def load_last_active_models() -> Dict[str, Optional[str]]:
    """Lädt die zuletzt aktivierten Modelle"""
    try:
        if os.path.exists(LAST_ACTIVE_MODELS_PATH):
            with open(LAST_ACTIVE_MODELS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Fehler beim Laden der zuletzt aktivierten Modelle: {e}")
    return {}


def register_client(model_type: str, model_id: str, client_id: str, app_name: str):
    """Registriert einen Client für ein Modell"""
    with tracking_lock:
        if model_id not in client_tracking[model_type]:
            client_tracking[model_type][model_id] = []
        
        # Prüfe ob Client bereits registriert ist
        for client in client_tracking[model_type][model_id]:
            if client["client_id"] == client_id:
                client["last_used"] = time.time()
                return
        
        # Neuer Client
        client_tracking[model_type][model_id].append({
            "client_id": client_id,
            "app_name": app_name,
            "timestamp": time.time(),
            "last_used": time.time()
        })
        logger.info(f"Client registriert: {client_id} ({app_name}) für {model_type}/{model_id}")


def unregister_client(model_type: str, model_id: str, client_id: str):
    """Entfernt einen Client"""
    with tracking_lock:
        if model_id in client_tracking[model_type]:
            client_tracking[model_type][model_id] = [
                c for c in client_tracking[model_type][model_id]
                if c["client_id"] != client_id
            ]
            logger.info(f"Client entfernt: {client_id} von {model_type}/{model_id}")


def cleanup_inactive_clients():
    """Entfernt inaktive Clients (älter als CLIENT_CLEANUP_INTERVAL)"""
    current_time = time.time()
    with tracking_lock:
        for model_type in client_tracking:
            for model_id in list(client_tracking[model_type].keys()):
                client_tracking[model_type][model_id] = [
                    c for c in client_tracking[model_type][model_id]
                    if (current_time - c["last_used"]) < CLIENT_CLEANUP_INTERVAL
                ]
                # Entferne leere Einträge
                if not client_tracking[model_type][model_id]:
                    del client_tracking[model_type][model_id]


def get_active_clients(model_type: str, model_id: str) -> List[Dict[str, Any]]:
    """Gibt aktive Clients für ein Modell zurück"""
    with tracking_lock:
        clients = client_tracking[model_type].get(model_id, [])
        current_time = time.time()
        return [
            {
                "client_id": c["client_id"],
                "app_name": c["app_name"],
                "active_since": int(current_time - c["timestamp"]),
                "last_used": int(current_time - c["last_used"])
            }
            for c in clients
            if (current_time - c["last_used"]) < CLIENT_CLEANUP_INTERVAL
        ]


# Background task für Cleanup
def cleanup_task():
    """Background-Task für Client-Cleanup"""
    while True:
        time.sleep(30)  # Alle 30 Sekunden prüfen
        try:
            cleanup_inactive_clients()
        except Exception as e:
            logger.error(f"Fehler beim Client-Cleanup: {e}")


# Start cleanup task
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()


# Request Models
class LoadModelRequest(BaseModel):
    model_id: str


class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[Dict[str, str]]] = None  # Vollständige Messages-Liste (mit System-Prompt und History)
    conversation_id: Optional[str] = None
    max_length: int = 2048
    temperature: float = 0.3  # Default: 0.3 für bessere Qualität (konsistent mit Frontend)
    language: Optional[str] = None  # Sprache für Antwort (z.B. "de", "en")


class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    language: Optional[str] = None


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    aspect_ratio: Optional[str] = None  # z.B. "16:9", "custom:2.5:1"


# Helper function to get client info from headers
def get_client_info(request: Request) -> tuple[str, str]:
    """Extrahiert Client-ID und App-Name aus Headers"""
    client_id = request.headers.get("X-Client-ID", f"unknown-{int(time.time())}")
    app_name = request.headers.get("X-App-Name", "Unknown App")
    return client_id, app_name

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


# API Endpoints

# Root-Endpoint wird später überschrieben wenn model_manager_dir existiert


@app.get("/status")
async def get_status():
    """Gibt Status aller Modelle zurück (inkl. Memory-Informationen)"""
    import torch
    
    # Text-Modell Status mit Memory-Info
    text_status = {
        "loaded": model_manager.is_model_loaded(),
        "model_id": model_manager.get_current_model(),
        "loading": loading_status["text"]["loading"],
        "error": loading_status["text"]["error"],
        "active_clients": get_active_clients("text", model_manager.get_current_model() or "") if model_manager.is_model_loaded() else []
    }
    
    # Memory-Informationen für Text-Modell
    if text_status["loaded"] and torch.cuda.is_available():
        try:
            # GPU Memory Usage
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_memory_free = gpu_memory_total - gpu_memory_reserved
            
            text_status["memory"] = {
                "gpu_allocated_gb": round(gpu_memory_allocated, 2),
                "gpu_reserved_gb": round(gpu_memory_reserved, 2),
                "gpu_total_gb": round(gpu_memory_total, 2),
                "gpu_free_gb": round(gpu_memory_free, 2),
                "gpu_usage_percent": round((gpu_memory_reserved / gpu_memory_total) * 100, 1) if gpu_memory_total > 0 else 0
            }
        except Exception as e:
            logger.model_load(f"Fehler beim Ermitteln der GPU-Memory-Info: {e}", level="warning")
            text_status["memory"] = None
    
    # Audio-Modell Status
    audio_status = {
        "loaded": whisper_manager.is_model_loaded(),
        "model_id": whisper_manager.get_current_model(),
        "loading": loading_status["audio"]["loading"],
        "error": loading_status["audio"]["error"],
        "active_clients": get_active_clients("audio", whisper_manager.get_current_model() or "") if whisper_manager.is_model_loaded() else []
    }
    
    # Image-Modell Status
    image_status = {
        "loaded": image_manager.is_model_loaded() if image_manager else False,
        "model_id": image_manager.get_current_model() if image_manager else None,
        "loading": loading_status["image"]["loading"],
        "error": loading_status["image"]["error"],
        "active_clients": get_active_clients("image", image_manager.get_current_model() or "") if (image_manager and image_manager.is_model_loaded()) else []
    }
    
    return {
        "text_model": text_status,
        "audio_model": audio_status,
        "image_model": image_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health-Check für alle geladenen Modelle"""
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


# Text Model Endpoints

def _load_text_model_async(model_id: str):
    """Lädt ein Text-Modell asynchron"""
    try:
        logger.model_load(f"Starte asynchrones Laden: {model_id}")
        logger.model_load(f"Thread: {threading.current_thread().name}, PID: {os.getpid()}")
        
        loading_status["text"]["loading"] = True
        loading_status["text"]["model_id"] = model_id
        loading_status["text"]["error"] = None
        
        # Prüfe ob Modell in Config existiert
        if model_id not in model_manager.config.get("models", {}):
            error_msg = f"Modell {model_id} nicht in Config gefunden"
            logger.error_log(error_msg)
            loading_status["text"]["loading"] = False
            loading_status["text"]["error"] = error_msg
            return
        
        model_info = model_manager.config["models"][model_id]
        model_path = model_info.get("path", "")
        logger.model_load(f"Modell-Pfad: {model_path}")
        
        # Prüfe ob Pfad existiert
        if not os.path.exists(model_path):
            error_msg = f"Modell-Pfad existiert nicht: {model_path}"
            logger.error_log(error_msg)
            loading_status["text"]["loading"] = False
            loading_status["text"]["error"] = error_msg
            return
        
        logger.model_load("Rufe model_manager.load_model() auf...")
        start_time = time.time()
        success = model_manager.load_model(model_id)
        load_time = time.time() - start_time
        logger.model_load(f"model_manager.load_model() zurückgegeben: success={success}, Dauer: {load_time:.2f}s")
        
        if success:
            # Prüfe ob Modell wirklich geladen ist
            is_loaded = model_manager.is_model_loaded()
            current_model = model_manager.get_current_model()
            logger.model_load(f"Status-Prüfung: is_loaded={is_loaded}, current_model={current_model}, erwartet={model_id}")
            
            if is_loaded and current_model == model_id:
                logger.model_load(f"✓ Text-Modell erfolgreich geladen: {model_id}")
                loading_status["text"]["loading"] = False
                loading_status["text"]["error"] = None
            else:
                error_msg = f"Status-Prüfung fehlgeschlagen: is_loaded={is_loaded}, current_model={current_model}, erwartet={model_id}"
                logger.error_log(error_msg)
                loading_status["text"]["loading"] = False
                loading_status["text"]["error"] = error_msg
        else:
            error_msg = f"model_manager.load_model() gab False zurück für Modell: {model_id}"
            logger.error_log(error_msg)
            logger.model_load(f"model_manager Status: is_loaded={model_manager.is_model_loaded()}, current_model={model_manager.get_current_model()}")
            loading_status["text"]["loading"] = False
            loading_status["text"]["error"] = error_msg
    except Exception as e:
        error_msg = f"Exception beim Laden des Text-Modells {model_id}: {str(e)}"
        logger.exception(error_msg, tag="MODEL_LOAD")
        loading_status["text"]["loading"] = False
        loading_status["text"]["error"] = error_msg

@app.post("/models/text/load")
async def load_text_model(request: LoadModelRequest):
    """Lädt ein Text-Modell (nur ein Modell gleichzeitig, automatisches Entladen)"""
    model_id = request.model_id
    
    # FIX: Prüfe ob bereits ein Modell geladen wird (verhindere gleichzeitige Ladevorgänge)
    # WICHTIG: Prüfe loading_status BEVOR wir es setzen (Race Condition vermeiden)
    if loading_status["text"]["loading"]:
        current_loading = loading_status["text"]["model_id"]
        logger.model_load(f"Ladevorgang bereits aktiv: {current_loading}, warte auf Abschluss...")
        return {
            "status": "loading",
            "message": f"Modell wird bereits geladen: {current_loading}. Bitte warten Sie.",
            "model_id": current_loading
        }
    
    # Prüfe ob Modell bereits geladen ist
    if model_manager.is_model_loaded() and model_manager.get_current_model() == model_id:
        logger.model_load(f"Modell bereits geladen: {model_id}")
        return {
            "status": "success",
            "model_id": model_id,
            "message": "Modell ist bereits geladen"
        }
    
    # FIX: Setze loading_status SOFORT um Race Conditions zu vermeiden
    loading_status["text"]["loading"] = True
    loading_status["text"]["model_id"] = model_id
    loading_status["text"]["error"] = None
    
    # FIX: Wenn anderes Modell geladen ist, entlade es zuerst
    if model_manager.is_model_loaded():
        current_model = model_manager.get_current_model()
        if current_model != model_id:
            logger.model_load(f"Entlade aktuelles Modell {current_model} vor Laden von {model_id}...")
            model_manager.unload_model()
            # Warte kurz damit Entladen abgeschlossen ist
            import time
            time.sleep(1)
    
    # Speichere als zuletzt aktiviertes Modell
    save_last_active_model("text", model_id)
    
    # Starte asynchrones Laden (nur ein Modell gleichzeitig)
    logger.model_load(f"Starte Laden von Text-Modell: {model_id}")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(model_load_executor, _load_text_model_async, model_id)
    
    return {
        "status": "loading",
        "message": f"Modell {model_id} wird geladen",
        "model_id": model_id
    }


@app.post("/models/text/unload")
async def unload_text_model():
    """Entlädt das aktuelle Text-Modell"""
    try:
        if model_manager.is_model_loaded():
            model_id = model_manager.get_current_model()
            # Entlade Modell
            if model_manager.model is not None:
                del model_manager.model
                del model_manager.tokenizer
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                model_manager.model = None
                model_manager.tokenizer = None
                model_manager.current_model_id = None
            
            # Entferne alle Clients
            with tracking_lock:
                if model_id in client_tracking["text"]:
                    del client_tracking["text"][model_id]
            
            return {"status": "success", "message": "Modell entladen"}
        else:
            return {"status": "success", "message": "Kein Modell geladen"}
    except Exception as e:
        logger.error(f"Fehler beim Entladen des Text-Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/text/status")
async def get_text_model_status():
    """Gibt Status des Text-Modells zurück"""
    model_id = model_manager.get_current_model()
    return {
        "loaded": model_manager.is_model_loaded(),
        "model_id": model_id,
        "active_clients": get_active_clients("text", model_id or "") if model_id else []
    }


# Audio Model Endpoints

@app.post("/models/audio/load")
async def load_audio_model(request: LoadModelRequest):
    """Lädt ein Audio-Modell"""
    try:
        # Prüfe ob bereits ein Modell geladen wird
        if loading_status["audio"]["loading"]:
            return {
                "status": "loading",
                "message": f"Modell {loading_status['audio']['model_id']} wird bereits geladen",
                "model_id": loading_status["audio"]["model_id"]
            }
        
        # Prüfe ob Modell bereits geladen ist
        if whisper_manager.is_model_loaded() and whisper_manager.get_current_model() == request.model_id:
            return {
                "status": "success",
                "model_id": request.model_id,
                "message": "Modell ist bereits geladen"
            }
        
        # Speichere als zuletzt aktiviertes Modell
        save_last_active_model("audio", request.model_id)
        
        success = whisper_manager.load_model(request.model_id)
        if success:
            return {"status": "success", "model_id": request.model_id, "message": "Modell geladen"}
        else:
            raise HTTPException(status_code=400, detail=f"Fehler beim Laden des Modells: {request.model_id}")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Audio-Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/audio/unload")
async def unload_audio_model():
    """Entlädt das aktuelle Audio-Modell"""
    try:
        if whisper_manager.is_model_loaded():
            model_id = whisper_manager.get_current_model()
            # Entlade Modell
            if whisper_manager.model is not None:
                del whisper_manager.model
                del whisper_manager.processor
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                whisper_manager.model = None
                whisper_manager.processor = None
                whisper_manager.current_model_id = None
            
            # Entferne alle Clients
            with tracking_lock:
                if model_id in client_tracking["audio"]:
                    del client_tracking["audio"][model_id]
            
            return {"status": "success", "message": "Modell entladen"}
        else:
            return {"status": "success", "message": "Kein Modell geladen"}
    except Exception as e:
        logger.error(f"Fehler beim Entladen des Audio-Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/audio/status")
async def get_audio_model_status():
    """Gibt Status des Audio-Modells zurück"""
    model_id = whisper_manager.get_current_model()
    return {
        "loaded": whisper_manager.is_model_loaded(),
        "model_id": model_id,
        "active_clients": get_active_clients("audio", model_id or "") if model_id else []
    }


# Image Model Endpoints

def _load_image_model_async(model_id: str):
    """Lädt ein Image-Modell asynchron"""
    try:
        
        
        loading_status["image"]["loading"] = True
        loading_status["image"]["model_id"] = model_id
        loading_status["image"]["error"] = None
        
        logger.info(f"Starte asynchrones Laden des Image-Modells: {model_id}")
        logger.info(f"Thread: {threading.current_thread().name}, PID: {os.getpid()}")
        
        # Wrapper um load_model, um sicherzustellen dass Exceptions gefangen werden
        try:
            
            
            success = image_manager.load_model(model_id)
            
            
        except KeyboardInterrupt:
            # KeyboardInterrupt sollte nicht im Thread auftreten, aber sicherheitshalber
            logger.warning("KeyboardInterrupt während Modell-Laden - wird ignoriert")
            success = False
        except SystemExit:
            # SystemExit sollte nicht den Server beenden
            logger.error("SystemExit während Modell-Laden - wird als Fehler behandelt")
            success = False
        except BaseException as e:
            # Fange alle anderen Exceptions (inkl. SystemExit, KeyboardInterrupt)
            
            logger.error(f"Kritischer Fehler beim Laden des Modells: {e}")
            import traceback
            logger.error(traceback.format_exc())
            success = False
        
        if success:
            # Detaillierte Status-Prüfung
            is_loaded = image_manager.is_model_loaded()
            current_model = image_manager.get_current_model()
            
            logger.info(f"Status-Prüfung nach Laden: is_loaded={is_loaded}, current_model={current_model}, erwartet={model_id}")
            
            # Prüfe Pipeline-Komponenten einzeln
            pipeline_status = {}
            if image_manager.pipeline is not None:
                pipeline_status["pipeline_exists"] = True
                pipeline_status["has_unet"] = hasattr(image_manager.pipeline, 'unet') and image_manager.pipeline.unet is not None
                pipeline_status["has_vae"] = hasattr(image_manager.pipeline, 'vae') and image_manager.pipeline.vae is not None
                pipeline_status["has_text_encoder"] = hasattr(image_manager.pipeline, 'text_encoder') and image_manager.pipeline.text_encoder is not None
                
                # Prüfe Device
                try:
                    if hasattr(image_manager.pipeline, 'unet') and image_manager.pipeline.unet is not None:
                        device_check = next(image_manager.pipeline.unet.parameters()).device
                        pipeline_status["device"] = str(device_check)
                except Exception as e:
                    pipeline_status["device"] = f"Fehler: {e}"
            else:
                pipeline_status["pipeline_exists"] = False
            
            logger.info(f"Pipeline-Status: {pipeline_status}")
            
            # Prüfe ob Modell wirklich geladen ist
            if is_loaded and current_model == model_id:
                logger.info(f"Image-Modell erfolgreich geladen: {model_id}")
                loading_status["image"]["loading"] = False
                loading_status["image"]["error"] = None
            else:
                error_details = []
                if not is_loaded:
                    error_details.append("is_model_loaded() = False")
                if current_model != model_id:
                    error_details.append(f"current_model ({current_model}) != erwartet ({model_id})")
                
                error_msg = f"Modell wurde geladen, aber Status-Prüfung fehlgeschlagen: {', '.join(error_details)}"
                logger.error(error_msg)
                logger.error(f"Pipeline-Status: {pipeline_status}")
                loading_status["image"]["loading"] = False
                loading_status["image"]["error"] = error_msg
                
                # Setze Status zurück bei Fehler
                if image_manager:
                    image_manager.current_model_id = None
                    image_manager.pipeline = None
        else:
            error_msg = f"Fehler beim Laden des Modells: {model_id}"
            logger.error(error_msg)
            loading_status["image"]["loading"] = False
            loading_status["image"]["error"] = error_msg
            
            # Setze Status zurück bei Fehler
            if image_manager:
                image_manager.current_model_id = None
                image_manager.pipeline = None
    except BaseException as e:
        # Fange alle Exceptions (inkl. SystemExit, KeyboardInterrupt) um Server zu schützen
        
        error_msg = f"Exception beim Laden des Image-Modells: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        loading_status["image"]["loading"] = False
        loading_status["image"]["error"] = error_msg
        # Stelle sicher dass current_model_id zurückgesetzt wird
        if image_manager:
            try:
                image_manager.current_model_id = None
                image_manager.pipeline = None
            except Exception as cleanup_error:
                logger.warning(f"Fehler beim Cleanup nach Modell-Lade-Fehler: {cleanup_error}")
        
        # Re-raise nur wenn es kein SystemExit oder KeyboardInterrupt ist
        # (diese sollten den Server nicht beenden)
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            logger.warning(f"{type(e).__name__} während Modell-Laden - wird ignoriert um Server zu schützen")
        # Andere Exceptions werden bereits geloggt, aber nicht weitergeworfen
        # um den Server nicht zum Absturz zu bringen

@app.post("/models/image/load")
async def load_image_model(request: LoadModelRequest):
    """Lädt ein Image-Modell (asynchron)"""
    
    try:
        if not image_manager:
            error_msg = "ImageManager nicht verfügbar. "
            if not ImageManager:
                error_msg += "Der ImageManager konnte nicht importiert werden. Bitte prüfen Sie, ob 'diffusers' installiert ist: pip install diffusers"
            else:
                error_msg += "Die Initialisierung des ImageManagers ist fehlgeschlagen. Bitte prüfen Sie die Logs für Details."
            raise HTTPException(status_code=503, detail=error_msg)
        
        # Prüfe ob bereits ein Modell geladen wird
        if loading_status["image"]["loading"]:
            return {
                "status": "loading",
                "message": f"Modell {loading_status['image']['model_id']} wird bereits geladen",
                "model_id": loading_status["image"]["model_id"]
            }
        
        # Prüfe ob Modell bereits geladen ist
        if image_manager.is_model_loaded() and image_manager.get_current_model() == request.model_id:
            return {
                "status": "success",
                "model_id": request.model_id,
                "message": "Modell ist bereits geladen"
            }
        
        # Speichere als zuletzt aktiviertes Modell
        save_last_active_model("image", request.model_id)
        
        # Starte asynchrones Laden
        loop = asyncio.get_event_loop()
        loop.run_in_executor(model_load_executor, _load_image_model_async, request.model_id)
        
        return {
            "status": "loading",
            "message": f"Modell {request.model_id} wird geladen",
            "model_id": request.model_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unerwarteter Fehler in load_image_model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Fehler beim Starten des Modell-Ladens: {str(e)}")


@app.post("/models/image/unload")
async def unload_image_model():
    """Entlädt das aktuelle Image-Modell"""
    if not image_manager:
        error_msg = "ImageManager nicht verfügbar. "
        if not ImageManager:
            error_msg += "Der ImageManager konnte nicht importiert werden. Bitte prüfen Sie, ob 'diffusers' installiert ist: pip install diffusers"
        else:
            error_msg += "Die Initialisierung des ImageManagers ist fehlgeschlagen. Bitte prüfen Sie die Logs für Details."
        raise HTTPException(status_code=503, detail=error_msg)
    try:
        if image_manager.is_model_loaded():
            model_id = image_manager.get_current_model()
            # Entlade Modell
            if image_manager.pipeline is not None:
                del image_manager.pipeline
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                image_manager.pipeline = None
                image_manager.current_model_id = None
            
            # Entferne alle Clients
            with tracking_lock:
                if model_id in client_tracking["image"]:
                    del client_tracking["image"][model_id]
            
            return {"status": "success", "message": "Modell entladen"}
        else:
            return {"status": "success", "message": "Kein Modell geladen"}
    except Exception as e:
        logger.error(f"Fehler beim Entladen des Image-Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/image/status")
async def get_image_model_status():
    """Gibt Status des Image-Modells zurück"""
    if not image_manager:
        return {
            "loaded": False,
            "model_id": None,
            "loading": loading_status["image"]["loading"],
            "error": loading_status["image"]["error"],
            "active_clients": []
        }
    model_id = image_manager.get_current_model()
    return {
        "loaded": image_manager.is_model_loaded(),
        "model_id": model_id,
        "loading": loading_status["image"]["loading"],
        "error": loading_status["image"]["error"],
        "active_clients": get_active_clients("image", model_id or "") if model_id else []
    }


# Chat Endpoint (delegiert an Text-Modell)

@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    """Chat-Request - delegiert an geladenes Text-Modell"""
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=400, detail="Kein Text-Modell geladen")
    
    client_id, app_name = get_client_info(http_request)
    model_id = model_manager.get_current_model()
    
    # Registriere Client
    register_client("text", model_id, client_id, app_name)
    
    try:
        # Nutze die generate-Methode des ModelManagers
        # Wenn vollständige Messages übergeben wurden, nutze diese, sonst nur die einzelne Message
        if request.messages:
            messages = request.messages
        else:
            messages = [{"role": "user", "content": request.message}]
        
        # Stelle sicher, dass System-Prompt vorhanden ist (wenn nicht bereits in messages)
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            # Bestimme Antwort-Sprache
            response_language = request.language or "de"  # Default: Deutsch
            
            # Generiere System-Prompt basierend auf Sprache und Modell
            if "mistral" in model_id.lower():
                if response_language == "en":
                    system_prompt = "You are a helpful AI assistant. Answer briefly, precisely and directly in English. Keep answers under 200 words. Answer ONLY the asked question, no additional explanations or technical details."
                else:  # Deutsch (de) oder andere
                    system_prompt = "Du bist ein hilfreicher AI-Assistent. Antworte kurz, präzise und direkt auf Deutsch. Halte Antworten unter 200 Wörtern. Antworte NUR auf die gestellte Frage, keine zusätzlichen Erklärungen oder technischen Details."
            else:
                if response_language == "en":
                    system_prompt = "You are a helpful, precise and friendly AI assistant. Answer clearly and directly in English. IMPORTANT: Answer ONLY with your response, do NOT repeat the system prompt or user messages."
                else:  # Deutsch (de) oder andere
                    system_prompt = "Du bist ein hilfreicher, präziser und freundlicher AI-Assistent. Antworte klar und direkt auf Deutsch. WICHTIG: Antworte NUR mit deiner Antwort, wiederhole NICHT den System-Prompt oder User-Nachrichten."
            
            # Füge System-Prompt am Anfang hinzu
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # FIX: Synchroner generate() Call muss in Thread-Pool laufen um Event Loop nicht zu blockieren
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response_text = await loop.run_in_executor(
                executor,
                lambda: model_manager.generate(
                    messages,
                    max_length=request.max_length,
                    temperature=request.temperature if request.temperature > 0 else 0.3
                )
            )
        
        return {
            "response": response_text,
            "model_id": model_id,
            "conversation_id": request.conversation_id
        }
    except Exception as e:
        logger.error(f"Fehler bei Chat-Request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Transcribe Endpoint (delegiert an Audio-Modell)

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest, http_request: Request):
    """Audio-Transkription - delegiert an Whisper"""
    if not whisper_manager.is_model_loaded():
        raise HTTPException(status_code=400, detail="Kein Audio-Modell geladen")
    
    client_id, app_name = get_client_info(http_request)
    model_id = whisper_manager.get_current_model()
    
    # Registriere Client
    register_client("audio", model_id, client_id, app_name)
    
    try:
        # Dekodiere Base64 Audio
        if not request.audio_base64:
            raise HTTPException(status_code=400, detail="Kein Audio-Daten übergeben")
        
        import base64
        import numpy as np
        from scipy.io import wavfile
        import tempfile
        
        # Dekodiere Base64
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Speichere temporär als WAV
        temp_dir = get_temp_directory()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as tmp_file:
            tmp_file.write(audio_bytes)
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
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            elif audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            else:
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
            
            # Transkribiere mit WhisperManager
            text = whisper_manager.transcribe(audio_data, language=request.language)
            
            return {
                "text": text,
                "model_id": model_id
            }
        finally:
            # Lösche temporäre Datei mit Retry-Mechanismus
            _safe_delete_file(tmp_path)
    except Exception as e:
        logger.error(f"Fehler bei Audio-Transkription: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Generate Image Endpoint (delegiert an Image-Modell)

@app.post("/generate_image")
async def generate_image(request: GenerateImageRequest, http_request: Request):
    """Bildgenerierung - delegiert an Image-Modell"""
    if not image_manager:
        error_msg = "ImageManager nicht verfügbar. "
        if not ImageManager:
            error_msg += "Der ImageManager konnte nicht importiert werden. Bitte prüfen Sie, ob 'diffusers' installiert ist: pip install diffusers"
        else:
            error_msg += "Die Initialisierung des ImageManagers ist fehlgeschlagen. Bitte prüfen Sie die Logs für Details."
        raise HTTPException(status_code=503, detail=error_msg)
    
    
    if not image_manager.is_model_loaded():
        # Prüfe ob Modell gerade geladen wird
        if loading_status["image"]["loading"]:
            raise HTTPException(status_code=202, detail=f"Modell wird noch geladen: {loading_status['image']['model_id']}")
        raise HTTPException(status_code=400, detail="Kein Image-Modell geladen")
    
    client_id, app_name = get_client_info(http_request)
    model_id = image_manager.get_current_model()
    
    # Registriere Client
    register_client("image", model_id, client_id, app_name)
    
    try:
        # Generiere Bild
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
        import io
        import base64
        from PIL import Image
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Speichere Bild automatisch mit Timestamp und Prompt
        saved_path = save_generated_image(image, request.prompt)
        if saved_path:
            logger.info(f"Bild gespeichert: {saved_path}")
        
        return {
            "image_base64": img_base64,
            "model_id": model_id,
            "width": actual_width,
            "height": actual_height,
            "auto_resized": auto_resized,
            "cpu_offload_used": cpu_offload_used
        }
    except Exception as e:
        logger.error(f"Fehler bei Bildgenerierung: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statische Dateien für Model Manager Frontend
model_manager_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_manager")

# Mount statische Dateien für Model Manager Frontend (MUSS VOR Root-Endpoint sein!)
if os.path.exists(model_manager_dir):
    try:
        app.mount("/static", StaticFiles(directory=model_manager_dir), name="model_manager_static")
        logger.info(f"Statische Dateien gemountet von: {model_manager_dir}")
    except Exception as e:
        logger.error(f"Fehler beim Mounten der statischen Dateien: {e}")

# Root-Endpoint - zeigt Model Manager UI wenn verfügbar
@app.get("/")
async def root():
    # Prüfe ob Model Manager UI existiert
    if os.path.exists(model_manager_dir):
        index_path = os.path.join(model_manager_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"message": "Model Service API", "status": "running", "version": "1.0.0"}


# Lifespan Event Handler (ersetzt deprecated on_event)
# Muss nach allen Definitionen (model_manager, load_last_active_models, _load_text_model_async) stehen
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan Event Handler - lädt Qwen Text-Modell beim Start"""
    # Startup
    logger.model_load("Starte Model Service - Lade nur Qwen Text-Modell beim Start...")
    
    # FIX: Nur ein Modell beim Start laden (Qwen) - keine gleichzeitigen Ladevorgänge
    # Suche nach Qwen-Modell in Config
    qwen_model_id = None
    for model_id in model_manager.config.get("models", {}):
        if "qwen" in model_id.lower():
            qwen_model_id = model_id
            break
    
    # Fallback: Verwende zuletzt aktiviertes Text-Modell wenn kein Qwen gefunden
    if not qwen_model_id:
        last_active = load_last_active_models()
        if "text" in last_active and last_active["text"]:
            qwen_model_id = last_active["text"]
            logger.model_load(f"Kein Qwen-Modell in Config gefunden, verwende zuletzt aktiviertes: {qwen_model_id}")
    
    # Lade nur Text-Modell (Qwen oder Fallback)
    if qwen_model_id:
        logger.model_load(f"Lade Text-Modell beim Start: {qwen_model_id}")
        try:
            # Prüfe ob Modell in Config existiert
            if qwen_model_id in model_manager.config.get("models", {}):
                # Starte asynchrones Laden
                loop = asyncio.get_event_loop()
                loop.run_in_executor(model_load_executor, _load_text_model_async, qwen_model_id)
            else:
                logger.model_load(f"Modell {qwen_model_id} nicht in Config gefunden", level="warning")
        except Exception as e:
            logger.exception(f"Fehler beim Laden des Text-Modells {qwen_model_id}: {e}", tag="MODEL_LOAD")
    else:
        logger.model_load("Kein Modell zum Laden beim Start gefunden", level="warning")
    
    # Audio- und Image-Modelle werden NICHT beim Start geladen (nur bei Bedarf)
    logger.model_load("Audio- und Image-Modelle werden nicht beim Start geladen (nur bei Bedarf)")
    
    yield
    
    # Shutdown (falls nötig)
    logger.info("Model Service wird beendet...")

# Setze lifespan für die bereits initialisierte App
# Die App wurde oben ohne lifespan initialisiert, jetzt setzen wir den Handler
app.router.lifespan_context = lifespan


# Restart Endpoint
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


if __name__ == "__main__":
    # Load config for port
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        port = config.get("model_service", {}).get("port", 8001)
        host = config.get("model_service", {}).get("host", "127.0.0.1")
    except:
        port = 8001
        host = "127.0.0.1"
    
    logger.info(f"Starte Model Service auf {host}:{port}")
    
    # Konfiguriere uvicorn für bessere Stabilität
    # timeout_keep_alive verhindert, dass Verbindungen während langem Laden abreißen
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=300,  # 5 Minuten - genug für Modell-Laden
        log_level="info"
    )

