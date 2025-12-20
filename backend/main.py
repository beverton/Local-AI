"""
FastAPI Server - Hauptserver für den lokalen AI-Dienst
"""
import json
import sys
import os
import logging
import subprocess

# #region agent log
log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cursor', 'debug.log')
try:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Use unbuffered mode to prevent terminal flicker
    with open(log_path, 'a', encoding='utf-8', buffering=0) as f:
        log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"main.py:10","message":"Script started","data":{"cwd":os.getcwd(),"script_path":__file__},"timestamp":int(__import__('time').time()*1000)}) + '\n'
        f.write(log_entry.encode('utf-8'))
except Exception:
    # Silent fail - don't print to console or cause flicker
    pass
# #endregion

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import csv
import io

# #region agent log
try:
    with open(log_path, 'ab') as f:
        log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"main.py:25","message":"FastAPI imports successful","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n'
        f.write(log_entry.encode('utf-8'))
except: pass
# #endregion

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# #region agent log
try:
    with open(log_path, 'ab') as f:
        log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"main.py:30","message":"Before local imports","data":{"sys_path":sys.path},"timestamp":int(__import__('time').time()*1000)}) + '\n'
        f.write(log_entry.encode('utf-8'))
except: pass
# #endregion

from model_manager import ModelManager
from conversation_manager import ConversationManager
from preference_learner import PreferenceLearner
from image_manager import ImageManager
from whisper_manager import WhisperManager
import psutil
import torch
import numpy as np
from scipy.io import wavfile

# #region agent log
try:
    with open(log_path, 'ab') as f:
        log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"main.py:35","message":"Local imports successful","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n'
        f.write(log_entry.encode('utf-8'))
except: pass
# #endregion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
model_manager = ModelManager(config_path=config_path)
conversation_manager = ConversationManager()
preference_learner = PreferenceLearner()
image_manager = ImageManager(config_path=config_path)
whisper_manager = WhisperManager(config_path=config_path)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.7


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
    model_id: Optional[str] = None


class GenerateImageResponse(BaseModel):
    image_base64: str
    model_id: str


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    model_id: Optional[str] = None


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
            stats["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 0)
            stats["gpu_memory_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 0)
            stats["gpu_memory_percent"] = round((stats["gpu_memory_used_mb"] / stats["gpu_memory_total_mb"]) * 100, 1) if stats["gpu_memory_total_mb"] > 0 else 0
            
            # GPU-Auslastung via nvidia-smi (falls verfügbar)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    stats["gpu_utilization"] = int(result.stdout.strip())
            except:
                pass
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


@app.post("/models/load")
async def load_model(request: LoadModelRequest):
    """Lädt ein Modell"""
    success = model_manager.load_model(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Fehler beim Laden des Modells: {request.model_id}")
    return {"message": f"Modell geladen: {request.model_id}", "model_id": request.model_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat-Endpunkt - verarbeitet eine Nachricht und gibt eine Antwort zurück
    """
    # Prüfe ob Modell geladen ist
    if not model_manager.is_model_loaded():
        # Lade Default-Modell
        default_model = model_manager.config.get("default_model")
        if default_model:
            logger.info(f"Lade Default-Modell: {default_model}")
            model_manager.load_model(default_model)
        else:
            raise HTTPException(status_code=400, detail="Kein Modell geladen und kein Default-Modell konfiguriert")
    
    # Conversation ID prüfen/erstellen
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = conversation_manager.create_conversation()
    
    # Lade Conversation History
    history = conversation_manager.get_conversation_history(conversation_id)
    
    # Erstelle Messages-Liste für Chat-Format
    messages = []
    
    # System-Prompt (mit Präferenzen wenn aktiviert)
    current_model = model_manager.get_current_model()
    if current_model and "phi-3" in current_model.lower():
        # Phi-3 verwendet ChatML-Format, kein separater System-Prompt nötig
        system_prompt = None
    elif current_model and "mistral" in current_model.lower():
        # Mistral: Kürzerer, präziserer Prompt
        system_prompt = "Du bist ein hilfreicher AI-Assistent. Antworte kurz, präzise und direkt auf Deutsch. Halte Antworten unter 200 Wörtern. Antworte NUR auf die gestellte Frage, keine zusätzlichen Erklärungen oder technischen Details."
        system_prompt = preference_learner.get_system_prompt(system_prompt)
    else:
        system_prompt = "Du bist ein hilfreicher, präziser und freundlicher AI-Assistent. Antworte klar und direkt auf Deutsch. WICHTIG: Antworte NUR mit deiner Antwort, wiederhole NICHT den System-Prompt oder User-Nachrichten. Generiere KEINE weiteren User- oder Assistant-Nachrichten."
        system_prompt = preference_learner.get_system_prompt(system_prompt)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Conversation History hinzufügen
    messages.extend(history)
    
    # Aktuelle Nachricht hinzufügen
    messages.append({"role": "user", "content": request.message})
    
    # Logge Messages (Memory 4468610)
    logger.info(f"Messages: {len(messages)} Nachrichten")
    
    # Generiere Antwort mit verbesserter Methode
    try:
        # Verwende niedrigere Temperature standardmäßig für bessere Qualität
        effective_temperature = request.temperature if request.temperature > 0 else 0.3
        
        response = model_manager.generate(
            messages,
            max_length=request.max_length,
            temperature=effective_temperature
        )
        
        # Logge Response
        logger.info(f"Response: {response[:200]}...")
        
        # FINALE BEREINIGUNG: Einfache Methode - finde "assistant" und nimm nur den Inhalt danach
        cleaned_response = response
        
        # Methode 1: Suche nach "assistant" Marker
        response_lower = cleaned_response.lower()
        assistant_markers = ["assistant ", "assistant:", "assistant\n"]
        assistant_pos = -1
        
        for marker in assistant_markers:
            pos = response_lower.find(marker)
            if pos != -1:
                assistant_pos = pos + len(marker)
                break
        
        if assistant_pos > 0:
            cleaned_response = cleaned_response[assistant_pos:].strip()
            logger.info("Response nach 'assistant' Marker extrahiert")
        else:
            # Methode 2: Zeilenweise durchgehen
            lines = cleaned_response.split('\n')
            final_lines = []
            found_assistant = False
            
            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Wenn wir "assistant" finden, nimm nur den Inhalt danach
                if line_lower.startswith('assistant'):
                    found_assistant = True
                    content = line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else line_stripped[9:].strip()
                    if content:
                        final_lines.append(content)
                    continue
                
                # Überspringe System/User-Zeilen
                if not found_assistant:
                    if (line_lower.startswith('system ') or 
                        line_lower.startswith('user ') or
                        "du bist ein hilfreicher" in line_lower):
                        continue
                
                # Nach "assistant": Füge alle Zeilen hinzu außer weitere Markierungen
                if found_assistant:
                    if not (line_lower.startswith('system ') or line_lower.startswith('user ')):
                        final_lines.append(line)
                else:
                    # Vor "assistant": Nur Zeilen ohne System/User-Markierungen
                    final_lines.append(line)
            
            cleaned_response = '\n'.join(final_lines).strip()
        
        # Entferne System-Prompt-Phrasen falls noch vorhanden
        system_keywords = ["du bist ein hilfreicher", "ai-assistent", "antworte klar und direkt"]
        for keyword in system_keywords:
            if keyword in cleaned_response.lower():
                lines = cleaned_response.split('\n')
                cleaned_response = '\n'.join([l for l in lines if keyword.lower() not in l.lower()]).strip()
        
        # Entferne User-Nachricht falls noch vorhanden
        if request.message in cleaned_response:
            cleaned_response = cleaned_response.replace(request.message, "").strip()
        
        # Entferne führende "assistant" falls noch vorhanden
        if cleaned_response.lower().startswith('assistant'):
            cleaned_response = cleaned_response.split(':', 1)[-1].strip() if ':' in cleaned_response else cleaned_response[9:].strip()
        
        # Speichere Nachrichten (nur bereinigte Response)
        conversation_manager.add_message(conversation_id, "user", request.message)
        conversation_manager.add_message(conversation_id, "assistant", cleaned_response)
        
        # Verwende bereinigte Response für Rückgabe
        response = cleaned_response
        
        # Lerne aus Conversation (wenn aktiviert)
        if preference_learner.is_enabled():
            all_messages = conversation_manager.get_conversation_history(conversation_id)
            preference_learner.learn_from_conversation(all_messages)
        
        return ChatResponse(response=response, conversation_id=conversation_id)
        
    except Exception as e:
        logger.error(f"Fehler bei der Generierung: {e}")
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
    """Erstellt eine neue Conversation"""
    conversation_id = conversation_manager.create_conversation()
    return {"conversation_id": conversation_id}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Löscht eine Conversation"""
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation nicht gefunden")
    return {"message": "Conversation gelöscht"}


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


@app.get("/image/models")
async def get_image_models():
    """Gibt alle verfügbaren Bildgenerierungsmodelle zurück"""
    return {
        "models": image_manager.get_available_models(),
        "current_model": image_manager.get_current_model()
    }


@app.post("/image/models/load")
async def load_image_model(request: LoadModelRequest):
    """Lädt ein Bildgenerierungsmodell"""
    success = image_manager.load_model(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Fehler beim Laden des Bildgenerierungsmodells: {request.model_id}")
    return {"message": f"Bildgenerierungsmodell geladen: {request.model_id}", "model_id": request.model_id}


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
    return {
        "models": whisper_manager.get_available_models(),
        "current_model": whisper_manager.get_current_model()
    }


@app.post("/audio/models/load")
async def load_audio_model(request: LoadModelRequest):
    """Lädt ein Audio-Modell"""
    success = whisper_manager.load_model(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Fehler beim Laden des Audio-Modells: {request.model_id}")
    return {"message": f"Audio-Modell geladen: {request.model_id}", "model_id": request.model_id}


class TranscribeAudioResponse(BaseModel):
    text: str
    model_id: str


@app.post("/audio/transcribe", response_model=TranscribeAudioResponse)
async def transcribe_audio(file: UploadFile = File(...), language: Optional[str] = None):
    """
    Transkribiert Audio-Datei zu Text
    """
    # Prüfe ob Modell geladen ist
    if not whisper_manager.is_model_loaded():
        # Lade Default-Audio-Modell
        available_models = whisper_manager.get_available_models()
        if available_models:
            default_model_id = list(available_models.keys())[0]
            logger.info(f"Lade Default-Audio-Modell: {default_model_id}")
            whisper_manager.load_model(default_model_id)
        else:
            raise HTTPException(status_code=400, detail="Kein Audio-Modell geladen und kein Modell verfügbar")
    
    try:
        # Lese Audio-Datei
        content = await file.read()
        
        # Speichere temporär als WAV
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Lade Audio-Datei
            sample_rate, audio_data = wavfile.read(tmp_path)
            
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
            text = whisper_manager.transcribe(audio_data, language=language)
            
            return TranscribeAudioResponse(
                text=text,
                model_id=whisper_manager.get_current_model()
            )
            
        finally:
            # Lösche temporäre Datei
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Fehler bei der Audio-Transkription: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Audio-Transkription: {str(e)}")


@app.post("/image/generate", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generiert ein Bild basierend auf einem Prompt"""
    # Lade Modell falls angegeben und noch nicht geladen
    if request.model_id:
        if image_manager.get_current_model() != request.model_id:
            success = image_manager.load_model(request.model_id)
            if not success:
                raise HTTPException(status_code=400, detail=f"Fehler beim Laden des Bildgenerierungsmodells: {request.model_id}")
    elif not image_manager.is_model_loaded():
        # Lade Default-Bildmodell falls vorhanden
        available_models = image_manager.get_available_models()
        if available_models:
            default_model_id = list(available_models.keys())[0]
            logger.info(f"Lade Default-Bildgenerierungsmodell: {default_model_id}")
            image_manager.load_model(default_model_id)
        else:
            raise HTTPException(status_code=400, detail="Kein Bildgenerierungsmodell geladen und kein Modell verfügbar")
    
    try:
        # Generiere Bild
        image = image_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height
        )
        
        if image is None:
            raise HTTPException(status_code=500, detail="Bildgenerierung fehlgeschlagen")
        
        # Konvertiere zu Base64
        image_base64 = image_manager.image_to_base64(image)
        
        return GenerateImageResponse(
            image_base64=image_base64,
            model_id=image_manager.get_current_model()
        )
        
    except Exception as e:
        logger.error(f"Fehler bei der Bildgenerierung: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildgenerierung: {str(e)}")


# Statische Dateien für Frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if __name__ == "__main__":
    # #region agent log
    try:
        with open(log_path, 'ab') as f:
            log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"main.py:230","message":"Before uvicorn import","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n'
            f.write(log_entry.encode('utf-8'))
    except: pass
    # #endregion
    
    import uvicorn
    
    # #region agent log
    try:
        with open(log_path, 'ab') as f:
            log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"main.py:235","message":"Before uvicorn.run","data":{"host":"127.0.0.1","port":8000},"timestamp":int(__import__('time').time()*1000)}) + '\n'
            f.write(log_entry.encode('utf-8'))
    except: pass
    # #endregion
    
    try:
        # #region agent log
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            port_check = sock.connect_ex(('127.0.0.1', 8000))
            sock.close()
            port_available = port_check != 0
            with open(log_path, 'ab') as f:
                log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"main.py:245","message":"Port 8000 check before start","data":{"port_available":port_available,"port_check_result":port_check},"timestamp":int(__import__('time').time()*1000)}) + '\n'
                f.write(log_entry.encode('utf-8'))
            if not port_available:
                # Port ist belegt - versuche Prozess zu finden
                try:
                    import subprocess
                    result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, timeout=2)
                    for line in result.stdout.split('\n'):
                        if ':8000' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) > 4:
                                pid = parts[-1]
                                with open(log_path, 'ab') as f:
                                    log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"main.py:255","message":"Port 8000 occupied by PID","data":{"pid":pid},"timestamp":int(__import__('time').time()*1000)}) + '\n'
                                    f.write(log_entry.encode('utf-8'))
                except: pass
        except Exception as port_e:
            try:
                with open(log_path, 'ab') as f:
                    log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"main.py:260","message":"Port check failed","data":{"error":str(port_e)},"timestamp":int(__import__('time').time()*1000)}) + '\n'
                    f.write(log_entry.encode('utf-8'))
            except: pass
        # #endregion
        
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        # #region agent log
        try:
            with open(log_path, 'ab') as f:
                log_entry = json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"main.py:255","message":"uvicorn.run exception","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n'
                f.write(log_entry.encode('utf-8'))
        except: pass
        # #endregion
        raise

