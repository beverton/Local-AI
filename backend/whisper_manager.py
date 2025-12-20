"""
Whisper Manager - Verwaltet das Laden und Transkribieren von Audio mit Whisper
"""
import json
import os
from typing import Optional, Dict, Any
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperManager:
    """Verwaltet Whisper-Modelle für Spracherkennung - lädt sie bei Bedarf und hält sie im Speicher"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model_id: Optional[str] = None
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper Manager - Verwende Device: {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration aus config.json"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config-Datei nicht gefunden: {self.config_path}")
            return {}
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Whisper-Modelle zurück"""
        models = {}
        for model_id, model_info in self.config.get("models", {}).items():
            if model_info.get("type") == "audio":
                models[model_id] = model_info
        return models
    
    def get_current_model(self) -> Optional[str]:
        """Gibt die ID des aktuell geladenen Modells zurück"""
        return self.current_model_id
    
    def is_model_loaded(self) -> bool:
        """Prüft ob ein Modell geladen ist"""
        return self.model is not None and self.processor is not None
    
    def load_model(self, model_id: str) -> bool:
        """
        Lädt ein Whisper-Modell. Wenn bereits ein Modell geladen ist, wird es entladen.
        
        Args:
            model_id: Die ID des Modells aus der Config
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if model_id not in self.config.get("models", {}):
            logger.error(f"Modell nicht gefunden: {model_id}")
            return False
        
        model_info = self.config["models"][model_id]
        if model_info.get("type") != "audio":
            logger.error(f"Modell {model_id} ist kein Audio-Modell")
            return False
        
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            logger.error(f"Modell-Pfad existiert nicht: {model_path}")
            return False
        
        try:
            # Altes Modell entladen (Speicher freigeben)
            if self.model is not None:
                logger.info("Entlade aktuelles Whisper-Modell...")
                del self.model
                del self.processor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.model = None
                self.processor = None
            
            logger.info(f"Lade Whisper-Modell: {model_id} von {model_path}")
            
            # Processor und Modell laden
            self.processor = WhisperProcessor.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True
            )
            
            # Auf Device verschieben
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
            
            self.current_model_id = model_id
            logger.info(f"Whisper-Modell erfolgreich geladen: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Whisper-Modells: {e}")
            self.model = None
            self.processor = None
            return False
    
    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        """
        Transkribiert Audio-Daten zu Text
        
        Args:
            audio_data: NumPy-Array mit Audio-Daten (16kHz, mono)
            language: Optional - Sprache (z.B. "de" für Deutsch, "en" für Englisch)
            
        Returns:
            Transkribierter Text
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Whisper-Modell geladen!")
        
        try:
            logger.info(f"Transkribiere Audio (Länge: {len(audio_data)} Samples)...")
            
            # Audio verarbeiten
            inputs = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Auf Device verschieben
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Transkribieren
            with torch.no_grad():
                # Sprache setzen wenn angegeben
                generate_kwargs = {
                    "max_length": 448,
                    "num_beams": 5
                }
                
                if language:
                    # Setze Sprache für Generation
                    generate_kwargs["language"] = language
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs
                )
            
            # Text dekodieren
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Transkription erfolgreich: {transcription[:50]}...")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Transkription: {e}")
            raise

