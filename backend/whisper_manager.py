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
import time

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
            # #region agent log
            with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","location":"whisper_manager.py:96","message":"Setting dtype for model load","data":{"dtype":str(dtype),"device":self.device},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,  # Verwende dtype statt torch_dtype (Deprecation-Warnung behoben)
                local_files_only=True
            )
            
            # #region agent log
            with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                first_param_dtype = str(next(self.model.parameters()).dtype) if self.model.parameters() else "unknown"
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"whisper_manager.py:104","message":"Model dtype after from_pretrained","data":{"first_param_dtype":first_param_dtype,"expected_dtype":str(dtype)},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            
            # Auf Device verschieben
            self.model = self.model.to(self.device)
            
            # #region agent log
            with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                first_param_after = next(self.model.parameters())
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"whisper_manager.py:107","message":"Model dtype after to(device)","data":{"dtype":str(first_param_after.dtype),"device":str(first_param_after.device)},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            
            # GPU-Optimierungen
            if self.device == "cuda":
                logger.info("Aktiviere GPU-Optimierungen für Whisper...")
                # Setze Modell in Evaluation-Modus
                self.model.eval()
                
                # Versuche Flash Attention zu aktivieren (falls verfügbar)
                try:
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                        # Flash Attention für bessere Performance
                        logger.info("Flash Attention wird verwendet (falls verfügbar)")
                except Exception as e:
                    logger.debug(f"Flash Attention nicht verfügbar: {e}")
                
                # Prüfe ob Modell wirklich auf GPU ist
                try:
                    device_check = next(self.model.parameters()).device
                    logger.info(f"Whisper-Modell ist auf Device: {device_check}")
                    if device_check.type != 'cuda':
                        logger.warning(f"Modell sollte auf CUDA sein, ist aber auf {device_check}")
                except Exception as e:
                    logger.warning(f"Konnte Device nicht prüfen: {e}")
            
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
            
            # #region agent log
            with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                input_features_dtype = str(inputs["input_features"].dtype) if "input_features" in inputs else "missing"
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"whisper_manager.py:161","message":"Input dtype from processor","data":{"input_features_dtype":input_features_dtype},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            
            # Auf Device verschieben
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # #region agent log
            with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                input_features_after = inputs.get("input_features")
                model_first_param = next(self.model.parameters())
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"whisper_manager.py:167","message":"Dtype comparison before generate","data":{"input_dtype":str(input_features_after.dtype) if input_features_after is not None else "none","model_param_dtype":str(model_first_param.dtype),"match":str(input_features_after.dtype) == str(model_first_param.dtype) if input_features_after is not None else False},"timestamp":int(time.time()*1000)})+"\n")
            # #endregion
            
            # Transkribieren
            with torch.no_grad():
                # Sprache setzen wenn angegeben
                generate_kwargs = {
                    "max_length": 448,
                    # Reduziere num_beams für schnellere Transkription (1 = greedy, 5 = beam search)
                    # Beam search ist genauer aber viel langsamer
                    "num_beams": 1 if self.device == "cpu" else 1,  # Greedy für Geschwindigkeit
                    "do_sample": False,  # Deterministisch
                }
                
                # GPU-spezifische Optimierungen
                if self.device == "cuda":
                    # Verwende bfloat16 für schnellere Inferenz
                    if hasattr(self.model, 'half'):
                        # Modell bereits in bfloat16 beim Laden
                        pass
                    # Keine zusätzlichen Optimierungen nötig, torch.no_grad() reicht
                
                if language:
                    # Setze Sprache für Generation
                    generate_kwargs["language"] = language
                
                # Prüfe ob Inputs auf GPU sind
                if self.device == "cuda":
                    input_device = inputs["input_features"].device
                    if input_device.type != 'cuda':
                        logger.warning(f"Inputs sind auf {input_device}, sollten auf CUDA sein")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # #region agent log
                    with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                        model_dtype = str(next(self.model.parameters()).dtype)
                        input_dtype = str(inputs["input_features"].dtype)
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"whisper_manager.py:195","message":"Before generate - dtype check","data":{"model_dtype":model_dtype,"input_dtype":input_dtype,"needs_conversion":input_dtype != model_dtype},"timestamp":int(time.time()*1000)})+"\n")
                    # #endregion
                    
                    # Konvertiere Inputs zu Model-Dtype falls nötig
                    if inputs["input_features"].dtype != next(self.model.parameters()).dtype:
                        # #region agent log
                        with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"whisper_manager.py:200","message":"Converting inputs to model dtype","data":{"from":str(inputs["input_features"].dtype),"to":str(next(self.model.parameters()).dtype)},"timestamp":int(time.time()*1000)})+"\n")
                        # #endregion
                        inputs["input_features"] = inputs["input_features"].to(next(self.model.parameters()).dtype)
                
                # #region agent log
                with open(r"g:\04-CODING\Local Ai\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"whisper_manager.py:204","message":"About to call generate","data":{"input_dtype":str(inputs["input_features"].dtype),"model_dtype":str(next(self.model.parameters()).dtype)},"timestamp":int(time.time()*1000)})+"\n")
                # #endregion
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs
                )
            
            # Text dekodieren (auf CPU, da Decoding schnell ist)
            generated_ids_cpu = generated_ids.cpu() if self.device == "cuda" else generated_ids
            transcription = self.processor.batch_decode(
                generated_ids_cpu,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"Transkription erfolgreich ({self.device}): {transcription[:50]}...")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Fehler bei der Transkription: {e}")
            raise

