"""
Image Manager - Verwaltet das Laden und Generieren von Bildern mit Diffusionsmodellen
"""
import json
import os
from typing import Optional, Dict, Any
from diffusers import DiffusionPipeline
import torch
import logging
from PIL import Image
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageManager:
    """Verwaltet Bildgenerierungsmodelle - lädt sie bei Bedarf und hält sie im Speicher"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model_id: Optional[str] = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Image Manager - Verwende Device: {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Lädt die Konfiguration aus config.json"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config-Datei nicht gefunden: {self.config_path}")
            return {}
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle verfügbaren Bildgenerierungsmodelle zurück"""
        models = {}
        for model_id, model_info in self.config.get("models", {}).items():
            if model_info.get("type") == "image":
                models[model_id] = model_info
        return models
    
    def get_current_model(self) -> Optional[str]:
        """Gibt die ID des aktuell geladenen Modells zurück"""
        return self.current_model_id
    
    def is_model_loaded(self) -> bool:
        """Prüft ob ein Modell geladen ist"""
        return self.pipeline is not None
    
    def load_model(self, model_id: str) -> bool:
        """
        Lädt ein Bildgenerierungsmodell. Wenn bereits ein Modell geladen ist, wird es entladen.
        
        Args:
            model_id: Die ID des Modells aus der Config
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if model_id not in self.config.get("models", {}):
            logger.error(f"Modell nicht gefunden: {model_id}")
            return False
        
        model_info = self.config["models"][model_id]
        if model_info.get("type") != "image":
            logger.error(f"Modell {model_id} ist kein Bildgenerierungsmodell")
            return False
        
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            logger.error(f"Modell-Pfad existiert nicht: {model_path}")
            return False
        
        try:
            # Altes Modell entladen (Speicher freigeben)
            if self.pipeline is not None:
                logger.info("Entlade aktuelles Modell...")
                del self.pipeline
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                self.pipeline = None
            
            logger.info(f"Lade Bildgenerierungsmodell: {model_id} von {model_path}")
            
            # Pipeline laden
            # Flux-Modelle verwenden DiffusionPipeline
            try:
                # Bestimme dtype (torch_dtype ist deprecated, verwende dtype)
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                
                # Versuche zuerst mit Standard-Parameters
                # Flux unterstützt kein trust_remote_code, also weglassen
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    dtype=dtype,  # Verwende dtype statt torch_dtype
                    local_files_only=False  # Erlaube Downloads falls nötig
                )
            except Exception as e1:
                logger.warning(f"Erster Ladeversuch fehlgeschlagen: {e1}, versuche mit local_files_only=True...")
                try:
                    # Fallback: Nur lokale Dateien
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        dtype=torch.float32,  # Verwende float32 für CPU
                        local_files_only=True
                    )
                except Exception as e2:
                    logger.error(f"Fehler beim Laden der Pipeline: {e2}")
                    raise
            
            # Auf GPU verschieben wenn verfügbar
            if self.device == "cuda":
                try:
                    logger.info("Verschiebe Pipeline auf GPU...")
                    self.pipeline = self.pipeline.to(self.device)
                    logger.info(f"Pipeline erfolgreich auf GPU verschoben: {torch.cuda.get_device_name(0)}")
                    
                    # Optimierungen für bessere Performance
                    # Versuche xformers zu aktivieren (bessere GPU-Performance)
                    try:
                        if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                            logger.info("Aktiviere xformers für bessere GPU-Performance...")
                            self.pipeline.enable_xformers_memory_efficient_attention()
                            logger.info("xformers erfolgreich aktiviert!")
                    except Exception as e:
                        logger.warning(f"xformers konnte nicht aktiviert werden: {e}")
                    
                    # Weitere Optimierungen
                    if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                        logger.info("Aktiviere CPU-Offloading für besseren Speicherverbrauch...")
                        self.pipeline.enable_model_cpu_offload()
                    elif hasattr(self.pipeline, 'enable_attention_slicing'):
                        logger.info("Aktiviere Attention-Slicing...")
                        self.pipeline.enable_attention_slicing()
                    
                    # Prüfe ob Pipeline wirklich auf GPU ist
                    if hasattr(self.pipeline, 'unet'):
                        device_check = next(self.pipeline.unet.parameters()).device
                        logger.info(f"UNet ist auf Device: {device_check}")
                    
                except Exception as e:
                    logger.warning(f"Fehler beim Verschieben auf GPU: {e}, verwende CPU")
                    self.device = "cpu"
                    self.pipeline = self.pipeline.to(self.device)
            else:
                logger.info("Verwende CPU für Bildgenerierung")
                self.pipeline = self.pipeline.to(self.device)
            
            self.current_model_id = model_id
            logger.info(f"Bildgenerierungsmodell erfolgreich geladen: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Bildgenerierungsmodells: {e}")
            self.pipeline = None
            return False
    
    def generate_image(self, prompt: str, negative_prompt: str = "", num_inference_steps: int = 20, guidance_scale: float = 7.5, width: int = 1024, height: int = 1024) -> Optional[Image.Image]:
        """
        Generiert ein Bild basierend auf einem Prompt
        
        Args:
            prompt: Der Text-Prompt für die Bildgenerierung
            negative_prompt: Was nicht im Bild sein soll
            num_inference_steps: Anzahl der Inferenz-Schritte (mehr = besser, aber langsamer)
            guidance_scale: Wie stark der Prompt befolgt wird
            width: Bildbreite
            height: Bildhöhe
            
        Returns:
            PIL Image oder None bei Fehler
        """
        if not self.is_model_loaded():
            raise RuntimeError("Kein Bildgenerierungsmodell geladen!")
        
        try:
            logger.info(f"Generiere Bild mit Prompt: {prompt[:50]}...")
            
            # Generiere Bild
            with torch.no_grad():
                # Flux-spezifische Parameter
                pipeline_kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
                
                # Negative Prompt nur wenn angegeben
                if negative_prompt:
                    pipeline_kwargs["negative_prompt"] = negative_prompt
                
                # Width/Height nur wenn Pipeline sie unterstützt
                if hasattr(self.pipeline, 'width') or 'width' in self.pipeline.__class__.__name__.lower():
                    pipeline_kwargs["width"] = width
                    pipeline_kwargs["height"] = height
                
                result = self.pipeline(**pipeline_kwargs)
            
            # Extrahiere Bild (kann variieren je nach Pipeline)
            if isinstance(result, dict):
                image = result.get("images", [None])[0]
            elif isinstance(result, (list, tuple)):
                image = result[0] if len(result) > 0 else None
            else:
                image = result
            
            if image is None:
                logger.error("Kein Bild von Pipeline erhalten")
                return None
            
            logger.info("Bild erfolgreich generiert")
            return image
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildgenerierung: {e}")
            raise
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Konvertiert ein PIL Image zu Base64-String"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

