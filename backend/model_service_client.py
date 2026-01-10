"""
Model Service Client - Wrapper für HTTP-Calls zum Model-Service
"""
import requests
import logging
import time
import socket
import os
from typing import Optional, Dict, Any, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelServiceClient:
    """Client für Kommunikation mit dem Model-Service"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client_id = self._generate_client_id()
        self.app_name = "Local AI Server"
        self._status_cache = {}
        self._cache_timeout = 2.0  # Cache für 2 Sekunden
        
    def _generate_client_id(self) -> str:
        """Generiert eine eindeutige Client-ID"""
        import platform
        import os as os_module
        hostname = platform.node()
        pid = os_module.getpid()
        return f"local-ai-server-{hostname}-{pid}"
    
    def _get_headers(self) -> Dict[str, str]:
        """Gibt Standard-Headers mit Client-Info zurück"""
        return {
            "X-Client-ID": self.client_id,
            "X-App-Name": self.app_name,
            "Content-Type": "application/json"
        }
    
    def is_available(self) -> bool:
        """Prüft ob Model-Service erreichbar ist"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Gibt Status aller Modelle zurück"""
        try:
            response = requests.get(
                f"{self.base_url}/status",
                headers=self._get_headers(),
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Model-Service Status-Fehler: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Model-Service-Status: {e}")
            return None
    
    def load_text_model(self, model_id: str) -> bool:
        """Lädt ein Text-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/text/load",
                json={"model_id": model_id},
                headers=self._get_headers(),
                timeout=300  # 5 Minuten für Modell-Laden
            )
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Fehler beim Laden des Text-Modells: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Laden des Text-Modells: {e}")
            return False
    
    def unload_text_model(self) -> bool:
        """Entlädt das Text-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/text/unload",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Fehler beim Entladen des Text-Modells: {e}")
            return False
    
    def get_text_model_status(self) -> Optional[Dict[str, Any]]:
        """Gibt Status des Text-Modells zurück"""
        try:
            response = requests.get(
                f"{self.base_url}/models/text/status",
                headers=self._get_headers(),
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Text-Modell-Status: {e}")
            return None
    
    def load_audio_model(self, model_id: str) -> bool:
        """Lädt ein Audio-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/audio/load",
                json={"model_id": model_id},
                headers=self._get_headers(),
                timeout=300
            )
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Fehler beim Laden des Audio-Modells: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Laden des Audio-Modells: {e}")
            return False
    
    def unload_audio_model(self) -> bool:
        """Entlädt das Audio-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/audio/unload",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Fehler beim Entladen des Audio-Modells: {e}")
            return False
    
    def get_audio_model_status(self) -> Optional[Dict[str, Any]]:
        """Gibt Status des Audio-Modells zurück"""
        try:
            response = requests.get(
                f"{self.base_url}/models/audio/status",
                headers=self._get_headers(),
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Audio-Modell-Status: {e}")
            return None
    
    def load_image_model(self, model_id: str) -> bool:
        """Lädt ein Image-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/image/load",
                json={"model_id": model_id},
                headers=self._get_headers(),
                timeout=600  # 10 Minuten für Image-Modelle
            )
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Fehler beim Laden des Image-Modells: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Fehler beim Laden des Image-Modells: {e}")
            return False
    
    def unload_image_model(self) -> bool:
        """Entlädt das Image-Modell"""
        try:
            response = requests.post(
                f"{self.base_url}/models/image/unload",
                headers=self._get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Fehler beim Entladen des Image-Modells: {e}")
            return False
    
    def get_image_model_status(self) -> Optional[Dict[str, Any]]:
        """Gibt Status des Image-Modells zurück"""
        try:
            response = requests.get(
                f"{self.base_url}/models/image/status",
                headers=self._get_headers(),
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Image-Modell-Status: {e}")
            return None
    
    def chat(self, message: str, messages: Optional[List[Dict[str, str]]] = None, conversation_id: Optional[str] = None, max_length: Optional[int] = None, temperature: Optional[float] = None, language: Optional[str] = None, profile: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sendet Chat-Request an Model-Service"""
        try:
            payload = {
                "message": message,
                "conversation_id": conversation_id
            }
            # Nur Parameter hinzufügen, wenn sie gesetzt sind (None = Model Service verwendet Settings/Profil)
            if max_length is not None:
                payload["max_length"] = max_length
            if temperature is not None:
                payload["temperature"] = temperature
            if messages:
                payload["messages"] = messages
            if language:
                payload["language"] = language
            if profile:
                payload["profile"] = profile
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers=self._get_headers(),
                timeout=120  # 2 Minuten für Chat
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Fehler bei Chat-Request: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Fehler bei Chat-Request: {e}")
            return None
    
    def transcribe(self, audio_base64: str, language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sendet Transkriptions-Request an Model-Service"""
        try:
            response = requests.post(
                f"{self.base_url}/transcribe",
                json={
                    "audio_base64": audio_base64,
                    "language": language
                },
                headers=self._get_headers(),
                timeout=60  # 1 Minute für Transkription
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Fehler bei Transkriptions-Request: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Fehler bei Transkriptions-Request: {e}")
            return None
    
    def generate_image(self, prompt: str, negative_prompt: str = "", num_inference_steps: int = 20, 
                      guidance_scale: float = 7.5, width: int = 1024, height: int = 1024, 
                      aspect_ratio: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sendet Bildgenerierungs-Request an Model-Service"""
        try:
            json_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height
            }
            if aspect_ratio:
                json_data["aspect_ratio"] = aspect_ratio
            response = requests.post(
                f"{self.base_url}/generate_image",
                json=json_data,
                headers=self._get_headers(),
                timeout=300  # 5 Minuten für Bildgenerierung
            )
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            #logger.error(f"Fehler bei Bildgenerierungs-Request: {e}")
            return None
    
    def list_text_models(self) -> List[Dict[str, Any]]:
        """Listet alle verfügbaren Text-Modelle"""
        try:
            # Lade Config direkt (Model Service hat Zugriff auf config.json)
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    models = config.get("models", {})
                    text_models = {k: v for k, v in models.items() if v.get("type") in ["qwen2", "phi3", "mistral", "text"]}
                    return [{"id": k, **v} for k, v in text_models.items()]
            return []
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Text-Modelle: {e}")
            return []
    
    def list_image_models(self) -> List[Dict[str, Any]]:
        """Listet alle verfügbaren Image-Modelle"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    models = config.get("models", {})
                    image_models = {k: v for k, v in models.items() if v.get("type") == "image"}
                    return [{"id": k, **v} for k, v in image_models.items()]
            return []
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Image-Modelle: {e}")
            return []
    
    def list_audio_models(self) -> List[Dict[str, Any]]:
        """Listet alle verfügbaren Audio-Modelle"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    models = config.get("models", {})
                    audio_models = {k: v for k, v in models.items() if v.get("type") == "audio"}
                    return [{"id": k, **v} for k, v in audio_models.items()]
            return []
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Audio-Modelle: {e}")
            return []

