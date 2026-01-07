"""
API Client für /audio/transcribe Endpoint
Nutzt die gleiche Schnittstelle wie das Frontend
"""
import requests
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    """Client für Kommunikation mit dem Local AI Server"""
    
    def __init__(self, api_base: str = "http://127.0.0.1:8000"):
        self.api_base = api_base
        self.timeout = 120  # 2 Minuten für Transkription
    
    def transcribe(self, wav_file_path: str, language: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
        """
        Sendet Audio-Datei zur Transkription
        
        Args:
            wav_file_path: Pfad zur WAV-Datei
            language: Optional - Sprache für Transkription (z.B. "de", "en")
            max_retries: Maximale Anzahl Wiederholungen bei Modell-Ladevorgang
            
        Returns:
            Transkribierter Text oder None bei Fehler
        """
        if not Path(wav_file_path).exists():
            logger.error(f"Audio-Datei nicht gefunden: {wav_file_path}")
            return None
        
        url = f"{self.api_base}/audio/transcribe"
        params = {}
        if language:
            params['language'] = language
        
        # Öffne Datei für Upload
        with open(wav_file_path, 'rb') as audio_file:
            files = {
                'file': ('recording.wav', audio_file, 'audio/wav')
            }
            
            retries = 0
            while retries < max_retries:
                try:
                    logger.info(f"Sende Audio zur Transkription (Versuch {retries + 1}/{max_retries})...")
                    
                    response = requests.post(
                        url,
                        files=files,
                        params=params if params else None,
                        timeout=self.timeout
                    )
                    
                    # Prüfe ob Modell geladen wird (202 Accepted)
                    if response.status_code == 202:
                        error_data = response.json()
                        if error_data.get('detail', {}).get('status') == 'model_loading':
                            model_id = error_data.get('detail', {}).get('model_id', 'unknown')
                            logger.info(f"Modell {model_id} wird geladen, warte...")
                            
                            # Warte auf Modell-Laden
                            await_model_load(model_id, self.api_base)
                            
                            # Wiederhole Request
                            retries += 1
                            # Datei-Pointer zurücksetzen
                            audio_file.seek(0)
                            continue
                    
                    if not response.ok:
                        error = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                        error_msg = error.get('detail', f'HTTP {response.status_code}')
                        logger.error(f"Transkriptions-Fehler: {error_msg}")
                        return None
                    
                    # Erfolgreich
                    data = response.json()
                    text = data.get('text', '')
                    logger.info(f"Transkription erfolgreich: {text[:50]}...")
                    return text
                    
                except requests.exceptions.Timeout:
                    logger.error("Timeout bei Transkription")
                    return None
                except requests.exceptions.ConnectionError:
                    logger.error("Verbindungsfehler - Server nicht erreichbar")
                    return None
                except Exception as e:
                    logger.error(f"Unerwarteter Fehler bei Transkription: {e}")
                    return None
            
            logger.error("Maximale Anzahl Wiederholungen erreicht")
            return None
    
    def check_server_status(self) -> bool:
        """Prüft ob Server erreichbar ist"""
        try:
            response = requests.get(f"{self.api_base}/status", timeout=2)
            return response.status_code == 200
        except:
            return False


def await_model_load(model_id: str, api_base: str, max_wait: int = 300) -> bool:
    """
    Wartet bis Modell geladen ist
    
    Args:
        model_id: ID des Modells
        api_base: Base URL des Servers
        max_wait: Maximale Wartezeit in Sekunden
        
    Returns:
        True wenn Modell geladen ist, False bei Timeout
    """
    start_time = time.time()
    check_interval = 2  # Prüfe alle 2 Sekunden
    
    while time.time() - start_time < max_wait:
        try:
            # Prüfe Audio-Modell-Status
            response = requests.get(
                f"{api_base}/audio/models/load/status",
                timeout=5
            )
            
            if response.status_code == 200:
                status = response.json()
                if not status.get('loading', False):
                    logger.info("Modell ist geladen")
                    return True
            
            time.sleep(check_interval)
        except Exception as e:
            logger.debug(f"Fehler beim Prüfen des Modell-Status: {e}")
            time.sleep(check_interval)
    
    logger.warning("Timeout beim Warten auf Modell-Laden")
    return False

