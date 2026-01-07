"""
Audio Recorder - Aufnahme mit drei Modi: Toggle, Push-to-Talk, Auto
"""
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import tempfile
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecordingMode(Enum):
    """Aufnahme-Modi"""
    TOGGLE = "toggle"
    PUSH_TO_TALK = "push_to_talk"
    AUTO = "auto"


class AudioRecorder:
    """Audio-Recorder mit verschiedenen Aufnahme-Modi"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.mode = RecordingMode.TOGGLE
        self.threshold = 0.5  # Für Auto-Modus
        self.on_recording_state_changed: Optional[Callable[[bool], None]] = None
        
    def set_mode(self, mode: RecordingMode):
        """Setzt Aufnahme-Modus"""
        if self.is_recording:
            self.stop_recording()
        self.mode = mode
        logger.info(f"Aufnahme-Modus geändert: {mode.value}")
    
    def set_threshold(self, threshold: float):
        """Setzt Schwellenwert für Auto-Modus (0.0 - 1.0)"""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Schwellenwert gesetzt: {self.threshold}")
    
    def start_recording(self) -> bool:
        """Startet Aufnahme"""
        if self.is_recording:
            logger.warning("Aufnahme läuft bereits")
            return False
        
        try:
            self.audio_data = []
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio-Status: {status}")
                
                # Konvertiere zu float32
                audio_chunk = indata[:, 0] if self.channels > 1 else indata.flatten()
                self.audio_data.append(audio_chunk.copy())
                
                # Auto-Modus: Prüfe Schwellenwert
                if self.mode == RecordingMode.AUTO and self.is_recording:
                    volume = np.abs(audio_chunk).mean()
                    if volume < self.threshold * 0.1:  # Normalisiere Schwellenwert
                        # Stille erkannt - könnte Aufnahme stoppen
                        # Aber wir sammeln weiter für bessere Erkennung
                        pass
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            
            if self.on_recording_state_changed:
                self.on_recording_state_changed(True)
            
            logger.info("Aufnahme gestartet")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Starten der Aufnahme: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[str]:
        """Stoppt Aufnahme und gibt Pfad zur WAV-Datei zurück"""
        if not self.is_recording:
            logger.warning("Keine Aufnahme aktiv")
            return None
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            
            if self.on_recording_state_changed:
                self.on_recording_state_changed(False)
            
            if not self.audio_data:
                logger.warning("Keine Audio-Daten aufgenommen")
                return None
            
            # Kombiniere alle Audio-Chunks
            audio_array = np.concatenate(self.audio_data)
            
            # Normalisiere
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
            
            # Konvertiere zu int16 für WAV
            audio_int16 = (audio_array * 32767.0).astype(np.int16)
            
            # Speichere als temporäre WAV-Datei
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"speech_input_{int(np.random.random() * 1000000)}.wav"
            
            wavfile.write(str(temp_file), self.sample_rate, audio_int16)
            
            logger.info(f"Audio gespeichert: {temp_file} ({len(audio_array)} Samples)")
            
            # Lösche Audio-Daten aus Speicher
            self.audio_data = []
            
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Fehler beim Stoppen der Aufnahme: {e}")
            self.is_recording = False
            self.audio_data = []
            return None
    
    def toggle_recording(self) -> Optional[str]:
        """Toggle-Aufnahme: Startet wenn gestoppt, stoppt wenn läuft"""
        if self.is_recording:
            return self.stop_recording()
        else:
            self.start_recording()
            return None
    
    def cleanup(self):
        """Bereinigt Ressourcen"""
        if self.is_recording:
            self.stop_recording()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None

