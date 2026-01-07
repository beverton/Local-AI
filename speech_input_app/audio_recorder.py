"""
Audio Recorder - Aufnahme mit drei Modi: Toggle, Push-to-Talk, Auto
Unterstützt jetzt auch segment-basierte Transkription mit Pause-Erkennung
"""
import logging
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import tempfile
from pathlib import Path
from typing import Optional, Callable
from enum import Enum
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecordingMode(Enum):
    """Aufnahme-Modi"""
    TOGGLE = "toggle"
    PUSH_TO_TALK = "push_to_talk"
    AUTO = "auto"
    CONTINUOUS = "continuous"  # Neuer Modus für segment-basierte Transkription


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
        
        # Für segment-basierte Transkription
        self.on_segment_ready: Optional[Callable[[str], None]] = None  # Callback wenn Segment fertig ist
        self.silence_threshold = 0.02  # Schwellenwert für Stille (0.0-1.0) - erhöht für bessere Genauigkeit
        self.silence_duration = 1.0  # Sekunden Stille bevor Segment extrahiert wird
        self.min_segment_duration = 1.0  # Minimale Segment-Dauer in Sekunden (erhöht für mehr Kontext)
        self.trailing_silence_duration = 0.3  # Sekunden Stille am Ende des Segments (wichtig für Whisper um Wortenden zu erkennen)
        
        # Audio-Verarbeitung (experimentell)
        self.clipping_protection = True  # Clipping-Schutz aktivieren/deaktivieren
        self.normalize_audio = True  # Normalisierung aktivieren/deaktivieren
        self.normalize_level = 1.0  # Normalisierungs-Level (0.5 - 1.0)
        self.compress_audio = False  # Kompression aktivieren/deaktivieren
        self.compress_ratio = 2.0  # Kompressions-Ratio (1.0 - 10.0)
        self.compress_threshold = 0.5  # Kompressions-Threshold (0.1 - 1.0)
        
        # Pause-Erkennung
        self.last_speech_time = None  # Zeitpunkt des letzten erkannten Sprechens
        self.current_segment_data = []  # Aktuelles Segment
        self.trailing_silence_data = []  # Trailing silence für aktuelles Segment
        self.is_in_speech = False  # Ob gerade gesprochen wird
        self.silence_buffer = deque(maxlen=int(sample_rate * 0.1 / 1024))  # Buffer für Stille-Erkennung
        
        # Adaptive Baseline für bessere Erkennung
        self.baseline_volume = 0.005  # Geschätzte Baseline (wird während Aufnahme angepasst)
        self.volume_history = deque(maxlen=50)  # Historie für Baseline-Anpassung
        
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
    
    def set_silence_threshold(self, threshold: float):
        """Setzt Schwellenwert für Stille-Erkennung (0.0 - 1.0)"""
        self.silence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Stille-Schwellenwert gesetzt: {self.silence_threshold}")
    
    def set_silence_duration(self, duration: float):
        """Setzt Dauer der Stille bevor Segment extrahiert wird (in Sekunden)"""
        self.silence_duration = max(0.1, duration)
        logger.info(f"Stille-Dauer gesetzt: {self.silence_duration} Sekunden")
    
    def set_clipping_protection(self, enabled: bool):
        """Aktiviert/deaktiviert Clipping-Schutz"""
        self.clipping_protection = enabled
        logger.info(f"Clipping-Schutz: {'aktiviert' if enabled else 'deaktiviert'}")
    
    def set_normalize_audio(self, enabled: bool):
        """Aktiviert/deaktiviert Normalisierung"""
        self.normalize_audio = enabled
        logger.info(f"Normalisierung: {'aktiviert' if enabled else 'deaktiviert'}")
    
    def set_normalize_level(self, level: float):
        """Setzt Normalisierungs-Level (0.5 - 1.0)"""
        self.normalize_level = max(0.5, min(1.0, level))
        logger.info(f"Normalisierungs-Level gesetzt: {self.normalize_level}")
    
    def set_compress_audio(self, enabled: bool):
        """Aktiviert/deaktiviert Kompression"""
        self.compress_audio = enabled
        logger.info(f"Kompression: {'aktiviert' if enabled else 'deaktiviert'}")
    
    def set_compress_ratio(self, ratio: float):
        """Setzt Kompressions-Ratio (1.0 - 10.0)"""
        self.compress_ratio = max(1.0, min(10.0, ratio))
        logger.info(f"Kompressions-Ratio gesetzt: {self.compress_ratio}:1")
    
    def set_compress_threshold(self, threshold: float):
        """Setzt Kompressions-Threshold (0.1 - 1.0)"""
        self.compress_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Kompressions-Threshold gesetzt: {self.compress_threshold}")
    
    def start_recording(self) -> bool:
        """Startet Aufnahme"""
        if self.is_recording:
            logger.warning("Aufnahme läuft bereits")
            return False
        
        try:
            self.audio_data = []
            
            # Für CONTINUOUS-Modus: Initialisiere Segment-Tracking
            if self.mode == RecordingMode.CONTINUOUS:
                self.current_segment_data = []
                self.trailing_silence_data = []
                self.last_speech_time = None
                self.is_in_speech = False
                self.silence_buffer.clear()
                self.volume_history.clear()
                self.baseline_volume = 0.005  # Reset Baseline
                logger.info(f"CONTINUOUS-Modus: Segment-Tracking initialisiert (min_duration={self.min_segment_duration}s, trailing={self.trailing_silence_duration}s)")
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio-Status: {status}")
                
                # Konvertiere zu float32
                audio_chunk = indata[:, 0] if self.channels > 1 else indata.flatten()
                current_time = time.time()
                
                # Speichere in Haupt-Buffer
                self.audio_data.append(audio_chunk.copy())
                
                # CONTINUOUS-Modus: Pause-Erkennung und Segment-Extraktion
                if self.mode == RecordingMode.CONTINUOUS and self.is_recording:
                    # Verwende RMS (Root Mean Square) für genauere Volume-Messung
                    rms = np.sqrt(np.mean(audio_chunk**2))
                    volume = rms
                    
                    # Aktualisiere Volume-Historie für adaptive Baseline
                    self.volume_history.append(volume)
                    
                    # Berechne adaptive Baseline (Median der niedrigsten 30% der Werte)
                    if len(self.volume_history) > 10:
                        sorted_volumes = sorted(self.volume_history)
                        baseline_idx = int(len(sorted_volumes) * 0.3)
                        self.baseline_volume = sorted_volumes[baseline_idx] if baseline_idx > 0 else sorted_volumes[0]
                    
                    # Verwende adaptiven Threshold: baseline + konfigurierter Threshold
                    # Dies passt sich automatisch an die Umgebung an
                    adaptive_threshold = self.baseline_volume + self.silence_threshold
                    
                    # Logging für Debugging (nur gelegentlich, um Logs nicht zu überfluten)
                    if len(self.volume_history) % 100 == 0:  # Alle 100 Chunks
                        logger.debug(f"Volume: {volume:.4f}, Baseline: {self.baseline_volume:.4f}, Threshold: {adaptive_threshold:.4f}")
                    
                    # Prüfe ob Sprache erkannt wird
                    if volume > adaptive_threshold:
                        # Sprache erkannt
                        if not self.is_in_speech:
                            self.is_in_speech = True
                            logger.debug("Sprache erkannt - Segment startet")
                            # Füge trailing silence vom vorherigen Segment hinzu (falls vorhanden)
                            if self.trailing_silence_data:
                                self.current_segment_data.extend(self.trailing_silence_data)
                                self.trailing_silence_data = []
                        
                        self.last_speech_time = current_time
                        self.current_segment_data.append(audio_chunk.copy())
                        self.silence_buffer.append(True)  # Nicht stille
                    else:
                        # Stille erkannt
                        self.silence_buffer.append(False)  # Stille
                        
                        # Wenn wir in Sprache waren
                        if self.is_in_speech:
                            # Sammle trailing silence (für bessere Transkription)
                            # Berechne aktuelle trailing silence Dauer
                            current_trailing_samples = sum(len(chunk) for chunk in self.trailing_silence_data)
                            max_trailing_samples = int(self.sample_rate * self.trailing_silence_duration)
                            
                            # Füge nur hinzu wenn noch Platz ist
                            if current_trailing_samples + len(audio_chunk) <= max_trailing_samples:
                                self.trailing_silence_data.append(audio_chunk.copy())
                            elif current_trailing_samples < max_trailing_samples:
                                # Füge nur den Teil hinzu der noch passt
                                remaining_samples = max_trailing_samples - current_trailing_samples
                                if remaining_samples > 0:
                                    self.trailing_silence_data.append(audio_chunk[:remaining_samples].copy())
                            
                            # Prüfe ob Stille lang genug ist
                            if self.last_speech_time and (current_time - self.last_speech_time) >= self.silence_duration:
                                # Segment fertig - extrahiere es
                                self._extract_segment()
                                self.is_in_speech = False
                                self.last_speech_time = None
                                self.trailing_silence_data = []
                
                # Auto-Modus: Prüfe Schwellenwert
                elif self.mode == RecordingMode.AUTO and self.is_recording:
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
            
            logger.info(f"Aufnahme gestartet (Modus: {self.mode.value})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Starten der Aufnahme: {e}")
            self.is_recording = False
            return False
    
    def _extract_segment(self):
        """Extrahiert aktuelles Segment und speichert es als WAV-Datei"""
        if not self.current_segment_data:
            logger.warning("Keine Segment-Daten zum Extrahieren")
            return
        
        try:
            # Kombiniere Segment-Daten (inkl. trailing silence)
            segment_chunks = list(self.current_segment_data)
            if self.trailing_silence_data:
                segment_chunks.extend(self.trailing_silence_data)
            segment_array = np.concatenate(segment_chunks)
            
            # Prüfe minimale Dauer
            segment_duration = len(segment_array) / self.sample_rate
            if segment_duration < self.min_segment_duration:
                logger.debug(f"Segment zu kurz ({segment_duration:.2f}s), ignoriere")
                self.current_segment_data = []
                return
            
            # KEINE leading silence - genau wie TOGGLE-Modus
            
            # Audio-Verarbeitung (Clipping-Schutz, Normalisierung und Kompression)
            max_amplitude_before = np.abs(segment_array).max()
            logger.info(f"🔊 Segment Audio-Info: max_amplitude={max_amplitude_before:.4f}, clipping_protection={self.clipping_protection}, normalize={self.normalize_audio}, compress={self.compress_audio}")
            
            # Clipping-Schutz (separat, wird ZUERST angewendet)
            if self.clipping_protection:
                if max_amplitude_before > 1.0:
                    segment_array = segment_array / max_amplitude_before
                    logger.info(f"✅ Clipping-Schutz angewendet: {max_amplitude_before:.4f} -> 1.0")
                    max_amplitude_before = 1.0  # Update für weitere Verarbeitung
                else:
                    logger.info(f"ℹ️ Clipping-Schutz aktiviert, aber kein Clipping (max_amplitude={max_amplitude_before:.4f})")
            else:
                logger.info(f"⚠️ Clipping-Schutz DEAKTIVIERT - Audio bleibt unverändert (auch bei Clipping)")
            
            # Normalisierung (wenn aktiviert)
            if self.normalize_audio:
                current_max = np.abs(segment_array).max()
                if current_max > 0.01:  # Nur normalisieren wenn Audio vorhanden
                    segment_array = segment_array / current_max * self.normalize_level
                    logger.info(f"✅ Normalisierung angewendet: {current_max:.4f} -> {np.abs(segment_array).max():.4f} (Level: {self.normalize_level})")
                else:
                    logger.warning("⚠️ Audio zu leise für Normalisierung (max_amplitude < 0.01)")
            else:
                logger.info(f"ℹ️ Normalisierung deaktiviert - Audio-Level bleibt unverändert")
            
            # Kompression (wenn aktiviert)
            if self.compress_audio:
                # Einfache Kompression: Reduziere Peaks über Threshold
                compressed = np.copy(segment_array)
                abs_audio = np.abs(compressed)
                mask = abs_audio > self.compress_threshold
                
                if np.any(mask):
                    samples_before = np.sum(mask)
                    # Komprimiere Signale über Threshold
                    # Formel: output = threshold + (input - threshold) / ratio
                    compressed[mask] = np.sign(compressed[mask]) * (
                        self.compress_threshold + 
                        (abs_audio[mask] - self.compress_threshold) / self.compress_ratio
                    )
                    segment_array = compressed
                    max_after = np.abs(segment_array).max()
                    logger.info(f"✅ Kompression angewendet: {samples_before} Samples komprimiert (Ratio: {self.compress_ratio}:1, Threshold: {self.compress_threshold}, max: {max_after:.4f})")
                else:
                    logger.info(f"ℹ️ Kompression aktiviert, aber keine Samples über Threshold ({self.compress_threshold})")
            else:
                logger.info(f"ℹ️ Kompression deaktiviert")
            
            max_amplitude_after = np.abs(segment_array).max()
            logger.info(f"📊 Finale Audio-Info: max_amplitude={max_amplitude_after:.4f} (vorher: {max_amplitude_before:.4f})")
            
            # Konvertiere zu int16 für WAV
            audio_int16 = (segment_array * 32767.0).astype(np.int16)
            
            # Speichere als temporäre WAV-Datei
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"speech_segment_{int(time.time() * 1000)}.wav"
            
            wavfile.write(str(temp_file), self.sample_rate, audio_int16)
            
            logger.info(f"Segment extrahiert: {temp_file} ({segment_duration:.2f}s, {len(segment_array)} Samples)")
            
            # Lösche Segment-Daten (aber behalte Haupt-Buffer)
            self.current_segment_data = []
            
            # Rufe Callback auf wenn verfügbar
            if self.on_segment_ready:
                try:
                    self.on_segment_ready(str(temp_file))
                except Exception as e:
                    logger.error(f"Fehler beim Aufruf des Segment-Callbacks: {e}")
            else:
                logger.warning("Kein Segment-Callback registriert")
                
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Segments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.current_segment_data = []
    
    def stop_recording(self) -> Optional[str]:
        """Stoppt Aufnahme und gibt Pfad zur WAV-Datei zurück"""
        if not self.is_recording:
            logger.warning("Keine Aufnahme aktiv")
            return None
        
        try:
            # Im CONTINUOUS-Modus: Extrahiere letztes Segment falls vorhanden
            if self.mode == RecordingMode.CONTINUOUS and self.current_segment_data:
                logger.info("Extrahiere letztes Segment vor Stopp...")
                self._extract_segment()
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            
            if self.on_recording_state_changed:
                self.on_recording_state_changed(False)
            
            # Im CONTINUOUS-Modus: Gib None zurück (Segmente wurden bereits verarbeitet)
            if self.mode == RecordingMode.CONTINUOUS:
                logger.info("CONTINUOUS-Modus: Aufnahme gestoppt, keine finale WAV-Datei")
                self.audio_data = []  # Lösche Buffer
                return None
            
            # Für andere Modi: Gib finale WAV-Datei zurück
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

