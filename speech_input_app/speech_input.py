"""
Speech Input App - Hauptanwendung
PyQt6 App mit System-Tray-Integration
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Füge Parent-Verzeichnis zum Python-Pfad hinzu für direkte Ausführung
if __name__ == "__main__":
    app_dir = Path(__file__).parent
    parent_dir = app_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QWidget, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QHBoxLayout,
    QMessageBox, QCheckBox, QGroupBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush

from speech_input_app.audio_recorder import AudioRecorder, RecordingMode
from speech_input_app.hotkey_handler import HotkeyHandler
from speech_input_app.api_client import APIClient
from speech_input_app.text_injector import TextInjector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionThread(QThread):
    """Thread für Transkription (blockiert UI nicht)"""
    finished = pyqtSignal(str)  # Signal mit transkribiertem Text
    error = pyqtSignal(str)  # Signal mit Fehlermeldung
    
    def __init__(self, wav_file_path: str, api_client: APIClient, language: Optional[str] = None, callback=None, app_instance=None):
        super().__init__()
        self.wav_file_path = wav_file_path
        self.api_client = api_client
        self.language = language
        self.callback = callback  # Direkter Callback als Fallback
        self.app_instance = app_instance  # QApplication-Instanz für Thread-sichere Aufrufe
    
    def run(self):
        """Führt Transkription aus"""
        try:
            logger.info(f"[TranscriptionThread] Starte Transkription für: {self.wav_file_path}")
            text = self.api_client.transcribe(self.wav_file_path, self.language)
            if text:
                logger.info(f"[TranscriptionThread] Transkription erfolgreich, sende Signal mit Text: '{text[:50]}...'")
                
                # Versuche zuerst Signal
                try:
                    self.finished.emit(text)
                    logger.info(f"[TranscriptionThread] Signal 'finished' wurde ausgelöst")
                except Exception as e:
                    logger.error(f"[TranscriptionThread] Fehler beim Emit des Signals: {e}")
                
                # WICHTIG: Direkter Callback-Aufruf (Signal sollte funktionieren, aber als Backup)
                # Das Signal sollte automatisch im Hauptthread verarbeitet werden
                # Aber wir rufen auch den Callback direkt auf, falls das Signal nicht ankommt
                if self.callback:
                    try:
                        logger.info(f"[TranscriptionThread] ⚡ Rufe Callback DIREKT auf mit Text: '{text[:50]}...'")
                        # Direkter Aufruf - sollte funktionieren, da PyQt6 Signale im Hauptthread verarbeitet werden
                        # Aber wir rufen es auch direkt auf, um sicherzustellen, dass es funktioniert
                        self.callback(text)
                        logger.info("[TranscriptionThread] ✅ Callback erfolgreich aufgerufen")
                    except Exception as e:
                        logger.error(f"[TranscriptionThread] ❌ Fehler beim direkten Callback: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            else:
                logger.error("[TranscriptionThread] Transkription fehlgeschlagen - kein Text erhalten")
                self.error.emit("Transkription fehlgeschlagen")
        except Exception as e:
            logger.error(f"[TranscriptionThread] Fehler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(f"Fehler: {str(e)}")




class ConfigWindow(QWidget):
    """Konfigurationsfenster"""
    
    # Signal für Thread-sichere UI-Updates
    key_detected = pyqtSignal(str)
    
    def __init__(self, config_path: Path, on_config_changed):
        super().__init__()
        self.config_path = config_path
        self.on_config_changed = on_config_changed
        self.config = self.load_config()
        self.audio_level_stream = None
        self.current_audio_level = 0.0
        
        # Verbinde Signal mit Slot
        self.key_detected.connect(self.finish_main_key_detection)
        
        self.init_ui()
    
    def load_config(self) -> dict:
        """Lädt Konfiguration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return {
            "hotkey": "alt+enter",
            "recording_mode": "toggle",
            "threshold": 0.5,
            "language": "",
            "auto_start": False,
            "api_base": "http://127.0.0.1:8000"
        }
    
    def save_config(self):
        """Speichert Konfiguration"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.on_config_changed(self.config)
            logger.info("Konfiguration gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
    
    def init_ui(self):
        """Initialisiert UI"""
        self.setWindowTitle("Speech Input - Einstellungen")
        self.setFixedSize(500, 650)
        
        layout = QVBoxLayout()
        
        # Hotkey
        hotkey_group = QGroupBox("Tastenkürzel")
        hotkey_layout = QVBoxLayout()
        
        # Modifier-Keys
        modifier_layout = QHBoxLayout()
        modifier_layout.addWidget(QLabel("Modifier-Keys:"))
        
        self.modifier_ctrl = QCheckBox("Ctrl")
        self.modifier_alt = QCheckBox("Alt")
        self.modifier_shift = QCheckBox("Shift")
        self.modifier_win = QCheckBox("Win")
        
        # Setze aktuelle Modifier-Keys aus Config
        current_hotkey = self.config.get("hotkey", "alt+enter").lower()
        if "ctrl" in current_hotkey:
            self.modifier_ctrl.setChecked(True)
        if "alt" in current_hotkey:
            self.modifier_alt.setChecked(True)
        if "shift" in current_hotkey:
            self.modifier_shift.setChecked(True)
        if "win" in current_hotkey:
            self.modifier_win.setChecked(True)
        
        modifier_layout.addWidget(self.modifier_ctrl)
        modifier_layout.addWidget(self.modifier_alt)
        modifier_layout.addWidget(self.modifier_shift)
        modifier_layout.addWidget(self.modifier_win)
        modifier_layout.addStretch()
        
        hotkey_layout.addLayout(modifier_layout)
        
        # Haupttaste
        main_key_layout = QHBoxLayout()
        main_key_layout.addWidget(QLabel("Haupttaste:"))
        
        self.main_key_input = QLineEdit()
        # Extrahiere Haupttaste aus Config
        main_key = self._extract_main_key(current_hotkey)
        self.main_key_input.setText(main_key)
        self.main_key_input.setPlaceholderText("Drücke Taste...")
        self.main_key_input.setReadOnly(True)
        
        self.detect_main_key_btn = QPushButton("🔍 Erkennen")
        self.detect_main_key_btn.clicked.connect(self.start_main_key_detection)
        self.detecting_main_key = False
        
        main_key_layout.addWidget(self.main_key_input)
        main_key_layout.addWidget(self.detect_main_key_btn)
        
        hotkey_layout.addLayout(main_key_layout)
        
        # Vorschau
        self.hotkey_preview = QLabel()
        self.hotkey_preview.setText(f"Vorschau: {self._build_hotkey_string()}")
        self.hotkey_preview.setStyleSheet("font-weight: bold; color: #2196F3;")
        hotkey_layout.addWidget(self.hotkey_preview)
        
        # Update Vorschau wenn sich etwas ändert
        self.modifier_ctrl.toggled.connect(self.update_hotkey_preview)
        self.modifier_alt.toggled.connect(self.update_hotkey_preview)
        self.modifier_shift.toggled.connect(self.update_hotkey_preview)
        self.modifier_win.toggled.connect(self.update_hotkey_preview)
        self.main_key_input.textChanged.connect(self.update_hotkey_preview)
        
        hotkey_group.setLayout(hotkey_layout)
        layout.addWidget(hotkey_group)
        
        # Aufnahme-Modus
        mode_group = QGroupBox("Aufnahme-Modus")
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Toggle", "Push-to-Talk", "Auto"])
        current_mode = self.config.get("recording_mode", "toggle")
        mode_index = {"toggle": 0, "push_to_talk": 1, "auto": 2}.get(current_mode, 0)
        self.mode_combo.setCurrentIndex(mode_index)
        mode_layout.addWidget(QLabel("Modus:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Schwellenwert (nur für Auto-Modus)
        threshold_group = QGroupBox("Audio-Level & Schwellenwert (Auto-Modus)")
        threshold_layout = QVBoxLayout()
        
        # Kombinierte Audio-Level Visualisierung mit Schwellenwert
        level_vis_layout = QVBoxLayout()
        level_vis_layout.addWidget(QLabel("Aktuelles Audio-Level (orange Linie = Schwellenwert):"))
        
        # Container für übereinander gelegte Balken
        level_container = QWidget()
        level_container.setMinimumHeight(35)
        level_container.setMaximumHeight(35)
        level_container.setStyleSheet("position: relative;")
        
        # ProgressBar für Audio-Level (Hintergrund)
        self.audio_level_bar = QProgressBar(level_container)
        self.audio_level_bar.setMinimum(0)
        self.audio_level_bar.setMaximum(100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setFormat("%v%")
        self.audio_level_bar.setTextVisible(True)
        self.audio_level_bar.setGeometry(0, 0, 500, 35)  # Wird später angepasst
        
        # Schwellenwert-Anzeige als Overlay (über dem Audio-Level-Balken)
        self.threshold_display_bar = QProgressBar(level_container)
        self.threshold_display_bar.setMinimum(0)
        self.threshold_display_bar.setMaximum(100)
        initial_threshold_value = int(self.config.get("threshold", 0.5) * 100)
        self.threshold_display_bar.setValue(initial_threshold_value)
        self.threshold_display_bar.setFormat("")
        self.threshold_display_bar.setTextVisible(False)
        self.threshold_display_bar.setGeometry(0, 0, 500, 35)  # Wird später angepasst
        # Transparent mit nur der Linie sichtbar
        self.threshold_display_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: transparent;
            }
            QProgressBar::chunk {
                background-color: transparent;
                border-right: 3px solid #ff5722;
            }
        """)
        
        # Resize-Event Handler für Container
        def resize_level_container(event):
            width = level_container.width()
            height = level_container.height()
            if width > 0 and height > 0:
                self.audio_level_bar.setGeometry(0, 0, width, height)
                self.threshold_display_bar.setGeometry(0, 0, width, height)
            QWidget.resizeEvent(level_container, event)
        
        level_container.resizeEvent = resize_level_container
        
        level_vis_layout.addWidget(level_container)
        
        # Info-Label
        info_label = QLabel("💡 Die orange vertikale Linie zeigt den Schwellenwert. Wenn das Level darüber liegt, wird die Aufnahme aktiviert.")
        info_label.setWordWrap(True)
        level_vis_layout.addWidget(info_label)
        
        threshold_layout.addLayout(level_vis_layout)
        threshold_layout.addWidget(QLabel(""))  # Abstand
        
        # Schwellenwert-Slider
        threshold_slider_layout = QHBoxLayout()
        threshold_slider_layout.addWidget(QLabel("Schwellenwert:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        initial_threshold_value = int(self.config.get("threshold", 0.5) * 100)
        self.threshold_slider.setValue(initial_threshold_value)
        self.threshold_label = QLabel(f"{self.config.get('threshold', 0.5):.2f}")
        self.threshold_label.setMinimumWidth(50)
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_threshold(v):
            threshold_value = v / 100.0
            self.threshold_label.setText(f"{threshold_value:.2f}")
            # Aktualisiere Schwellenwert-Anzeige
            if hasattr(self, 'threshold_display_bar'):
                self.threshold_display_bar.setValue(v)
            self.update_audio_level_display()
        
        self.threshold_slider.valueChanged.connect(update_threshold)
        threshold_slider_layout.addWidget(self.threshold_slider)
        threshold_slider_layout.addWidget(self.threshold_label)
        threshold_layout.addLayout(threshold_slider_layout)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Initialisiere Audio-Level-Monitoring
        self.audio_level_stream = None
        self.current_audio_level = 0.0
        self.level_monitor_timer = QTimer()
        self.level_monitor_timer.timeout.connect(self.update_audio_level_display)
        self.start_audio_level_monitoring()
        
        # Sprache
        language_group = QGroupBox("Sprache")
        language_layout = QVBoxLayout()
        self.language_combo = QComboBox()
        # Gängige Sprachen für Whisper
        languages = [
            ("Auto (Automatisch)", ""),
            ("Deutsch", "de"),
            ("Englisch", "en"),
            ("Französisch", "fr"),
            ("Spanisch", "es"),
            ("Italienisch", "it"),
            ("Portugiesisch", "pt"),
            ("Niederländisch", "nl"),
            ("Polnisch", "pl"),
            ("Russisch", "ru"),
            ("Chinesisch", "zh"),
            ("Japanisch", "ja"),
            ("Koreanisch", "ko"),
            ("Türkisch", "tr"),
            ("Arabisch", "ar"),
        ]
        for lang_name, lang_code in languages:
            self.language_combo.addItem(lang_name, lang_code)
        
        # Setze aktuelle Sprache
        current_lang = self.config.get("language", "")
        for i in range(self.language_combo.count()):
            if self.language_combo.itemData(i) == current_lang:
                self.language_combo.setCurrentIndex(i)
                break
        
        language_layout.addWidget(QLabel("Sprache:"))
        language_layout.addWidget(self.language_combo)
        language_group.setLayout(language_layout)
        layout.addWidget(language_group)
        
        # Auto-Start
        auto_start_check = QCheckBox("Beim Windows-Start starten")
        auto_start_check.setChecked(self.config.get("auto_start", False))
        self.auto_start_check = auto_start_check
        layout.addWidget(auto_start_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Speichern")
        save_btn.clicked.connect(self.save_and_close)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _extract_main_key(self, hotkey: str) -> str:
        """Extrahiert Haupttaste aus Hotkey-String"""
        parts = hotkey.split('+')
        # Die letzte Taste ist normalerweise die Haupttaste
        for part in reversed(parts):
            part = part.strip().lower()
            if part not in ['ctrl', 'alt', 'shift', 'win', 'left windows', 'right windows']:
                return part
        return ""
    
    def _build_hotkey_string(self) -> str:
        """Baut Hotkey-String aus aktuellen Einstellungen"""
        modifiers = []
        if self.modifier_ctrl.isChecked():
            modifiers.append('ctrl')
        if self.modifier_alt.isChecked():
            modifiers.append('alt')
        if self.modifier_shift.isChecked():
            modifiers.append('shift')
        if self.modifier_win.isChecked():
            modifiers.append('win')
        
        main_key = self.main_key_input.text().strip()
        
        # Ignoriere Platzhaltertext
        if main_key in ["Warte auf Eingabe...", "warte auf eingabe...", ""]:
            main_key = ""
        
        if main_key:
            if modifiers:
                return '+'.join(modifiers) + '+' + main_key
            else:
                return main_key
        elif modifiers:
            return '+'.join(modifiers) + '+?'
        else:
            return ""
    
    def update_hotkey_preview(self):
        """Aktualisiert Hotkey-Vorschau"""
        preview = self._build_hotkey_string()
        if preview:
            self.hotkey_preview.setText(f"Vorschau: {preview}")
        else:
            self.hotkey_preview.setText("Vorschau: (keine Auswahl)")
    
    def start_main_key_detection(self):
        """Startet Erkennung der Haupttaste"""
        if self.detecting_main_key:
            return
        
        self.detecting_main_key = True
        self.detect_main_key_btn.setText("Drücke Taste...")
        self.detect_main_key_btn.setEnabled(False)
        self.main_key_input.setText("Warte auf Eingabe...")
        self.main_key_input.setStyleSheet("background-color: #ffeb3b;")
        
        # Starte Tastenerkennung in separatem Thread
        import threading
        
        def detect_key_thread():
            """Thread für Tastenerkennung"""
            try:
                import keyboard
                
                self.detected_main_key = None
                self.key_hook = None
                
                def on_key_press(event):
                    """Wird aufgerufen wenn eine Taste gedrückt wird"""
                    try:
                        if self.detected_main_key is not None:
                            return  # Bereits erkannt
                        
                        key_name = event.name.lower()
                        logger.info(f"Taste gedrückt: {key_name}")
                        
                        # Ignoriere Modifier-Keys
                        modifier_keys = [
                            'ctrl', 'alt', 'shift', 'win',
                            'left windows', 'right windows',
                            'left ctrl', 'right ctrl',
                            'left alt', 'right alt',
                            'left shift', 'right shift'
                        ]
                        
                        if key_name in modifier_keys:
                            logger.debug(f"Ignoriere Modifier-Key: {key_name}")
                            return
                        
                        # Normalisiere Tastennamen
                        key_name = self._normalize_key_name(key_name)
                        
                        logger.info(f"Taste erkannt: {key_name}")
                        self.detected_main_key = key_name
                        
                        # Beende Erkennung sofort
                        try:
                            if self.key_hook:
                                keyboard.unhook(self.key_hook)
                        except:
                            pass
                        
                        # Aktualisiere UI im Hauptthread über Signal
                        logger.info(f"Sende Signal für Taste: {key_name}")
                        # Verwende PyQt-Signal für Thread-sichere Kommunikation
                        self.key_detected.emit(key_name)
                        logger.info("Signal gesendet")
                        
                    except Exception as e:
                        logger.error(f"Fehler in on_key_press: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Registriere Hook
                logger.info("Registriere Keyboard-Hook...")
                self.key_hook = keyboard.on_press(on_key_press)
                logger.info("Keyboard-Hook registriert")
                
                # Warte max 10 Sekunden
                import time
                start_time = time.time()
                while time.time() - start_time < 10:
                    if self.detected_main_key is not None:
                        break
                    time.sleep(0.1)
                
                # Timeout oder keine Taste erkannt
                if self.detected_main_key is None:
                    try:
                        if self.key_hook:
                            keyboard.unhook(self.key_hook)
                    except:
                        pass
                    # Verwende Signal für Thread-sichere Kommunikation
                    self.key_detected.emit("")
                    
            except Exception as e:
                logger.error(f"Fehler bei Tastenerkennung: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    import keyboard
                    if hasattr(self, 'key_hook') and self.key_hook:
                        keyboard.unhook(self.key_hook)
                except:
                    pass
                # Verwende Signal für Thread-sichere Kommunikation
                self.key_detected.emit("")
        
        threading.Thread(target=detect_key_thread, daemon=True).start()
    
    def _normalize_key_name(self, key_name: str) -> str:
        """Normalisiert Tastennamen"""
        # Normalisiere häufige Tastennamen
        key_map = {
            'return': 'enter',
            'space': 'space',
            'backspace': 'backspace',
            'tab': 'tab',
            'escape': 'esc',
            'delete': 'delete',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'page up': 'page up',
            'page down': 'page down',
            'home': 'home',
            'end': 'end',
        }
        return key_map.get(key_name, key_name)
    
    def finish_main_key_detection(self, key: str):
        """Beendet Tastenerkennung und aktualisiert UI (Slot für Signal)"""
        # Konvertiere leeren String zu None
        if not key or key == "":
            key = None
        
        logger.info(f"finish_main_key_detection aufgerufen mit key: {key}")
        
        try:
            import keyboard
            if hasattr(self, 'key_hook') and self.key_hook:
                try:
                    keyboard.unhook(self.key_hook)
                    logger.info("Keyboard-Hook entfernt")
                except:
                    pass
            # Entferne alle Hooks sicherheitshalber
            try:
                keyboard.unhook_all()
            except:
                pass
        except Exception as e:
            logger.warning(f"Fehler beim Entfernen der Hooks: {e}")
        
        # Aktualisiere UI
        self.detecting_main_key = False
        logger.info("Setze detecting_main_key auf False")
        
        self.detect_main_key_btn.setText("🔍 Erkennen")
        self.detect_main_key_btn.setEnabled(True)
        logger.info("Button aktualisiert")
        
        self.main_key_input.setStyleSheet("")
        logger.info("StyleSheet zurückgesetzt")
        
        if key:
            self.main_key_input.setText(key)
            logger.info(f"✓ Haupttaste gesetzt: {key}")
            self.update_hotkey_preview()
            logger.info("Hotkey-Vorschau aktualisiert")
        else:
            # Setze zurück auf vorherigen Wert
            current_hotkey = self.config.get("hotkey", "alt+enter")
            main_key = self._extract_main_key(current_hotkey)
            self.main_key_input.setText(main_key)
            logger.warning("Keine Taste erkannt - setze zurück")
            QMessageBox.warning(self, "Tastenerkennung", "Keine Taste erkannt. Bitte versuchen Sie es erneut.")
    
    def save_and_close(self):
        """Speichert und schließt"""
        hotkey = self._build_hotkey_string().lower()
        
        # Prüfe ob Hotkey gültig ist
        if not hotkey or "warte auf eingabe" in hotkey or hotkey.endswith("+?"):
            QMessageBox.warning(
                self,
                "Ungültige Tastenkombination",
                "Bitte wählen Sie eine gültige Tastenkombination aus.\n"
                "Mindestens eine Modifier-Taste (Ctrl, Alt, Shift, Win) und eine Haupttaste müssen ausgewählt sein."
            )
            return
        
        self.config["hotkey"] = hotkey
        self.config["recording_mode"] = {
            0: "toggle",
            1: "push_to_talk",
            2: "auto"
        }[self.mode_combo.currentIndex()]
        self.config["threshold"] = self.threshold_slider.value() / 100.0
        self.config["language"] = self.language_combo.currentData() or ""
        self.config["auto_start"] = self.auto_start_check.isChecked()
        self.save_config()
        self.stop_audio_level_monitoring()
        self.close()
    
    def closeEvent(self, event):
        """Wird aufgerufen wenn Fenster geschlossen wird"""
        # Speichere Schwellenwert automatisch beim Schließen
        self.config["threshold"] = self.threshold_slider.value() / 100.0
        try:
            self.save_config()
            logger.info("Schwellenwert automatisch gespeichert beim Schließen")
        except Exception as e:
            logger.warning(f"Fehler beim automatischen Speichern: {e}")
        
        self.stop_audio_level_monitoring()
        event.accept()
    
    def start_audio_level_monitoring(self):
        """Startet Audio-Level-Monitoring für Visualisierung"""
        try:
            import sounddevice as sd
            import numpy as np
            
            def audio_callback(indata, frames, time_info, status):
                """Callback für Audio-Level-Messung"""
                if status:
                    logger.debug(f"Audio-Status: {status}")
                
                # Berechne RMS (Root Mean Square) für Level
                audio_chunk = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                # Konvertiere zu 0-100% (RMS kann bis ~0.7 gehen bei normaler Lautstärke)
                # Normalisiere auf 0-1, dann auf 0-100
                normalized_level = min(1.0, rms / 0.7)  # 0.7 ist ungefähr maximale RMS
                self.current_audio_level = normalized_level * 100
            
            # Starte Stream für Level-Monitoring
            self.audio_level_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=1024
            )
            self.audio_level_stream.start()
            
            # Starte Timer für UI-Updates (alle 50ms)
            self.level_monitor_timer.start(50)
            
            logger.info("Audio-Level-Monitoring gestartet")
            
        except Exception as e:
            logger.error(f"Fehler beim Starten des Audio-Level-Monitorings: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def stop_audio_level_monitoring(self):
        """Stoppt Audio-Level-Monitoring"""
        try:
            if self.level_monitor_timer.isActive():
                self.level_monitor_timer.stop()
            
            if self.audio_level_stream:
                self.audio_level_stream.stop()
                self.audio_level_stream.close()
                self.audio_level_stream = None
            
            logger.info("Audio-Level-Monitoring gestoppt")
            
        except Exception as e:
            logger.warning(f"Fehler beim Stoppen des Audio-Level-Monitorings: {e}")
    
    def update_audio_level_display(self):
        """Aktualisiert Audio-Level-Anzeige mit Schwellenwert-Markierung"""
        try:
            level = int(self.current_audio_level)
            if hasattr(self, 'audio_level_bar'):
                self.audio_level_bar.setValue(level)
                
                # Ändere Farbe basierend auf Level und Schwellenwert
                threshold = self.threshold_slider.value() if hasattr(self, 'threshold_slider') else 50
                
                if level < threshold:
                    if level < threshold * 0.5:
                        color = "#4CAF50"  # Grün
                    else:
                        color = "#FFC107"  # Gelb
                else:
                    if level < 80:
                        color = "#FF9800"  # Orange
                    else:
                        color = "#F44336"  # Rot
                
                self.audio_level_bar.setStyleSheet(f"""
                    QProgressBar {{
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        text-align: center;
                    }}
                    QProgressBar::chunk {{
                        background-color: {color};
                        border-radius: 3px;
                    }}
                """)
            
        except Exception as e:
            logger.debug(f"Fehler beim Aktualisieren der Audio-Level-Anzeige: {e}")


class SpeechInputApp:
    """Hauptanwendung"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        
        # Pfade
        self.base_path = Path(__file__).parent
        self.config_path = self.base_path / "config.json"
        self.icon_path = self.base_path / "icons" / "microphone.ico"
        
        # Konfiguration
        self.config = self.load_config()
        
        # Komponenten
        self.recorder = AudioRecorder()
        self.recorder.on_recording_state_changed = self.on_recording_state_changed
        self.hotkey_handler = HotkeyHandler()
        self.api_client = APIClient(self.config.get("api_base", "http://127.0.0.1:8000"))
        self.text_injector = TextInjector()
        
        # Thread für Transkription
        self.transcription_thread: Optional[TranscriptionThread] = None
        
        # Speichere aktives Fenster-Handle für Text-Injection
        self.active_window_handle = None
        
        # System-Tray
        self.tray_icon = QSystemTrayIcon()
        
        # Prüfe ob System-Tray verfügbar ist
        if not QSystemTrayIcon.isSystemTrayAvailable():
            QMessageBox.critical(None, "System Tray", 
                               "System Tray ist nicht verfügbar auf diesem System.")
            sys.exit(1)
        
        self.create_icon()
        self.tray_icon.setToolTip("Speech Input - Rechtsklick für Menü")
        
        # Menü
        self.create_menu()
        self.tray_icon.show()
        
        # Setup
        self.setup_recording_mode()
        self.setup_hotkey()
    
    def load_config(self) -> dict:
        """Lädt Konfiguration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return {
            "hotkey": "alt+enter",
            "recording_mode": "toggle",
            "threshold": 0.5,
            "language": "",
            "auto_start": False,
            "api_base": "http://127.0.0.1:8000"
        }
    
    def create_icon(self):
        """Erstellt Icon falls nicht vorhanden"""
        # Stelle sicher, dass Icon-Verzeichnis existiert
        self.icon_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Erstelle Icon direkt aus QPixmap (funktioniert besser als ICO-Datei)
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Mikrofon zeichnen
        painter.setBrush(QColor(74, 158, 255))  # Blau
        painter.setPen(QColor(50, 120, 200))
        painter.drawEllipse(8, 4, 16, 20)  # Mikrofon-Körper
        painter.drawRect(14, 24, 4, 4)  # Stand
        
        painter.end()
        
        # Erstelle QIcon direkt aus QPixmap
        icon = QIcon(pixmap)
        self.tray_icon.setIcon(icon)
        
        # Speichere auch als PNG für später (optional)
        if not self.icon_path.exists():
            png_path = self.icon_path.with_suffix('.png')
            pixmap.save(str(png_path), "PNG")
            logger.info(f"Icon erstellt: {png_path}")
    
    def create_menu(self):
        """Erstellt System-Tray-Menü"""
        menu = QMenu()
        
        # Status
        self.status_action = menu.addAction("Status: Bereit")
        self.status_action.setEnabled(False)
        
        menu.addSeparator()
        
        # Aufnahme starten/stoppen
        self.record_action = menu.addAction("🎤 Aufnahme starten")
        self.record_action.triggered.connect(self.toggle_recording)
        
        menu.addSeparator()
        
        # Einstellungen
        settings_action = menu.addAction("⚙️ Einstellungen")
        settings_action.triggered.connect(self.show_settings)
        
        # Beenden
        menu.addSeparator()
        quit_action = menu.addAction("Beenden")
        quit_action.triggered.connect(self.quit)
        
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
    
    def on_tray_activated(self, reason):
        """Wird aufgerufen wenn Tray-Icon aktiviert wird"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_settings()
    
    def setup_recording_mode(self):
        """Setzt Aufnahme-Modus"""
        mode_str = self.config.get("recording_mode", "toggle")
        mode = RecordingMode(mode_str)
        self.recorder.set_mode(mode)
        self.recorder.set_threshold(self.config.get("threshold", 0.5))
    
    def setup_hotkey(self):
        """Setzt Hotkey"""
        hotkey = self.config.get("hotkey", "alt+enter")
        
        # Validiere Hotkey - entferne ungültige Werte
        if not hotkey or "warte auf eingabe" in hotkey.lower() or hotkey.endswith("+?") or hotkey == "":
            logger.warning(f"Ungültiger Hotkey '{hotkey}', verwende Standard: alt+enter")
            hotkey = "alt+enter"
            self.config["hotkey"] = hotkey
            # Speichere korrigierten Hotkey
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            except:
                pass
        
        mode_str = self.config.get("recording_mode", "toggle")
        
        try:
            if mode_str == "toggle":
                self.hotkey_handler.set_hotkey(hotkey, callback=self.toggle_recording)
            elif mode_str == "push_to_talk":
                self.hotkey_handler.set_hotkey(
                    hotkey,
                    press_callback=self.start_recording,
                    release_callback=self.stop_recording
                )
            elif mode_str == "auto":
                self.hotkey_handler.set_hotkey(hotkey, callback=self.toggle_recording)
        except Exception as e:
            logger.error(f"Fehler beim Registrieren des Hotkeys: {e}")
            # Versuche mit Standard-Hotkey
            try:
                self.hotkey_handler.set_hotkey("alt+enter", callback=self.toggle_recording)
                logger.info("Verwende Standard-Hotkey: alt+enter")
            except:
                logger.error("Konnte auch Standard-Hotkey nicht registrieren")
    
    def toggle_recording(self):
        """Toggle-Aufnahme"""
        if self.recorder.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Startet Aufnahme"""
        # WICHTIG: Speichere aktives Fenster-Handle VOR der Aufnahme
        # Das ist das Fenster, in das später der Text eingefügt werden soll
        try:
            import win32gui
            self.active_window_handle = win32gui.GetForegroundWindow()
            if self.active_window_handle:
                # Hole auch Fenster-Titel für Logging
                try:
                    window_title = win32gui.GetWindowText(self.active_window_handle)
                    logger.info(f"Aktives Fenster gespeichert: {window_title} (Handle: {self.active_window_handle})")
                except:
                    logger.info(f"Aktives Fenster gespeichert (Handle: {self.active_window_handle})")
            else:
                logger.warning("Kein aktives Fenster gefunden")
        except Exception as e:
            self.active_window_handle = None
            logger.warning(f"Fehler beim Speichern des Fenster-Handles: {e}")
        
        if self.recorder.start_recording():
            self.record_action.setText("⏹️ Aufnahme stoppen")
            self.status_action.setText("Status: Aufnahme läuft...")
    
    def stop_recording(self):
        """Stoppt Aufnahme und startet Transkription"""
        wav_file = self.recorder.stop_recording()
        if wav_file:
            self.record_action.setText("🎤 Aufnahme starten")
            self.status_action.setText("Status: Transkribiere...")
            self.start_transcription(wav_file)
        else:
            self.record_action.setText("🎤 Aufnahme starten")
            self.status_action.setText("Status: Bereit")
    
    def start_transcription(self, wav_file: str):
        """Startet Transkription in separatem Thread"""
        language = self.config.get("language", "")
        if not language:
            language = None
        
        logger.info(f"[start_transcription] Erstelle TranscriptionThread für: {wav_file}")
        self.transcription_thread = TranscriptionThread(
            wav_file,
            self.api_client,
            language,
            callback=self.on_transcription_finished,  # Direkter Callback als Fallback
            app_instance=self.app  # QApplication-Instanz für Thread-sichere Aufrufe
        )
        
        logger.info("[start_transcription] Verbinde Signale...")
        
        # Test: Prüfe ob Signal existiert
        if not hasattr(self.transcription_thread, 'finished'):
            logger.error("[start_transcription] FEHLER: Signal 'finished' existiert nicht!")
        else:
            logger.info("[start_transcription] Signal 'finished' gefunden")
        
        # Verbinde Signale - WICHTIG: QueuedConnection für Thread-sichere Kommunikation
        from PyQt6.QtCore import Qt
        self.transcription_thread.finished.connect(
            self.on_transcription_finished,
            Qt.ConnectionType.QueuedConnection
        )
        self.transcription_thread.error.connect(
            self.on_transcription_error,
            Qt.ConnectionType.QueuedConnection
        )
        logger.info("[start_transcription] Signale verbunden (QueuedConnection)")
        
        logger.info("[start_transcription] Starte Thread...")
        self.transcription_thread.start()
        logger.info("[start_transcription] Thread gestartet")
    
    def on_transcription_finished(self, text: str):
        """Wird aufgerufen wenn Transkription fertig ist"""
        logger.info("=" * 60)
        logger.info(f"🎯 on_transcription_finished aufgerufen mit Text: '{text}'")
        logger.info("=" * 60)
        
        if not text or not text.strip():
            logger.error("❌ Leerer Text erhalten!")
            return
        
        # Setze Text ZUERST in Clipboard (unabhängig von Injection)
        clipboard_success = False
        try:
            import win32clipboard
            logger.info("Öffne Clipboard...")
            win32clipboard.OpenClipboard()
            logger.info("Leere Clipboard...")
            win32clipboard.EmptyClipboard()
            logger.info(f"Setze Text in Clipboard: '{text[:50]}...'")
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
            logger.info("✓ Clipboard geschlossen")
            
            # VERIFIZIERE: Prüfe ob Text wirklich im Clipboard ist
            try:
                win32clipboard.OpenClipboard()
                clipboard_text = win32clipboard.GetClipboardData()
                win32clipboard.CloseClipboard()
                if clipboard_text == text:
                    logger.info(f"✅ VERIFIZIERT: Text ist im Clipboard: '{clipboard_text[:50]}...'")
                    clipboard_success = True
                else:
                    logger.error(f"❌ FEHLER: Clipboard-Inhalt stimmt nicht! Erwartet: '{text[:30]}...', Gefunden: '{clipboard_text[:30] if clipboard_text else 'None'}...'")
            except Exception as e:
                logger.error(f"❌ Fehler bei Clipboard-Verifizierung: {e}")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Setzen des Clipboards: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        if not clipboard_success:
            logger.error("❌ Clipboard konnte nicht gesetzt werden - ABBRUCH")
            self.status_action.setText("Status: Fehler - Clipboard konnte nicht gesetzt werden")
            return
        
        # Führe Text-Injection SOFORT aus (keine Verzögerung)
        logger.info(f"🚀 Starte Text-Injection: '{text[:50]}...'")
        logger.info(f"Fenster-Handle: {self.active_window_handle}")
        
        # Füge Text ein - nutze gespeichertes Fenster-Handle falls verfügbar
        success = self.text_injector.inject_text(text, self.active_window_handle)
        
        if success:
            logger.info("✅✅✅ Text-Injection erfolgreich ✅✅✅")
            self.status_action.setText(f"Status: Text eingefügt: {text[:30]}...")
        else:
            logger.error("❌❌❌ Text-Injection fehlgeschlagen ❌❌❌")
            self.status_action.setText("Status: Fehler - Text im Clipboard (STRG+V)")
        
        # Reset nach 3 Sekunden
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: self.status_action.setText("Status: Bereit"))
    
    def on_transcription_error(self, error: str):
        """Wird aufgerufen bei Transkriptions-Fehler"""
        logger.error(f"Transkriptions-Fehler: {error}")
        self.status_action.setText(f"Status: Fehler - {error}")
        QMessageBox.warning(None, "Transkriptions-Fehler", error)
        
        # Reset nach 3 Sekunden
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: self.status_action.setText("Status: Bereit"))
    
    def on_recording_state_changed(self, is_recording: bool):
        """Wird aufgerufen wenn Aufnahme-Status sich ändert"""
        if is_recording:
            self.tray_icon.setToolTip("Speech Input - Aufnahme läuft")
        else:
            self.tray_icon.setToolTip("Speech Input - Bereit")
    
    def show_settings(self):
        """Zeigt Einstellungsfenster"""
        def on_config_changed(new_config):
            self.config = new_config
            self.setup_recording_mode()
            self.setup_hotkey()
        
        settings = ConfigWindow(self.config_path, on_config_changed)
        settings.show()
    
    def quit(self):
        """Beendet Anwendung"""
        self.recorder.cleanup()
        self.hotkey_handler.cleanup()
        self.app.quit()
    
    def run(self):
        """Startet Anwendung"""
        sys.exit(self.app.exec())


def main():
    """Hauptfunktion"""
    app = SpeechInputApp()
    app.run()


if __name__ == "__main__":
    main()

