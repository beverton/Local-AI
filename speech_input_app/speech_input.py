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
        self.setFixedSize(900, 700)  # Breiter, aber nicht so hoch
        
        # Haupt-Layout mit 2 Spalten
        main_layout = QHBoxLayout()
        
        # Linke Spalte
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        
        # Rechte Spalte
        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        
        layout = left_column  # Temporär für bestehenden Code
        
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
        left_column.addWidget(hotkey_group)
        
        # Aufnahme-Modus
        mode_group = QGroupBox("Aufnahme-Modus")
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Toggle", "Push-to-Talk", "Auto", "Continuous (Segment-basiert)"])
        current_mode = self.config.get("recording_mode", "toggle")
        mode_index = {"toggle": 0, "push_to_talk": 1, "auto": 2, "continuous": 3}.get(current_mode, 0)
        self.mode_combo.setCurrentIndex(mode_index)
        mode_layout.addWidget(QLabel("Modus:"))
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        left_column.addWidget(mode_group)
        
        # Verbinde Signal für Modus-Änderung
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        
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
        # Wird dynamisch angepasst
        
        # Schwellenwert-Anzeige als Overlay (über dem Audio-Level-Balken)
        self.threshold_display_bar = QProgressBar(level_container)
        self.threshold_display_bar.setMinimum(0)
        self.threshold_display_bar.setMaximum(100)
        initial_threshold_value = int(self.config.get("threshold", 0.5) * 100)
        self.threshold_display_bar.setValue(initial_threshold_value)
        self.threshold_display_bar.setFormat("")
        self.threshold_display_bar.setTextVisible(False)
        # Wird dynamisch angepasst
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
        left_column.addWidget(threshold_group)
        
        # CONTINUOUS-Modus Einstellungen (Pause-Erkennung) - in rechte Spalte
        continuous_group = QGroupBox("Segment-basierte Transkription (Continuous-Modus)")
        continuous_layout = QVBoxLayout()
        
        # Silence Threshold
        silence_threshold_layout = QHBoxLayout()
        silence_threshold_layout.addWidget(QLabel("Stille-Schwellenwert:"))
        self.silence_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_threshold_slider.setMinimum(1)  # 0.001
        self.silence_threshold_slider.setMaximum(100)  # 0.1
        initial_silence_threshold = int(self.config.get("silence_threshold", 0.02) * 1000)
        self.silence_threshold_slider.setValue(initial_silence_threshold)
        self.silence_threshold_label = QLabel(f"{self.config.get('silence_threshold', 0.02):.3f}")
        self.silence_threshold_label.setMinimumWidth(60)
        self.silence_threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_silence_threshold(v):
            threshold_value = v / 1000.0
            self.silence_threshold_label.setText(f"{threshold_value:.3f}")
        
        self.silence_threshold_slider.valueChanged.connect(update_silence_threshold)
        silence_threshold_layout.addWidget(self.silence_threshold_slider)
        silence_threshold_layout.addWidget(self.silence_threshold_label)
        continuous_layout.addLayout(silence_threshold_layout)
        
        # Info für Silence Threshold
        silence_threshold_info = QLabel("💡 Niedrigere Werte = empfindlicher (erkennt leise Sprache besser, aber auch mehr Hintergrundgeräusche)")
        silence_threshold_info.setWordWrap(True)
        silence_threshold_info.setStyleSheet("font-size: 9pt; color: #666;")
        continuous_layout.addWidget(silence_threshold_info)
        
        continuous_layout.addWidget(QLabel(""))  # Abstand
        
        # Silence Duration
        silence_duration_layout = QHBoxLayout()
        silence_duration_layout.addWidget(QLabel("Pause-Dauer (Sekunden):"))
        self.silence_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.silence_duration_slider.setMinimum(5)  # 0.5 Sekunden
        self.silence_duration_slider.setMaximum(50)  # 5.0 Sekunden
        initial_silence_duration = int(self.config.get("silence_duration", 1.0) * 10)
        self.silence_duration_slider.setValue(initial_silence_duration)
        self.silence_duration_label = QLabel(f"{self.config.get('silence_duration', 1.0):.1f}s")
        self.silence_duration_label.setMinimumWidth(50)
        self.silence_duration_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_silence_duration(v):
            duration_value = v / 10.0
            self.silence_duration_label.setText(f"{duration_value:.1f}s")
        
        self.silence_duration_slider.valueChanged.connect(update_silence_duration)
        silence_duration_layout.addWidget(self.silence_duration_slider)
        silence_duration_layout.addWidget(self.silence_duration_label)
        continuous_layout.addLayout(silence_duration_layout)
        
        # Info für Silence Duration
        silence_duration_info = QLabel("💡 Wie lange Stille erkannt werden muss, bevor ein Segment transkribiert wird. Längere Werte = weniger Segment-Unterbrechungen")
        silence_duration_info.setWordWrap(True)
        silence_duration_info.setStyleSheet("font-size: 9pt; color: #666;")
        continuous_layout.addWidget(silence_duration_info)
        
        continuous_layout.addWidget(QLabel(""))  # Abstand
        continuous_layout.addWidget(QLabel("─" * 50))  # Trennlinie
        continuous_layout.addWidget(QLabel(""))  # Abstand
        
        # Audio-Verarbeitung
        audio_processing_label = QLabel("Audio-Verarbeitung (Experimentell)")
        audio_processing_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        continuous_layout.addWidget(audio_processing_label)
        
        # Clipping-Schutz aktivieren/deaktivieren
        self.clipping_protection_checkbox = QCheckBox("Clipping-Schutz aktivieren")
        clipping_protection_enabled = self.config.get("clipping_protection", True)  # Standard: aktiviert
        self.clipping_protection_checkbox.setChecked(clipping_protection_enabled)
        continuous_layout.addWidget(self.clipping_protection_checkbox)
        
        # Info für Clipping-Schutz
        clipping_info = QLabel("💡 Verhindert Audio-Clipping (Übersteuerung). Sollte normalerweise aktiviert bleiben.")
        clipping_info.setWordWrap(True)
        clipping_info.setStyleSheet("font-size: 9pt; color: #666;")
        continuous_layout.addWidget(clipping_info)
        
        continuous_layout.addWidget(QLabel(""))  # Abstand
        
        # Normalisierung aktivieren/deaktivieren
        self.normalize_checkbox = QCheckBox("Normalisierung aktivieren")
        normalize_enabled = self.config.get("normalize_audio", True)  # Standard: aktiviert
        self.normalize_checkbox.setChecked(normalize_enabled)
        continuous_layout.addWidget(self.normalize_checkbox)
        
        # Normalisierungs-Level
        normalize_level_layout = QHBoxLayout()
        normalize_level_layout.addWidget(QLabel("Normalisierungs-Level:"))
        self.normalize_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.normalize_level_slider.setMinimum(50)  # 0.5
        self.normalize_level_slider.setMaximum(100)  # 1.0
        initial_normalize_level = int(self.config.get("normalize_level", 1.0) * 100)
        self.normalize_level_slider.setValue(initial_normalize_level)
        self.normalize_level_label = QLabel(f"{self.config.get('normalize_level', 1.0):.2f}")
        self.normalize_level_label.setMinimumWidth(50)
        self.normalize_level_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_normalize_level(v):
            level_value = v / 100.0
            self.normalize_level_label.setText(f"{level_value:.2f}")
        
        self.normalize_level_slider.valueChanged.connect(update_normalize_level)
        normalize_level_layout.addWidget(self.normalize_level_slider)
        normalize_level_layout.addWidget(self.normalize_level_label)
        continuous_layout.addLayout(normalize_level_layout)
        
        # Info für Normalisierung
        normalize_info = QLabel("💡 Normalisiert Audio auf einheitliches Level. 1.0 = maximale Lautstärke, niedrigere Werte = mehr Headroom")
        normalize_info.setWordWrap(True)
        normalize_info.setStyleSheet("font-size: 9pt; color: #666;")
        continuous_layout.addWidget(normalize_info)
        
        continuous_layout.addWidget(QLabel(""))  # Abstand
        
        # Kompression aktivieren/deaktivieren
        self.compress_checkbox = QCheckBox("Kompression aktivieren")
        compress_enabled = self.config.get("compress_audio", False)  # Standard: deaktiviert
        self.compress_checkbox.setChecked(compress_enabled)
        continuous_layout.addWidget(self.compress_checkbox)
        
        # Kompressions-Ratio
        compress_ratio_layout = QHBoxLayout()
        compress_ratio_layout.addWidget(QLabel("Kompressions-Ratio:"))
        self.compress_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.compress_ratio_slider.setMinimum(10)  # 1.0 (keine Kompression)
        self.compress_ratio_slider.setMaximum(100)  # 10.0 (starke Kompression)
        initial_compress_ratio = int(self.config.get("compress_ratio", 2.0) * 10)
        self.compress_ratio_slider.setValue(initial_compress_ratio)
        self.compress_ratio_label = QLabel(f"{self.config.get('compress_ratio', 2.0):.1f}:1")
        self.compress_ratio_label.setMinimumWidth(50)
        self.compress_ratio_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_compress_ratio(v):
            ratio_value = v / 10.0
            self.compress_ratio_label.setText(f"{ratio_value:.1f}:1")
        
        self.compress_ratio_slider.valueChanged.connect(update_compress_ratio)
        compress_ratio_layout.addWidget(self.compress_ratio_slider)
        compress_ratio_layout.addWidget(self.compress_ratio_label)
        continuous_layout.addLayout(compress_ratio_layout)
        
        # Kompressions-Threshold
        compress_threshold_layout = QHBoxLayout()
        compress_threshold_layout.addWidget(QLabel("Kompressions-Threshold:"))
        self.compress_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.compress_threshold_slider.setMinimum(10)  # 0.1
        self.compress_threshold_slider.setMaximum(100)  # 1.0
        initial_compress_threshold = int(self.config.get("compress_threshold", 0.5) * 100)
        self.compress_threshold_slider.setValue(initial_compress_threshold)
        self.compress_threshold_label = QLabel(f"{self.config.get('compress_threshold', 0.5):.2f}")
        self.compress_threshold_label.setMinimumWidth(50)
        self.compress_threshold_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        def update_compress_threshold(v):
            threshold_value = v / 100.0
            self.compress_threshold_label.setText(f"{threshold_value:.2f}")
        
        self.compress_threshold_slider.valueChanged.connect(update_compress_threshold)
        compress_threshold_layout.addWidget(self.compress_threshold_slider)
        compress_threshold_layout.addWidget(self.compress_threshold_label)
        continuous_layout.addLayout(compress_threshold_layout)
        
        # Info für Kompression
        compress_info = QLabel("💡 Kompression reduziert dynamische Range. Höhere Ratio = stärkere Kompression. Threshold = ab welchem Level komprimiert wird")
        compress_info.setWordWrap(True)
        compress_info.setStyleSheet("font-size: 9pt; color: #666;")
        continuous_layout.addWidget(compress_info)
        
        continuous_group.setLayout(continuous_layout)
        right_column.addWidget(continuous_group)
        
        # Speichere Referenz für Sichtbarkeit
        self.continuous_group = continuous_group
        
        # Initialisiere Sichtbarkeit basierend auf aktuellem Modus
        self.on_mode_changed()
        
        # Initialisiere Audio-Level-Monitoring
        self.audio_level_stream = None
        self.current_audio_level = 0.0
        self.level_monitor_timer = QTimer()
        self.level_monitor_timer.timeout.connect(self.update_audio_level_display)
        self.start_audio_level_monitoring()
        
        # Sprache - in rechte Spalte
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
        right_column.addWidget(language_group)
        
        # Auto-Start - in rechte Spalte
        auto_start_check = QCheckBox("Beim Windows-Start starten")
        auto_start_check.setChecked(self.config.get("auto_start", False))
        self.auto_start_check = auto_start_check
        right_column.addWidget(auto_start_check)
        
        # Stretch für rechte Spalte
        right_column.addStretch()
        
        # Füge beide Spalten zum Haupt-Layout hinzu
        left_widget = QWidget()
        left_widget.setLayout(left_column)
        right_widget = QWidget()
        right_widget.setLayout(right_column)
        
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)
        
        # Buttons unten zentriert
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        save_btn = QPushButton("Speichern")
        save_btn.clicked.connect(self.save_and_close)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        
        # Haupt-Layout mit Buttons
        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(button_layout)
        
        self.setLayout(final_layout)
    
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
            2: "auto",
            3: "continuous"
        }[self.mode_combo.currentIndex()]
        self.config["threshold"] = self.threshold_slider.value() / 100.0
        self.config["language"] = self.language_combo.currentData() or ""
        self.config["auto_start"] = self.auto_start_check.isChecked()
        
        # Für CONTINUOUS-Modus: Speichere Einstellungen
        if self.config["recording_mode"] == "continuous":
            self.config["silence_threshold"] = self.silence_threshold_slider.value() / 1000.0
            self.config["silence_duration"] = self.silence_duration_slider.value() / 10.0
            # Audio-Verarbeitung
            self.config["clipping_protection"] = self.clipping_protection_checkbox.isChecked()
            self.config["normalize_audio"] = self.normalize_checkbox.isChecked()
            self.config["normalize_level"] = self.normalize_level_slider.value() / 100.0
            self.config["compress_audio"] = self.compress_checkbox.isChecked()
            self.config["compress_ratio"] = self.compress_ratio_slider.value() / 10.0
            self.config["compress_threshold"] = self.compress_threshold_slider.value() / 100.0
        else:
            # Setze Standardwerte falls nicht vorhanden (für später)
            if "silence_threshold" not in self.config:
                self.config["silence_threshold"] = 0.02
            if "silence_duration" not in self.config:
                self.config["silence_duration"] = 1.0
        
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
    
    def on_mode_changed(self):
        """Wird aufgerufen wenn Aufnahme-Modus geändert wird"""
        current_index = self.mode_combo.currentIndex()
        is_continuous = (current_index == 3)  # Continuous-Modus
        
        # Zeige/Verstecke CONTINUOUS-Einstellungen
        if hasattr(self, 'continuous_group'):
            self.continuous_group.setVisible(is_continuous)
    
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
        self.recorder.on_segment_ready = self.on_segment_ready  # Callback für Segmente
        self.hotkey_handler = HotkeyHandler()
        self.api_client = APIClient(self.config.get("api_base", "http://127.0.0.1:8000"))
        self.text_injector = TextInjector()
        
        # Thread für Transkription
        self.transcription_thread: Optional[TranscriptionThread] = None
        self.active_transcriptions = set()  # Set von laufenden Transkriptions-Threads
        
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
        
        # Für CONTINUOUS-Modus: Setze Pause-Erkennung und Audio-Verarbeitung
        if mode == RecordingMode.CONTINUOUS:
            self.recorder.set_silence_threshold(self.config.get("silence_threshold", 0.01))
            self.recorder.set_silence_duration(self.config.get("silence_duration", 1.0))
            # Audio-Verarbeitung
            self.recorder.set_clipping_protection(self.config.get("clipping_protection", True))
            self.recorder.set_normalize_audio(self.config.get("normalize_audio", True))
            self.recorder.set_normalize_level(self.config.get("normalize_level", 1.0))
            self.recorder.set_compress_audio(self.config.get("compress_audio", False))
            self.recorder.set_compress_ratio(self.config.get("compress_ratio", 2.0))
            self.recorder.set_compress_threshold(self.config.get("compress_threshold", 0.5))
    
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
            elif mode_str == "continuous":
                # CONTINUOUS-Modus: Toggle startet/stoppt kontinuierliche Aufnahme
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
        # Im CONTINUOUS-Modus: Stoppe nur die Aufnahme, keine finale Transkription
        if self.recorder.mode == RecordingMode.CONTINUOUS:
            self.recorder.stop_recording()
            self.record_action.setText("🎤 Aufnahme starten")
            self.status_action.setText("Status: Bereit")
            return
        
        # Für andere Modi: Normale Logik
        wav_file = self.recorder.stop_recording()
        if wav_file:
            self.record_action.setText("🎤 Aufnahme starten")
            self.status_action.setText("Status: Transkribiere...")
            self.start_transcription(wav_file)
        else:
            self.record_action.setText("🎤 Aufnahme starten")
            self.status_action.setText("Status: Bereit")
    
    def start_transcription(self, wav_file: str, is_segment: bool = False):
        """Startet Transkription in separatem Thread"""
        language = self.config.get("language", "")
        if not language:
            language = None
        
        logger.info(f"[start_transcription] Erstelle TranscriptionThread für: {wav_file} (Segment: {is_segment})")
        
        # Erstelle neuen Thread (für Segmente können mehrere parallel laufen)
        transcription_thread = TranscriptionThread(
            wav_file,
            self.api_client,
            language,
            callback=self.on_transcription_finished,  # Direkter Callback als Fallback
            app_instance=self.app  # QApplication-Instanz für Thread-sichere Aufrufe
        )
        
        # Für Segmente: Speichere Thread-ID zur Nachverfolgung
        if is_segment:
            thread_id = id(transcription_thread)
            self.active_transcriptions.add(thread_id)
            logger.info(f"[start_transcription] Segment-Transkription gestartet (Thread-ID: {thread_id})")
        else:
            # Für normale Transkription: Überschreibe alten Thread
            self.transcription_thread = transcription_thread
        
        logger.info("[start_transcription] Verbinde Signale...")
        
        # Test: Prüfe ob Signal existiert
        if not hasattr(transcription_thread, 'finished'):
            logger.error("[start_transcription] FEHLER: Signal 'finished' existiert nicht!")
        else:
            logger.info("[start_transcription] Signal 'finished' gefunden")
        
        # Verbinde Signale - WICHTIG: QueuedConnection für Thread-sichere Kommunikation
        from PyQt6.QtCore import Qt
        transcription_thread.finished.connect(
            lambda text: self.on_transcription_finished(text, is_segment=is_segment, thread_id=id(transcription_thread) if is_segment else None),
            Qt.ConnectionType.QueuedConnection
        )
        transcription_thread.error.connect(
            lambda error: self.on_transcription_error(error, is_segment=is_segment, thread_id=id(transcription_thread) if is_segment else None),
            Qt.ConnectionType.QueuedConnection
        )
        logger.info("[start_transcription] Signale verbunden (QueuedConnection)")
        
        logger.info("[start_transcription] Starte Thread...")
        transcription_thread.start()
        logger.info("[start_transcription] Thread gestartet")
    
    def on_transcription_finished(self, text: str, is_segment: bool = False, thread_id: Optional[int] = None):
        """Wird aufgerufen wenn Transkription fertig ist"""
        logger.info("=" * 60)
        logger.info(f"🎯 on_transcription_finished aufgerufen mit Text: '{text}' (Segment: {is_segment})")
        logger.info("=" * 60)
        
        # Entferne Thread-ID aus aktiven Transkriptionen wenn Segment
        if is_segment and thread_id:
            self.active_transcriptions.discard(thread_id)
            logger.info(f"Segment-Transkription abgeschlossen (Thread-ID: {thread_id})")
        
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
            if not is_segment:  # Nur Status-Update wenn nicht Segment (sonst zu viele Updates)
                self.status_action.setText("Status: Fehler - Clipboard konnte nicht gesetzt werden")
            return
        
        # Führe Text-Injection SOFORT aus (keine Verzögerung)
        logger.info(f"🚀 Starte Text-Injection: '{text[:50]}...'")
        logger.info(f"Fenster-Handle: {self.active_window_handle}")
        
        # Füge Text ein - nutze gespeichertes Fenster-Handle falls verfügbar
        success = self.text_injector.inject_text(text, self.active_window_handle)
        
        if success:
            logger.info("✅✅✅ Text-Injection erfolgreich ✅✅✅")
            if is_segment:
                # Für Segmente: Kurze Status-Update, dann zurück zu "Aufnahme läuft"
                self.status_action.setText(f"Status: Segment eingefügt: {text[:30]}...")
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self.status_action.setText("Status: Aufnahme läuft..."))
            else:
                self.status_action.setText(f"Status: Text eingefügt: {text[:30]}...")
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(3000, lambda: self.status_action.setText("Status: Bereit"))
        else:
            logger.error("❌❌❌ Text-Injection fehlgeschlagen ❌❌❌")
            if is_segment:
                self.status_action.setText("Status: Segment-Fehler - Text im Clipboard (STRG+V)")
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self.status_action.setText("Status: Aufnahme läuft..."))
            else:
                self.status_action.setText("Status: Fehler - Text im Clipboard (STRG+V)")
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(3000, lambda: self.status_action.setText("Status: Bereit"))
    
    def on_transcription_error(self, error: str, is_segment: bool = False, thread_id: Optional[int] = None):
        """Wird aufgerufen bei Transkriptions-Fehler"""
        logger.error(f"Transkriptions-Fehler: {error} (Segment: {is_segment})")
        
        # Entferne Thread-ID aus aktiven Transkriptionen wenn Segment
        if is_segment and thread_id:
            self.active_transcriptions.discard(thread_id)
        
        if not is_segment:
            # Nur für normale Transkriptionen: Zeige Fehler-Dialog
            self.status_action.setText(f"Status: Fehler - {error}")
            QMessageBox.warning(None, "Transkriptions-Fehler", error)
            # Reset nach 3 Sekunden
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self.status_action.setText("Status: Bereit"))
        else:
            # Für Segmente: Kurze Fehlermeldung, dann zurück zu "Aufnahme läuft"
            self.status_action.setText(f"Status: Segment-Fehler - {error}")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.status_action.setText("Status: Aufnahme läuft..."))
    
    def on_segment_ready(self, wav_file: str):
        """Wird aufgerufen wenn ein Segment fertig ist (CONTINUOUS-Modus)"""
        logger.info(f"🎯 Segment bereit: {wav_file}")
        
        # Starte Transkription für dieses Segment
        # WICHTIG: Recording läuft weiter, wir transkribieren parallel
        self.start_transcription(wav_file, is_segment=True)
    
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

