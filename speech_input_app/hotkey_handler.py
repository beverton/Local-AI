"""
Hotkey Handler - Globale Tastenkürzel
"""
import logging
import keyboard
from typing import Optional, Callable, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotkeyHandler:
    """Verwaltet globale Tastenkürzel"""
    
    def __init__(self):
        self.current_hotkey: Optional[str] = None
        self.hotkey_callback: Optional[Callable[[], None]] = None
        self.hotkey_press_callback: Optional[Callable[[], None]] = None
        self.hotkey_release_callback: Optional[Callable[[], None]] = None
        self.is_pressed = False
    
    def set_hotkey(self, hotkey: str, callback: Optional[Callable[[], None]] = None,
                   press_callback: Optional[Callable[[], None]] = None,
                   release_callback: Optional[Callable[[], None]] = None):
        """
        Setzt Tastenkürzel
        
        Args:
            hotkey: Tastenkombination (z.B. "alt+enter", "shift+x")
            callback: Callback für Toggle-Modus (wird bei Drücken aufgerufen)
            press_callback: Callback für Push-to-Talk (wird bei Drücken aufgerufen)
            release_callback: Callback für Push-to-Talk (wird bei Loslassen aufgerufen)
        """
        # Entferne altes Hotkey
        if self.current_hotkey:
            self.remove_hotkey()
        
        self.current_hotkey = hotkey
        self.hotkey_callback = callback
        self.hotkey_press_callback = press_callback
        self.hotkey_release_callback = release_callback
        
        try:
            # Normalisiere Hotkey-String für keyboard-Bibliothek
            normalized_hotkey = self._normalize_hotkey(hotkey)
            
            if press_callback and release_callback:
                # Push-to-Talk: Nutze on_press und on_release
                main_key = self._parse_hotkey(hotkey)
                
                def on_press(event):
                    if self._check_modifiers(hotkey):
                        self.hotkey_press_callback()
                
                def on_release(event):
                    if event.name.lower() == main_key.lower():
                        self.hotkey_release_callback()
                
                keyboard.on_press_key(main_key, on_press, suppress=False)
                keyboard.on_release_key(main_key, on_release, suppress=False)
            else:
                # Toggle-Modus: Nutze add_hotkey
                keyboard.add_hotkey(
                    normalized_hotkey,
                    self._on_hotkey_triggered,
                    suppress=False
                )
            
            logger.info(f"Hotkey registriert: {hotkey}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Registrieren des Hotkeys: {e}")
            self.current_hotkey = None
            return False
    
    def _normalize_hotkey(self, hotkey: str) -> str:
        """Normalisiert Hotkey-String für keyboard-Bibliothek"""
        # Ersetze ctrl durch control
        hotkey = hotkey.lower().replace('ctrl', 'ctrl')
        return hotkey
    
    def _check_modifiers(self, hotkey: str) -> bool:
        """Prüft ob alle Modifier-Keys gedrückt sind"""
        parts = hotkey.lower().split('+')
        modifiers = [p.strip() for p in parts[:-1]]
        
        if 'alt' in modifiers and not keyboard.is_pressed('alt'):
            return False
        if 'shift' in modifiers and not keyboard.is_pressed('shift'):
            return False
        if 'ctrl' in modifiers and not keyboard.is_pressed('ctrl'):
            return False
        
        return True
    
    def _on_hotkey_triggered(self):
        """Wird aufgerufen wenn Hotkey getriggert wird (Toggle-Modus)"""
        if self.hotkey_callback:
            self.hotkey_callback()
    
    def _parse_hotkey(self, hotkey: str) -> str:
        """
        Parst Hotkey-String und gibt Haupttaste zurück
        
        Args:
            hotkey: Tastenkombination (z.B. "alt+enter")
            
        Returns:
            Haupttaste (z.B. "enter")
        """
        parts = hotkey.lower().split('+')
        # Letzter Teil ist die Haupttaste
        return parts[-1].strip()
    
    def remove_hotkey(self):
        """Entfernt aktuelles Hotkey"""
        if self.current_hotkey:
            try:
                # Entferne alle Hotkeys (keyboard-Bibliothek hat keine bessere Methode)
                keyboard.unhook_all()
                logger.info(f"Hotkey entfernt: {self.current_hotkey}")
            except Exception as e:
                logger.error(f"Fehler beim Entfernen des Hotkeys: {e}")
            
            self.current_hotkey = None
            self.hotkey_callback = None
            self.hotkey_press_callback = None
            self.hotkey_release_callback = None
    
    def cleanup(self):
        """Bereinigt Ressourcen"""
        self.remove_hotkey()

