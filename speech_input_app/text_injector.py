"""
Text Injection - Fügt Text in aktives Fenster ein
Nutzt Clipboard + Strg+V (zuverlässigste Methode)
"""
import logging
import time
from typing import Optional

try:
    import win32clipboard
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("pywin32 nicht verfügbar - Text-Injection wird eingeschränkt funktionieren")

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logging.warning("keyboard-Bibliothek nicht verfügbar - Text-Injection wird eingeschränkt funktionieren")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("pyautogui nicht verfügbar - Alternative Text-Injection-Methode nicht verfügbar")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextInjector:
    """Fügt Text in aktives Fenster ein"""
    
    def __init__(self, use_clipboard_fallback: bool = True):
        self.use_clipboard_fallback = use_clipboard_fallback
        if not WIN32_AVAILABLE:
            logger.warning("pywin32 nicht verfügbar - nur Clipboard-Fallback verfügbar")
    
    def inject_text(self, text: str, window_handle: Optional[int] = None) -> bool:
        """
        Fügt Text in aktives Fenster ein
        
        Args:
            text: Text der eingefügt werden soll
            window_handle: Optional - Handle des Fensters, in das eingefügt werden soll
            
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        if not text:
            logger.warning("Kein Text zum Einfügen")
            return False
        
        logger.info(f"Versuche Text einzufügen: {text[:50]}...")
        
        # WICHTIG: Fokussiere aktives Fenster ZUERST
        # Wenn window_handle gegeben ist, nutze das, sonst hole aktives Fenster
        if window_handle and WIN32_AVAILABLE:
            logger.info(f"Nutze gespeichertes Fenster-Handle: {window_handle}")
            self._focus_active_window(window_handle)
        else:
            logger.info("Nutze aktuelles aktives Fenster")
            self._focus_active_window()
        
        # Nutze Clipboard + Strg+V (zuverlässigste Methode)
        # Ein Versuch - wenn es funktioniert, funktioniert es
        return self._inject_with_clipboard(text, window_handle)
    
    def _focus_active_window(self, hwnd=None):
        """Fokussiert das aktive Fenster"""
        try:
            if WIN32_AVAILABLE:
                # Nutze gegebenes Handle oder hole aktives Fenster
                if hwnd is None:
                    hwnd = win32gui.GetForegroundWindow()
                
                if hwnd:
                    logger.info(f"Fokussiere Fenster: {hwnd}")
                    
                    # Versuche verschiedene Methoden, um Fenster zu aktivieren
                    try:
                        # Methode 1: SetForegroundWindow (mehrfach für Zuverlässigkeit)
                        for _ in range(3):
                            win32gui.SetForegroundWindow(hwnd)
                            win32gui.BringWindowToTop(hwnd)
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32gui.SetActiveWindow(hwnd)
                            time.sleep(0.1)
                        
                        # Warte länger, damit Fenster wirklich aktiv ist
                        time.sleep(0.4)
                        
                        # Prüfe ob Fenster wirklich aktiv ist
                        current_hwnd = win32gui.GetForegroundWindow()
                        if current_hwnd == hwnd:
                            logger.info("✓ Fenster erfolgreich fokussiert")
                        else:
                            logger.warning(f"⚠ Fenster-Fokus könnte fehlgeschlagen sein (erwartet: {hwnd}, aktuell: {current_hwnd})")
                            # Versuche nochmal
                            win32gui.SetForegroundWindow(hwnd)
                            time.sleep(0.2)
                    except Exception as e:
                        logger.error(f"Fehler beim Fokussieren: {e}")
                else:
                    logger.warning("Kein Fenster-Handle verfügbar")
        except Exception as e:
            logger.error(f"Fehler beim Fokussieren des Fensters: {e}")
    
    def _inject_with_clipboard(self, text: str, window_handle: Optional[int] = None) -> bool:
        """
        Fügt Text mit Clipboard ein (Fallback)
        """
        try:
            import win32clipboard
            
            # Speichere aktuellen Clipboard-Inhalt
            try:
                win32clipboard.OpenClipboard()
                old_clipboard = win32clipboard.GetClipboardData()
                win32clipboard.CloseClipboard()
            except:
                old_clipboard = None
            
            # Setze Text in Clipboard
            # WICHTIG: Warte bis Clipboard frei ist (kann blockiert sein)
            clipboard_ready = False
            for attempt in range(10):  # Max 1 Sekunde warten
                try:
                    win32clipboard.OpenClipboard()
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
                    win32clipboard.CloseClipboard()
                    clipboard_ready = True
                    logger.info(f"✓ Text in Clipboard gesetzt: '{text[:50]}...'")
                    
                    # VERIFIZIERE: Prüfe ob Text wirklich im Clipboard ist
                    try:
                        win32clipboard.OpenClipboard()
                        clipboard_text = win32clipboard.GetClipboardData()
                        win32clipboard.CloseClipboard()
                        if clipboard_text == text:
                            logger.info("✓ Clipboard-Verifizierung erfolgreich")
                        else:
                            logger.warning(f"⚠ Clipboard-Inhalt stimmt nicht überein! Erwartet: '{text[:30]}...', Gefunden: '{clipboard_text[:30] if clipboard_text else 'None'}...'")
                    except Exception as e:
                        logger.warning(f"Konnte Clipboard nicht verifizieren: {e}")
                    
                    break
                except Exception as e:
                    logger.debug(f"Clipboard blockiert (Versuch {attempt + 1}/10): {e}")
                    time.sleep(0.1)
            
            if not clipboard_ready:
                logger.error("❌ Clipboard konnte nicht gesetzt werden - möglicherweise blockiert")
                return False
            
            # Warte kurz, damit Clipboard bereit ist
            time.sleep(0.15)
            
            # WICHTIG: Fokussiere aktives Fenster VOR Strg+V
            if WIN32_AVAILABLE:
                try:
                    hwnd = window_handle if window_handle else win32gui.GetForegroundWindow()
                    if hwnd:
                        win32gui.SetForegroundWindow(hwnd)
                        win32gui.BringWindowToTop(hwnd)
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        time.sleep(0.15)  # Kurze Wartezeit
                        logger.info(f"Fenster fokussiert: {hwnd}")
                except Exception as e:
                    logger.warning(f"Fehler beim Fokussieren: {e}")
            
            # Simuliere Strg+V - EINFACH und DIREKT
            logger.info("Sende Strg+V...")
            
            # Methode 1: pyautogui (zuverlässigste)
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.hotkey('ctrl', 'v')
                    logger.info("✓ Strg+V gesendet (pyautogui)")
                    return True
                except Exception as e:
                    logger.warning(f"Fehler mit pyautogui: {e}")
            
            # Methode 2: keyboard-Bibliothek (Fallback)
            if KEYBOARD_AVAILABLE:
                try:
                    keyboard.press_and_release('ctrl+v')
                    logger.info("✓ Strg+V gesendet (keyboard)")
                    return True
                except Exception as e:
                    logger.warning(f"Fehler mit keyboard: {e}")
            
            # Methode 3: SendInput (letzter Fallback)
            try:
                self._send_key_combination(['V'], ctrl=True)
                logger.info("✓ Strg+V gesendet (SendInput)")
                return True
            except Exception as e:
                logger.error(f"Fehler mit SendInput: {e}")
            
            logger.error("❌ Alle Methoden fehlgeschlagen - Text ist im Clipboard")
            return False
            
        except Exception as e:
            logger.error(f"Fehler bei Clipboard-Methode: {e}")
            return False
    
    def _send_key_combination(self, keys: list, ctrl: bool = False, alt: bool = False, shift: bool = False):
        """
        Sendet Tastenkombination
        
        Args:
            keys: Liste von Tasten (z.B. ['V'])
            ctrl: Strg-Taste gedrückt halten
            alt: Alt-Taste gedrückt halten
            shift: Shift-Taste gedrückt halten
        """
        if not WIN32_AVAILABLE:
            return
        
        try:
            import ctypes
            from ctypes import wintypes
            
            # Virtual Key Codes
            VK_CONTROL = 0x11
            VK_ALT = 0x12
            VK_SHIFT = 0x10
            
            # Key Codes
            key_map = {
                'V': 0x56,
                'ENTER': 0x0D,
                'SPACE': 0x20,
            }
            
            # Erstelle INPUT-Struktur
            INPUT_KEYBOARD = 1
            KEYEVENTF_KEYUP = 0x0002
            
            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", wintypes.WORD),
                    ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
                ]
            
            class INPUT(ctypes.Structure):
                class _INPUT(ctypes.Union):
                    _fields_ = [("ki", KEYBDINPUT)]
                _anonymous_ = ("_input",)
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("_input", _INPUT)
                ]
            
            inputs = []
            
            # Drücke Modifier-Keys
            if ctrl:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_CONTROL, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
            
            if alt:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_ALT, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
            
            if shift:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_SHIFT, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
            
            # Drücke Haupttasten
            for key in keys:
                vk = key_map.get(key.upper(), ord(key.upper()))
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)))
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)))
            
            # Lasse Modifier-Keys los
            if shift:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_SHIFT, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)))
            
            if alt:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_ALT, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)))
            
            if ctrl:
                inputs.append(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=VK_CONTROL, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)))
            
            # Sende Inputs
            ctypes.windll.user32.SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))
            
        except Exception as e:
            logger.error(f"Fehler beim Senden von Tastenkombination: {e}")

