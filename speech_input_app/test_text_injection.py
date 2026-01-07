"""
Test-Script für Text-Injection
Testet verschiedene Methoden und gibt Feedback
"""
import sys
import time
from pathlib import Path

# Füge Parent-Verzeichnis zum Python-Pfad hinzu
if __name__ == "__main__":
    app_dir = Path(__file__).parent
    parent_dir = app_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from speech_input_app.text_injector import TextInjector
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_text_injection():
    """Testet Text-Injection mit verschiedenen Methoden"""
    print("=" * 60)
    print("Text-Injection Test")
    print("=" * 60)
    print()
    print("WICHTIG: Öffnen Sie ein Textfeld (z.B. Notepad, OpenOffice)")
    print("und setzen Sie den Cursor hinein!")
    print()
    input("Drücken Sie Enter, wenn Sie bereit sind...")
    print()
    
    injector = TextInjector()
    test_text = "TEST TEXT EINFUEGEN"
    
    print(f"Teste Text-Injection mit: '{test_text}'")
    print()
    
    # Test 1: Normale Injection
    print("[Test 1] Normale Text-Injection...")
    result = injector.inject_text(test_text)
    print(f"Ergebnis: {'ERFOLG' if result else 'FEHLGESCHLAGEN'}")
    print()
    
    input("Prüfen Sie, ob der Text eingefügt wurde. Drücken Sie Enter zum Fortfahren...")
    print()
    
    # Test 2: Mit längerer Wartezeit
    print("[Test 2] Text-Injection mit längerer Wartezeit...")
    test_text2 = "TEST 2 - LANGERE WARTEZEIT"
    result = injector.inject_text(test_text2)
    print(f"Ergebnis: {'ERFOLG' if result else 'FEHLGESCHLAGEN'}")
    print()
    
    input("Prüfen Sie erneut. Drücken Sie Enter zum Fortfahren...")
    print()
    
    # Test 3: Direkter Clipboard-Test
    print("[Test 3] Direkter Clipboard-Test...")
    try:
        import win32clipboard
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText("TEST 3 - DIREKTER CLIPBOARD", win32clipboard.CF_UNICODETEXT)
        win32clipboard.CloseClipboard()
        print("Text in Clipboard gesetzt!")
        print("Bitte drücken Sie manuell STRG+V im Textfeld")
        input("Drücken Sie Enter, wenn Sie STRG+V gedrückt haben...")
    except Exception as e:
        print(f"Fehler: {e}")
    
    print()
    print("=" * 60)
    print("Test abgeschlossen!")
    print("=" * 60)


if __name__ == "__main__":
    test_text_injection()


