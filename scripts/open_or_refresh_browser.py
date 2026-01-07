#!/usr/bin/env python3
"""
Smart Browser Tab Manager
Öffnet URLs nur in neuen Tabs, wenn sie nicht bereits geöffnet sind.
Bei bereits geöffneten Tabs wird versucht, diese zu refreshen.
"""

import sys
import webbrowser
import time
import os
from pathlib import Path

def get_tab_state_file():
    """Gibt den Pfad zur Tab-Status-Datei zurück"""
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data"
    # Stelle sicher, dass das data-Verzeichnis existiert
    data_dir.mkdir(exist_ok=True)
    return data_dir / "browser_tabs.state"

def read_open_tabs():
    """Liest die Liste der bereits geöffneten Tabs"""
    state_file = get_tab_state_file()
    if not state_file.exists():
        return set()
    
    try:
        with open(state_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"Fehler beim Lesen der Tab-Status-Datei: {e}")
        return set()

def save_open_tabs(tabs):
    """Speichert die Liste der geöffneten Tabs"""
    state_file = get_tab_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(state_file, 'w') as f:
            for tab in tabs:
                f.write(f"{tab}\n")
    except Exception as e:
        print(f"Fehler beim Speichern der Tab-Status-Datei: {e}")

def open_or_refresh_url(url):
    """
    Öffnet eine URL oder refresht einen existierenden Tab
    
    Args:
        url: Die zu öffnende URL
    
    Returns:
        bool: True wenn ein neuer Tab geöffnet wurde, False wenn refreshed
    """
    open_tabs = read_open_tabs()
    
    if url in open_tabs:
        print(f"Tab bereits geöffnet, versuche Refresh: {url}")
        # Bei bereits geöffnetem Tab: Öffne die URL erneut
        # Die meisten Browser refreshen dann den existierenden Tab
        # oder fokussieren ihn, anstatt einen neuen zu öffnen
        try:
            # new=0 bedeutet: Im gleichen Browser-Fenster öffnen wenn möglich
            # Das führt bei den meisten Browsern zu einem Refresh des existierenden Tabs
            webbrowser.get().open(url, new=0, autoraise=True)
            print(f"✓ Refresh-Signal gesendet für: {url}")
            return False
        except Exception as e:
            print(f"Fehler beim Refresh: {e}")
            # Fallback: Versuche es normal zu öffnen
            webbrowser.open(url, new=2)
            return True
    else:
        print(f"Öffne neuen Tab: {url}")
        try:
            # new=2 bedeutet: In neuem Tab öffnen
            webbrowser.open(url, new=2)
            open_tabs.add(url)
            save_open_tabs(open_tabs)
            print(f"✓ Neuer Tab geöffnet: {url}")
            return True
        except Exception as e:
            print(f"Fehler beim Öffnen: {e}")
            return False

def clear_tab_state():
    """Löscht die Tab-Status-Datei (für Cleanup)"""
    state_file = get_tab_state_file()
    if state_file.exists():
        try:
            state_file.unlink()
            print("Tab-Status-Datei gelöscht")
        except Exception as e:
            print(f"Fehler beim Löschen der Tab-Status-Datei: {e}")

def main():
    """Hauptfunktion"""
    if len(sys.argv) < 2:
        print("Verwendung: open_or_refresh_browser.py <URL> [URL2] [URL3] ...")
        print("          oder: open_or_refresh_browser.py --clear (um Status zu löschen)")
        sys.exit(1)
    
    # Spezialbefehl: Status löschen
    if sys.argv[1] == "--clear":
        clear_tab_state()
        sys.exit(0)
    
    # URLs aus Kommandozeilenargumenten
    urls = sys.argv[1:]
    
    print("========================================")
    print("Smart Browser Tab Manager")
    print("========================================")
    
    # Kleine Verzögerung zwischen den URLs
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(1)  # 1 Sekunde Pause zwischen URLs
        open_or_refresh_url(url)
    
    print("========================================")
    print("Fertig!")
    print("========================================")

if __name__ == "__main__":
    main()

