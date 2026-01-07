"""
Installer/Deinstaller für Speech Input App
"""
import sys
import os
import shutil
import subprocess
import json
from pathlib import Path
import winreg
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SpeechInputInstaller:
    """Installer für Speech Input App"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.app_path = self.base_path / "speech_input_app"
        self.start_script = self.base_path / "start_speech_input.bat"
        self.requirements = self.base_path / "requirements-speech-input.txt"
        
        # Registry-Pfad für Auto-Start
        self.registry_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        self.registry_value = "LocalAISpeechInput"
    
    def install(self):
        """Installiert die App"""
        print("=" * 50)
        print("Speech Input App - Installation")
        print("=" * 50)
        print()
        
        # Prüfe ob App bereits läuft
        if self._is_app_running():
            print("WARNUNG: Speech Input App läuft bereits!")
            print("Bitte beenden Sie die App vor der Installation.")
            input("Drücken Sie Enter zum Fortfahren...")
            print()
        
        # Prüfe Python
        if not self._check_python():
            print("FEHLER: Python ist nicht installiert oder nicht im PATH!")
            return False
        
        # Installiere Dependencies
        print("[1/4] Installiere Dependencies...")
        if not self._install_dependencies():
            print("FEHLER: Dependencies konnten nicht installiert werden!")
            return False
        print("✓ Dependencies installiert")
        print()
        
        # Prüfe App-Dateien
        print("[2/4] Prüfe App-Dateien...")
        if not self._check_app_files():
            print("FEHLER: App-Dateien nicht gefunden!")
            return False
        print("✓ App-Dateien gefunden")
        print()
        
        # Erstelle Desktop-Verknüpfung
        print("[3/4] Erstelle Desktop-Verknüpfung...")
        if self._create_desktop_shortcut():
            print("✓ Desktop-Verknüpfung erstellt")
        else:
            print("⚠ Desktop-Verknüpfung konnte nicht erstellt werden")
        print()
        
        # Frage nach Auto-Start
        print("[4/4] Auto-Start einrichten...")
        auto_start = input("Soll die App beim Windows-Start automatisch gestartet werden? (j/n): ").lower().strip()
        if auto_start in ['j', 'y', 'ja', 'yes']:
            if self._setup_autostart():
                print("✓ Auto-Start eingerichtet")
            else:
                print("⚠ Auto-Start konnte nicht eingerichtet werden")
        else:
            print("○ Auto-Start übersprungen")
        print()
        
        print("=" * 50)
        print("Installation abgeschlossen!")
        print("=" * 50)
        print()
        print("Die App kann jetzt über 'start_speech_input.bat' gestartet werden.")
        print("Oder nutzen Sie die Desktop-Verknüpfung.")
        print()
        
        return True
    
    def uninstall(self):
        """Deinstalliert die App"""
        print("=" * 50)
        print("Speech Input App - Deinstallation")
        print("=" * 50)
        print()
        
        # Prüfe ob App läuft
        if self._is_app_running():
            print("WARNUNG: Speech Input App läuft noch!")
            print("Bitte beenden Sie die App vor der Deinstallation.")
            stop = input("Soll die App jetzt beendet werden? (j/n): ").lower().strip()
            if stop in ['j', 'y', 'ja', 'yes']:
                self._stop_app()
            else:
                print("Deinstallation abgebrochen.")
                return False
            print()
        
        # Bestätigung
        confirm = input("Möchten Sie die Speech Input App wirklich deinstallieren? (j/n): ").lower().strip()
        if confirm not in ['j', 'y', 'ja', 'yes']:
            print("Deinstallation abgebrochen.")
            return False
        
        print()
        
        # Entferne Auto-Start
        print("[1/4] Entferne Auto-Start...")
        if self._remove_autostart():
            print("✓ Auto-Start entfernt")
        else:
            print("⚠ Auto-Start konnte nicht entfernt werden (möglicherweise nicht eingerichtet)")
        print()
        
        # Entferne Desktop-Verknüpfung
        print("[2/4] Entferne Desktop-Verknüpfung...")
        if self._remove_desktop_shortcut():
            print("✓ Desktop-Verknüpfung entfernt")
        else:
            print("⚠ Desktop-Verknüpfung konnte nicht entfernt werden")
        print()
        
        # Frage nach App-Dateien
        print("[3/4] App-Dateien...")
        remove_files = input("Sollen die App-Dateien gelöscht werden? (j/n): ").lower().strip()
        if remove_files in ['j', 'y', 'ja', 'yes']:
            if self._remove_app_files():
                print("✓ App-Dateien entfernt")
            else:
                print("⚠ App-Dateien konnten nicht vollständig entfernt werden")
        else:
            print("○ App-Dateien bleiben erhalten")
        print()
        
        # Frage nach Dependencies
        print("[4/4] Dependencies...")
        remove_deps = input("Sollen die Python-Dependencies deinstalliert werden? (j/n): ").lower().strip()
        if remove_deps in ['j', 'y', 'ja', 'yes']:
            print("⚠ Hinweis: Dependencies werden manuell deinstalliert.")
            print("   Führen Sie aus: pip uninstall PyQt6 sounddevice keyboard pywin32")
        else:
            print("○ Dependencies bleiben installiert")
        print()
        
        print("=" * 50)
        print("Deinstallation abgeschlossen!")
        print("=" * 50)
        print()
        
        return True
    
    def _check_python(self) -> bool:
        """Prüft ob Python verfügbar ist"""
        try:
            result = subprocess.run(['python', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _install_dependencies(self) -> bool:
        """Installiert Dependencies"""
        if not self.requirements.exists():
            logger.error(f"Requirements-Datei nicht gefunden: {self.requirements}")
            return False
        
        try:
            result = subprocess.run(
                ['pip', 'install', '-r', str(self.requirements)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Fehler beim Installieren der Dependencies: {e}")
            return False
    
    def _check_app_files(self) -> bool:
        """Prüft ob App-Dateien vorhanden sind"""
        required_files = [
            self.app_path / "speech_input.py",
            self.app_path / "audio_recorder.py",
            self.app_path / "hotkey_handler.py",
            self.app_path / "api_client.py",
            self.app_path / "text_injector.py",
            self.start_script
        ]
        
        for file in required_files:
            if not file.exists():
                logger.error(f"Datei nicht gefunden: {file}")
                return False
        
        return True
    
    def _create_desktop_shortcut(self) -> bool:
        """Erstellt Desktop-Verknüpfung"""
        try:
            desktop = Path.home() / "Desktop"
            if not desktop.exists():
                # Versuche öffentlichen Desktop
                desktop = Path(os.environ.get('PUBLIC', '')) / "Desktop"
                if not desktop.exists():
                    return False
            
            shortcut_path = desktop / "Speech Input.lnk"
            
            # Erstelle VBScript für Verknüpfung
            vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{self.start_script}"
oLink.WorkingDirectory = "{self.base_path}"
oLink.Description = "Local AI Speech Input"
oLink.Save
"""
            
            vbs_file = self.base_path / "create_shortcut.vbs"
            with open(vbs_file, 'w', encoding='utf-8') as f:
                f.write(vbs_script)
            
            # Führe VBScript aus
            result = subprocess.run(
                ['cscript', '//nologo', str(vbs_file)],
                capture_output=True,
                text=True
            )
            
            # Lösche temporäres VBScript
            vbs_file.unlink(missing_ok=True)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Desktop-Verknüpfung: {e}")
            return False
    
    def _remove_desktop_shortcut(self) -> bool:
        """Entfernt Desktop-Verknüpfung"""
        try:
            desktop = Path.home() / "Desktop"
            if not desktop.exists():
                desktop = Path(os.environ.get('PUBLIC', '')) / "Desktop"
                if not desktop.exists():
                    return False
            
            shortcut_path = desktop / "Speech Input.lnk"
            if shortcut_path.exists():
                shortcut_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Fehler beim Entfernen der Desktop-Verknüpfung: {e}")
            return False
    
    def _setup_autostart(self) -> bool:
        """Richtet Auto-Start ein"""
        try:
            # Erstelle vollständigen Pfad zum Start-Script
            script_path = str(self.start_script.absolute())
            
            # Öffne Registry-Key
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.registry_key,
                0,
                winreg.KEY_SET_VALUE
            )
            
            # Setze Wert
            winreg.SetValueEx(
                key,
                self.registry_value,
                0,
                winreg.REG_SZ,
                script_path
            )
            
            winreg.CloseKey(key)
            
            # Aktualisiere auch config.json
            config_path = self.app_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    config['auto_start'] = True
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                except:
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Einrichten des Auto-Starts: {e}")
            return False
    
    def _remove_autostart(self) -> bool:
        """Entfernt Auto-Start"""
        try:
            # Öffne Registry-Key
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                self.registry_key,
                0,
                winreg.KEY_SET_VALUE
            )
            
            # Versuche Wert zu löschen
            try:
                winreg.DeleteValue(key, self.registry_value)
                winreg.CloseKey(key)
                return True
            except FileNotFoundError:
                # Wert existiert nicht
                winreg.CloseKey(key)
                return False
            
        except Exception as e:
            logger.error(f"Fehler beim Entfernen des Auto-Starts: {e}")
            return False
    
    def _remove_app_files(self) -> bool:
        """Entfernt App-Dateien"""
        try:
            # Entferne App-Verzeichnis
            if self.app_path.exists():
                shutil.rmtree(self.app_path)
            
            # Entferne Start-Script
            if self.start_script.exists():
                self.start_script.unlink()
            
            # Entferne Installer/Deinstaller-Scripts (optional)
            installer_script = self.base_path / "install_speech_input.bat"
            deinstaller_script = self.base_path / "uninstall_speech_input.bat"
            
            if installer_script.exists():
                remove_installer = input("Soll auch der Installer entfernt werden? (j/n): ").lower().strip()
                if remove_installer in ['j', 'y', 'ja', 'yes']:
                    installer_script.unlink()
            
            if deinstaller_script.exists():
                deinstaller_script.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Entfernen der App-Dateien: {e}")
            return False
    
    def _is_app_running(self) -> bool:
        """Prüft ob die App läuft"""
        try:
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True,
                text=True
            )
            
            if 'speech_input.py' in result.stdout:
                return True
            
            return False
        except:
            return False
    
    def _stop_app(self) -> bool:
        """Beendet die App"""
        try:
            # Finde Prozess
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True,
                text=True
            )
            
            # Versuche Prozess zu beenden (vereinfacht)
            print("Versuche App zu beenden...")
            # In einer echten Implementierung würde man hier den Prozess finden und beenden
            # Für jetzt: Benutzer muss manuell beenden
            print("Bitte beenden Sie die App manuell (Rechtsklick auf Tray-Icon → Beenden)")
            input("Drücken Sie Enter wenn die App beendet wurde...")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Beenden der App: {e}")
            return False


def main():
    """Hauptfunktion"""
    if len(sys.argv) > 1 and sys.argv[1] == 'uninstall':
        installer = SpeechInputInstaller()
        installer.uninstall()
    else:
        installer = SpeechInputInstaller()
        installer.install()


if __name__ == "__main__":
    main()

