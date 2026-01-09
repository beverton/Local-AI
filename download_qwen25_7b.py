"""Download Qwen2.5-7B-Instruct von HuggingFace"""
import os
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-7B-Instruct"
local_dir = r"G:\KI Modelle\coding\qwen-2.5-7b-instruct"

print("=" * 60)
print("Qwen2.5-7B-Instruct Download")
print("=" * 60)
print(f"\nModell: {model_id}")
print(f"Zielordner: {local_dir}")
print("\nDownload startet...")
print("(Dies kann 10-30 Minuten dauern, je nach Internetverbindung)")
print("=" * 60)

try:
    # Erstelle Verzeichnis falls nicht vorhanden
    os.makedirs(local_dir, exist_ok=True)
    
    # Download mit Fortschrittsanzeige
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.md", "*.txt"]  # Nur wichtige Dateien
    )
    
    print("\n" + "=" * 60)
    print("✓ Download erfolgreich abgeschlossen!")
    print("=" * 60)
    print(f"\nModell gespeichert in: {local_dir}")
    print("\nDas Modell kann jetzt geladen werden:")
    print("  - Über Model Manager UI")
    print("  - Oder direkt beim Start (ist jetzt default_model)")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("✗ Fehler beim Download:")
    print("=" * 60)
    print(str(e))
    print("\nMögliche Lösungen:")
    print("  1. Prüfen Sie Ihre Internetverbindung")
    print("  2. Stellen Sie sicher, dass Sie genug Speicherplatz haben (~15 GB)")
    print("  3. Falls HuggingFace Login nötig: huggingface-cli login")




