"""Test-Skript um den Server zu starten und Fehler zu sehen"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from main import app  # backend ist bereits im Pfad, daher direkt 'main' importieren
    import uvicorn
    print("Starte Server auf http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
except Exception as e:
    print(f"Fehler: {e}")
    import traceback
    traceback.print_exc()





