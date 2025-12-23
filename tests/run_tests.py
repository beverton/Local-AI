"""
Haupt-Skript zum Ausführen aller Tests
"""
import sys
from pathlib import Path

# Füge tests zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from test_runner import TestRunner


def main():
    """Führt alle Tests aus"""
    print("=" * 80)
    print("Local AI - Test Suite")
    print("=" * 80)
    print("\nStelle sicher, dass der Server läuft auf http://127.0.0.1:8000")
    print("Starte Tests in 2 Sekunden...")
    import time
    time.sleep(2)
    
    runner = TestRunner(api_base_url="http://127.0.0.1:8000")
    
    # Führe alle Tests aus
    print("\nFühre Tests aus...")
    summary = runner.run_all_tests()
    
    # Zeige Zusammenfassung
    print("\n" + "=" * 80)
    print("Test-Zusammenfassung")
    print("=" * 80)
    print(f"Gesamt: {summary['total_tests']}")
    print(f"Erfolgreich: {summary['passed']}")
    print(f"Fehlgeschlagen: {summary['failed']}")
    print(f"Fehler: {summary['errors']}")
    print(f"Erfolgsrate: {summary['success_rate']:.1f}%")
    
    if summary['failed'] > 0 or summary['errors'] > 0:
        print("\n⚠️  Einige Tests sind fehlgeschlagen!")
        print("Prüfe die Debug-Reports in .cursor/debug_reports/")
        print("Führe 'python tests/analyze_logs.py --reports' aus für Details")
        return 1
    else:
        print("\n✅ Alle Tests erfolgreich!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

