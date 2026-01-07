"""
Standalone Log-Analyse-Tool
Kann unabhängig von Tests verwendet werden um Logs zu analysieren
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Füge tests zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from log_manager import LogManager
from system_checker import SystemChecker


def analyze_test_logs(test_name: str = None, show_details: bool = False):
    """
    Analysiert Test-Logs
    
    Args:
        test_name: Optional - spezifischer Test-Name
        show_details: Zeige detaillierte Informationen
    """
    log_manager = LogManager()
    
    print("=" * 80)
    print("Test-Log Analyse")
    print("=" * 80)
    
    if test_name:
        logs = log_manager.read_test_logs(test_name=test_name)
        print(f"\nAnalysiere Test: {test_name}")
        print(f"Gefundene Log-Einträge: {len(logs)}")
    else:
        logs = log_manager.read_test_logs()
        print(f"\nAnalysiere alle Tests")
        print(f"Gefundene Log-Einträge: {len(logs)}")
    
    if not logs:
        print("\nKeine Logs gefunden.")
        return
    
    # Gruppiere nach Test-Name
    tests = {}
    for log in logs:
        test = log.get("test_name", "unknown")
        if test not in tests:
            tests[test] = []
        tests[test].append(log)
    
    print(f"\nGefundene Tests: {len(tests)}")
    print("-" * 80)
    
    for test_name, test_logs in tests.items():
        print(f"\nTest: {test_name}")
        print(f"  Log-Einträge: {len(test_logs)}")
        
        # Analysiere Test
        problems = log_manager.analyze_logs(test_name)
        
        if problems:
            print(f"  Probleme gefunden: {len(problems)}")
            for problem in problems:
                print(f"    - {problem.get('type')}: {problem.get('description')}")
        else:
            print("  Keine Probleme gefunden")
        
        # Zeige Status
        status_logs = [log for log in test_logs if log.get("step") == "end"]
        if status_logs:
            last_status = status_logs[-1].get("status")
            print(f"  Status: {last_status}")
        
        if show_details:
            print("\n  Detaillierte Logs:")
            for log in test_logs[-5:]:  # Letzte 5 Einträge
                step = log.get("step", "unknown")
                status = log.get("status", "unknown")
                timestamp = log.get("timestamp", 0)
                print(f"    [{timestamp}] {step}: {status}")


def analyze_debug_logs(since_timestamp: float = None, show_warnings: bool = False):
    """
    Analysiert Debug-Logs (.cursor/debug.log)
    
    Args:
        since_timestamp: Optional - nur Logs nach diesem Timestamp
        show_warnings: Zeige auch Warnungen
    """
    system_checker = SystemChecker()
    
    print("=" * 80)
    print("Debug-Log Analyse")
    print("=" * 80)
    
    log_analysis = system_checker.check_logs_for_errors(since_timestamp=since_timestamp)
    
    print(f"\nFehler gefunden: {len(log_analysis['errors'])}")
    if log_analysis["errors"]:
        print("\nFehler:")
        for error in log_analysis["errors"][:10]:  # Erste 10 Fehler
            location = error.get("location", "unknown")
            message = error.get("message", "no message")
            timestamp = error.get("timestamp")
            print(f"  [{timestamp}] {location}: {message}")
    
    if show_warnings:
        print(f"\nWarnungen gefunden: {len(log_analysis['warnings'])}")
        if log_analysis["warnings"]:
            print("\nWarnungen:")
            for warning in log_analysis["warnings"][:10]:
                location = warning.get("location", "unknown")
                message = warning.get("message", "no message")
                timestamp = warning.get("timestamp")
                print(f"  [{timestamp}] {location}: {message}")


def show_debug_reports(test_name: str = None):
    """
    Zeigt Debug-Reports
    
    Args:
        test_name: Optional - spezifischer Test-Name
    """
    log_manager = LogManager()
    
    print("=" * 80)
    print("Debug-Reports")
    print("=" * 80)
    
    reports = log_manager.read_debug_reports(test_name=test_name)
    
    if not reports:
        print("\nKeine Debug-Reports gefunden.")
        return
    
    print(f"\nGefundene Reports: {len(reports)}")
    print("-" * 80)
    
    for report in reports[:10]:  # Erste 10 Reports
        test_name = report.get("test_name", "unknown")
        status = report.get("status", "unknown")
        timestamp = report.get("timestamp", 0)
        problems = report.get("problems", [])
        suggested_fixes = report.get("suggested_fixes", [])
        
        print(f"\nTest: {test_name}")
        print(f"  Status: {status}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Probleme: {len(problems)}")
        
        if problems:
            print("\n  Probleme:")
            for problem in problems[:5]:  # Erste 5 Probleme
                problem_type = problem.get("type", "unknown")
                description = problem.get("description", "no description")
                location = problem.get("location", "unknown")
                print(f"    - [{problem_type}] {description}")
                print(f"      Location: {location}")
        
        if suggested_fixes:
            print("\n  Lösungsvorschläge:")
            for fix in suggested_fixes[:3]:  # Erste 3 Vorschläge
                problem = fix.get("problem", "unknown")
                fix_text = fix.get("fix", "no fix")
                print(f"    - {problem}: {fix_text}")


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Analysiere Test-Logs und Debug-Reports")
    parser.add_argument("--test", help="Spezifischer Test-Name")
    parser.add_argument("--debug-logs", action="store_true", help="Analysiere Debug-Logs")
    parser.add_argument("--reports", action="store_true", help="Zeige Debug-Reports")
    parser.add_argument("--details", action="store_true", help="Zeige detaillierte Informationen")
    parser.add_argument("--warnings", action="store_true", help="Zeige auch Warnungen")
    parser.add_argument("--since", type=float, help="Nur Logs nach diesem Timestamp")
    
    args = parser.parse_args()
    
    if args.debug_logs:
        analyze_debug_logs(since_timestamp=args.since, show_warnings=args.warnings)
    elif args.reports:
        show_debug_reports(test_name=args.test)
    else:
        analyze_test_logs(test_name=args.test, show_details=args.details)


if __name__ == "__main__":
    main()









