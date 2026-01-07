"""
Log-Manager für strukturierte Test-Logs
Ermöglicht automatisches Auslesen und Analysieren von Test-Logs
"""
import json
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path


class LogManager:
    """Verwaltet strukturierte Test-Logs für automatische Analyse"""
    
    def __init__(self, log_dir: str = None):
        """
        Initialisiert Log-Manager
        
        Args:
            log_dir: Verzeichnis für Logs (Standard: .cursor/)
        """
        if log_dir is None:
            # Verwende .cursor Verzeichnis im Workspace-Root
            workspace_root = Path(__file__).parent.parent
            log_dir = workspace_root / ".cursor"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_logs_file = self.log_dir / "test_logs.jsonl"
        self.debug_reports_dir = self.log_dir / "debug_reports"
        self.debug_reports_dir.mkdir(parents=True, exist_ok=True)
    
    def write_test_log(self, test_name: str, step: str, status: str, data: Dict[str, Any]):
        """
        Schreibt strukturierten Test-Log
        
        Args:
            test_name: Name des Tests
            step: Schritt im Test (z.B. "start", "load_model", "state_check")
            status: Status ("start", "success", "error", "info")
            data: Zusätzliche Daten
        """
        log_entry = {
            "test_name": test_name,
            "step": step,
            "status": status,
            "data": data,
            "timestamp": time.time()
        }
        
        try:
            with open(self.test_logs_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Fehler beim Schreiben des Test-Logs: {e}")
    
    def read_test_logs(self, test_name: Optional[str] = None, status: Optional[str] = None, 
                      step: Optional[str] = None, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Liest Test-Logs mit Filtern
        
        Args:
            test_name: Filter nach Test-Name
            status: Filter nach Status
            step: Filter nach Schritt
            since_timestamp: Nur Logs nach diesem Timestamp
            
        Returns:
            Liste von Log-Einträgen
        """
        logs = []
        
        if not self.test_logs_file.exists():
            return logs
        
        try:
            with open(self.test_logs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        log_entry = json.loads(line)
                        
                        # Filter anwenden
                        if test_name and log_entry.get("test_name") != test_name:
                            continue
                        if status and log_entry.get("status") != status:
                            continue
                        if step and log_entry.get("step") != step:
                            continue
                        if since_timestamp and log_entry.get("timestamp", 0) < since_timestamp:
                            continue
                        
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Fehler beim Lesen der Test-Logs: {e}")
        
        return logs
    
    def analyze_logs(self, test_name: str) -> List[Dict[str, Any]]:
        """
        Analysiert Logs auf Probleme
        
        Args:
            test_name: Name des Tests
            
        Returns:
            Liste von identifizierten Problemen
        """
        logs = self.read_test_logs(test_name=test_name)
        problems = []
        
        # Prüfe auf Fehler
        error_logs = [log for log in logs if log.get("status") == "error"]
        if error_logs:
            for error_log in error_logs:
                problems.append({
                    "type": "error",
                    "step": error_log.get("step"),
                    "description": f"Fehler in Schritt '{error_log.get('step')}'",
                    "data": error_log.get("data", {}),
                    "timestamp": error_log.get("timestamp")
                })
        
        # Prüfe auf fehlgeschlagene Tests
        success_logs = [log for log in logs if log.get("status") == "success" and log.get("step") == "end"]
        if not success_logs:
            # Test wurde nicht erfolgreich abgeschlossen
            start_logs = [log for log in logs if log.get("step") == "start"]
            if start_logs:
                problems.append({
                    "type": "test_not_completed",
                    "description": "Test wurde nicht erfolgreich abgeschlossen",
                    "start_time": start_logs[0].get("timestamp"),
                    "last_step": logs[-1].get("step") if logs else None
                })
        
        # Prüfe auf Warnungen in State-Checks
        state_check_logs = [log for log in logs if log.get("step") == "state_check"]
        for state_log in state_check_logs:
            state_data = state_log.get("data", {})
            if not state_data.get("is_loaded", False) and state_data.get("expected_loaded", True):
                problems.append({
                    "type": "model_not_loaded",
                    "step": "state_check",
                    "description": "Modell sollte geladen sein, ist es aber nicht",
                    "state": state_data
                })
        
        return problems
    
    def check_expected_vs_actual(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Vergleicht erwartete vs. tatsächliche Werte
        
        Args:
            expected: Erwartete Werte
            actual: Tatsächliche Werte
            
        Returns:
            Liste von Unterschieden
        """
        differences = []
        
        for key, expected_value in expected.items():
            if key not in actual:
                differences.append({
                    "key": key,
                    "type": "missing",
                    "expected": expected_value,
                    "actual": None
                })
            elif actual[key] != expected_value:
                differences.append({
                    "key": key,
                    "type": "mismatch",
                    "expected": expected_value,
                    "actual": actual[key]
                })
        
        # Prüfe auf zusätzliche Keys in actual
        for key in actual:
            if key not in expected:
                differences.append({
                    "key": key,
                    "type": "unexpected",
                    "expected": None,
                    "actual": actual[key]
                })
        
        return differences
    
    def write_debug_report(self, test_name: str, problems: List[Dict[str, Any]], 
                          system_state: Optional[Dict[str, Any]] = None,
                          relevant_logs: Optional[List[Dict[str, Any]]] = None):
        """
        Schreibt Debug-Report
        
        Args:
            test_name: Name des Tests
            problems: Liste von Problemen
            system_state: Aktueller System-Status
            relevant_logs: Relevante Log-Einträge
        """
        report = {
            "test_name": test_name,
            "timestamp": time.time(),
            "status": "failed" if problems else "success",
            "problems": problems,
            "system_state": system_state or {},
            "relevant_logs": relevant_logs or []
        }
        
        # Generiere Lösungsvorschläge basierend auf Problem-Typen
        suggested_fixes = []
        for problem in problems:
            problem_type = problem.get("type")
            if problem_type == "model_not_loaded":
                suggested_fixes.append({
                    "problem": "model_not_loaded",
                    "fix": "Prüfe ob load_model() korrekt aufgerufen wurde und current_model_id gesetzt ist",
                    "location": problem.get("step", "unknown")
                })
            elif problem_type == "error":
                suggested_fixes.append({
                    "problem": "error",
                    "fix": f"Prüfe Fehler-Details in Schritt '{problem.get('step')}'",
                    "location": problem.get("step", "unknown")
                })
            elif problem_type == "test_not_completed":
                suggested_fixes.append({
                    "problem": "test_not_completed",
                    "fix": "Test wurde nicht abgeschlossen - prüfe ob Exception aufgetreten ist",
                    "location": "test_execution"
                })
        
        report["suggested_fixes"] = suggested_fixes
        
        # Speichere Report
        report_file = self.debug_reports_dir / f"{test_name}_{int(time.time())}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Schreiben des Debug-Reports: {e}")
        
        return report
    
    def read_debug_reports(self, test_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Liest Debug-Reports
        
        Args:
            test_name: Filter nach Test-Name
            
        Returns:
            Liste von Debug-Reports
        """
        reports = []
        
        if not self.debug_reports_dir.exists():
            return reports
        
        for report_file in self.debug_reports_dir.glob("*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    if test_name is None or report.get("test_name") == test_name:
                        reports.append(report)
            except Exception as e:
                print(f"Fehler beim Lesen des Debug-Reports {report_file}: {e}")
        
        # Sortiere nach Timestamp (neueste zuerst)
        reports.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return reports









