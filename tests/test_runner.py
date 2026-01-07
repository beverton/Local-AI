"""
Test-Runner mit Auto-Debugging und Report-Generierung
"""
import sys
import os
import traceback
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

# Füge backend zum Python-Pfad hinzu
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "backend"))

from log_manager import LogManager
from system_checker import SystemChecker


class TestRunner:
    """Führt Tests aus mit vollständigem Logging und Auto-Debugging"""
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000"):
        """
        Initialisiert Test-Runner
        
        Args:
            api_base_url: Base URL der API
        """
        self.log_manager = LogManager()
        self.system_checker = SystemChecker(api_base_url=api_base_url)
        self.results = []
    
    def run_all_tests(self, test_modules: List[str] = None) -> Dict[str, Any]:
        """
        Führt alle Tests aus mit Logging
        
        Args:
            test_modules: Optional - Liste von Test-Modul-Namen
            
        Returns:
            Zusammenfassung der Ergebnisse
        """
        if test_modules is None:
            # Finde alle Test-Module
            test_modules = self._discover_test_modules()
        
        results = []
        for test_module in test_modules:
            try:
                result = self.run_test_module(test_module)
                results.append(result)
            except Exception as e:
                results.append({
                    "module": test_module,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
        
        # Generiere Gesamt-Report
        summary = self.generate_summary_report(results)
        return summary
    
    def run_test_module(self, module_name: str) -> Dict[str, Any]:
        """
        Führt alle Tests in einem Modul aus
        
        Args:
            module_name: Name des Test-Moduls (ohne .py)
            
        Returns:
            Ergebnisse des Moduls
        """
        module_path = Path(__file__).parent / f"{module_name}.py"
        if not module_path.exists():
            return {
                "module": module_name,
                "status": "error",
                "error": f"Modul nicht gefunden: {module_path}"
            }
        
        # Importiere Modul
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Finde alle Test-Funktionen
        test_functions = [name for name in dir(module) if name.startswith("test_")]
        
        results = []
        for test_func_name in test_functions:
            test_func = getattr(module, test_func_name)
            if callable(test_func):
                result = self.run_test_with_logging(test_func_name, test_func, module)
                results.append(result)
        
        return {
            "module": module_name,
            "tests": results,
            "total": len(results),
            "passed": sum(1 for r in results if r.get("status") == "passed"),
            "failed": sum(1 for r in results if r.get("status") == "failed")
        }
    
    def run_test_with_logging(self, test_name: str, test_func: Callable, 
                             test_module: Any) -> Dict[str, Any]:
        """
        Führt einzelnen Test aus mit vollständigem Logging
        
        Args:
            test_name: Name des Tests
            test_func: Test-Funktion
            test_module: Test-Modul (für Setup/Teardown)
            
        Returns:
            Test-Ergebnis
        """
        result = {
            "test_name": test_name,
            "status": "unknown",
            "duration": 0,
            "error": None,
            "debug_report": None
        }
        
        start_time = os.times().elapsed
        
        # Start-Log
        self.log_manager.write_test_log(test_name, "start", "start", {
            "test_function": test_func.__name__ if hasattr(test_func, "__name__") else test_name
        })
        
        try:
            # Setup (falls vorhanden)
            if hasattr(test_module, "setUp"):
                self.log_manager.write_test_log(test_name, "setup", "info", {})
                test_module.setUp()
            
            # Führe Test aus
            self.log_manager.write_test_log(test_name, "test_execution", "info", {})
            test_func()
            
            # Test erfolgreich
            result["status"] = "passed"
            self.log_manager.write_test_log(test_name, "end", "success", {
                "result": "passed"
            })
        
        except AssertionError as e:
            # Assertion-Fehler
            result["status"] = "failed"
            result["error"] = str(e)
            
            self.log_manager.write_test_log(test_name, "assertion_error", "error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Auto-Debug
            debug_info = self.auto_debug(test_name, e)
            result["debug_report"] = debug_info
        
        except Exception as e:
            # Andere Exception
            result["status"] = "error"
            result["error"] = str(e)
            
            self.log_manager.write_test_log(test_name, "exception", "error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Auto-Debug
            debug_info = self.auto_debug(test_name, e)
            result["debug_report"] = debug_info
        
        finally:
            # Teardown (falls vorhanden)
            if hasattr(test_module, "tearDown"):
                try:
                    self.log_manager.write_test_log(test_name, "teardown", "info", {})
                    test_module.tearDown()
                except Exception as e:
                    self.log_manager.write_test_log(test_name, "teardown_error", "error", {
                        "error": str(e)
                    })
            
            end_time = os.times().elapsed
            result["duration"] = end_time - start_time
        
        return result
    
    def auto_debug(self, test_name: str, exception: Exception) -> Dict[str, Any]:
        """
        Automatisches Debugging nach Test-Fehler
        
        Args:
            test_name: Name des Tests
            exception: Aufgetretene Exception
            
        Returns:
            Debug-Informationen
        """
        # Lese alle Logs für diesen Test
        test_logs = self.log_manager.read_test_logs(test_name=test_name)
        
        # Prüfe System-State (versuche aus Logs zu extrahieren)
        test_context = {
            "test": test_name,
            "exception": str(exception),
            "model_id": None,
            "model_type": "text"
        }
        
        # Versuche Modell-Info aus Logs zu extrahieren
        for log in test_logs:
            data = log.get("data", {})
            if "model_id" in data:
                test_context["model_id"] = data["model_id"]
            if "model_type" in data:
                test_context["model_type"] = data["model_type"]
        
        # Identifiziere Probleme
        problems = self.system_checker.identify_problems(test_context)
        
        # Generiere Debug-Report
        debug_report = self.generate_debug_report(test_name, problems, test_logs)
        
        return debug_report
    
    def generate_debug_report(self, test_name: str, problems: List[Dict[str, Any]],
                             relevant_logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generiert strukturierten Debug-Report
        
        Args:
            test_name: Name des Tests
            problems: Liste von Problemen
            relevant_logs: Relevante Log-Einträge
            
        Returns:
            Debug-Report
        """
        # Hole System-State
        system_state = {}
        if problems:
            # Versuche Modell-Typ aus Problemen zu extrahieren
            for problem in problems:
                if "model_type" in str(problem):
                    # Prüfe System-State
                    model_type = "text"  # Default
                    state = self.system_checker.check_model_state(model_type)
                    system_state[f"{model_type}_model"] = state
        
        # Schreibe Debug-Report
        report = self.log_manager.write_debug_report(
            test_name=test_name,
            problems=problems,
            system_state=system_state,
            relevant_logs=relevant_logs or []
        )
        
        return report
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generiert Gesamt-Report
        
        Args:
            results: Liste von Test-Ergebnissen
            
        Returns:
            Zusammenfassung
        """
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for result in results:
            if "tests" in result:
                # Modul-Ergebnis
                total_tests += result.get("total", 0)
                passed_tests += result.get("passed", 0)
                failed_tests += result.get("failed", 0)
            else:
                # Einzelner Test
                total_tests += 1
                status = result.get("status", "unknown")
                if status == "passed":
                    passed_tests += 1
                elif status == "failed":
                    failed_tests += 1
                else:
                    error_tests += 1
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "results": results
        }
        
        # Speichere Summary
        summary_file = self.log_manager.log_dir / "test_summary.json"
        try:
            import json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Fehler beim Speichern der Summary: {e}")
        
        return summary
    
    def _discover_test_modules(self) -> List[str]:
        """
        Findet alle Test-Module
        
        Returns:
            Liste von Test-Modul-Namen
        """
        test_dir = Path(__file__).parent
        test_modules = []
        
        for file in test_dir.glob("test_*.py"):
            module_name = file.stem
            test_modules.append(module_name)
        
        return test_modules


def generate_debug_report(test_name: str, problems: List[Dict[str, Any]],
                         relevant_logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Hilfsfunktion zum Generieren von Debug-Reports
    
    Args:
        test_name: Name des Tests
        problems: Liste von Problemen
        relevant_logs: Relevante Log-Einträge
        
    Returns:
        Debug-Report
    """
    log_manager = LogManager()
    system_checker = SystemChecker()
    
    # Hole System-State
    system_state = {}
    
    # Generiere Report
    report = log_manager.write_debug_report(
        test_name=test_name,
        problems=problems,
        system_state=system_state,
        relevant_logs=relevant_logs or []
    )
    
    return report









