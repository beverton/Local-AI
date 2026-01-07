"""
System-State-Checker für Tests
Prüft Modell-Status, API-Endpunkte und Logs
"""
import json
import os
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path


class SystemChecker:
    """Prüft System-Status für automatisches Debugging"""
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000", 
                 debug_log_path: Optional[str] = None):
        """
        Initialisiert System-Checker
        
        Args:
            api_base_url: Base URL der API
            debug_log_path: Pfad zur debug.log Datei
        """
        self.api_base_url = api_base_url.rstrip('/')
        
        if debug_log_path is None:
            # Verwende .cursor/debug.log im Workspace-Root
            workspace_root = Path(__file__).parent.parent
            debug_log_path = workspace_root / ".cursor" / "debug.log"
        
        self.debug_log_path = Path(debug_log_path)
    
    def check_model_state(self, model_type: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Prüft Modell-Status über API
        
        Args:
            model_type: "text", "image" oder "audio"
            model_id: Optional - spezifische Modell-ID
            
        Returns:
            Status-Informationen
        """
        state = {
            "is_loaded": False,
            "is_loading": False,
            "current_model": None,
            "error": None,
            "logs": []
        }
        
        try:
            # Prüfe Status-Endpoint
            if model_type == "text":
                status_endpoint = f"{self.api_base_url}/models/load/status"
            elif model_type == "image":
                status_endpoint = f"{self.api_base_url}/image/models/load/status"
            elif model_type == "audio":
                status_endpoint = f"{self.api_base_url}/audio/models/load/status"
            else:
                state["error"] = f"Unbekannter Modell-Typ: {model_type}"
                return state
            
            response = requests.get(status_endpoint, timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                state["is_loading"] = status_data.get("loading", False)
                state["current_model"] = status_data.get("model_id")
                state["error"] = status_data.get("error")
                
                # Prüfe ob Modell geladen ist
                if not state["is_loading"] and state["current_model"]:
                    if model_id is None or state["current_model"] == model_id:
                        state["is_loaded"] = True
            else:
                state["error"] = f"Status-Endpoint antwortete mit {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            state["error"] = f"API nicht erreichbar: {str(e)}"
        except Exception as e:
            state["error"] = f"Fehler beim Prüfen des Modell-Status: {str(e)}"
        
        # Lese relevante Logs
        state["logs"] = self._get_relevant_logs(model_type, model_id)
        
        return state
    
    def check_api_endpoint(self, endpoint: str, method: str = "GET", 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prüft ob API-Endpunkt funktioniert
        
        Args:
            endpoint: API-Endpunkt (z.B. "/models" oder "/chat")
            method: HTTP-Methode
            data: Optional - Request-Daten
            
        Returns:
            Status-Informationen
        """
        result = {
            "status": "error",
            "response": {},
            "error": None,
            "status_code": None
        }
        
        try:
            url = f"{self.api_base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=5)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=5)
            else:
                result["error"] = f"Unbekannte Methode: {method}"
                return result
            
            result["status_code"] = response.status_code
            
            if response.status_code < 400:
                result["status"] = "ok"
                try:
                    result["response"] = response.json()
                except:
                    result["response"] = {"text": response.text}
            else:
                result["status"] = "error"
                try:
                    error_data = response.json()
                    result["error"] = error_data.get("detail", f"HTTP {response.status_code}")
                except:
                    result["error"] = f"HTTP {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            result["status"] = "error"
            result["error"] = f"API nicht erreichbar: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"Fehler: {str(e)}"
        
        return result
    
    def check_logs_for_errors(self, since_timestamp: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sucht in Logs nach Fehlern
        
        Args:
            since_timestamp: Optional - nur Logs nach diesem Timestamp
            
        Returns:
            Dictionary mit "errors" und "warnings"
        """
        result = {
            "errors": [],
            "warnings": []
        }
        
        if not self.debug_log_path.exists():
            return result
        
        try:
            with open(self.debug_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        
                        # Prüfe Timestamp
                        if since_timestamp and log_entry.get("timestamp", 0) < since_timestamp:
                            continue
                        
                        # Prüfe auf Fehler
                        message = log_entry.get("message", "").lower()
                        location = log_entry.get("location", "")
                        
                        if "error" in message or "exception" in message or "traceback" in message:
                            result["errors"].append({
                                "timestamp": log_entry.get("timestamp"),
                                "location": location,
                                "message": log_entry.get("message"),
                                "data": log_entry.get("data", {})
                            })
                        elif "warning" in message:
                            result["warnings"].append({
                                "timestamp": log_entry.get("timestamp"),
                                "location": location,
                                "message": log_entry.get("message"),
                                "data": log_entry.get("data", {})
                            })
                    
                    except json.JSONDecodeError:
                        # Nicht-JSON Zeile - prüfe auf Text-Fehler
                        line_lower = line.lower()
                        if "error" in line_lower or "exception" in line_lower or "traceback" in line_lower:
                            result["errors"].append({
                                "timestamp": None,
                                "location": "unknown",
                                "message": line.strip(),
                                "data": {}
                            })
        
        except Exception as e:
            result["errors"].append({
                "timestamp": None,
                "location": "system_checker",
                "message": f"Fehler beim Lesen der Logs: {str(e)}",
                "data": {}
            })
        
        return result
    
    def identify_problems(self, test_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identifiziert konkrete Probleme basierend auf Kontext
        
        Args:
            test_context: Kontext-Informationen (test, exception, model_id, etc.)
            
        Returns:
            Liste von identifizierten Problemen
        """
        problems = []
        test_name = test_context.get("test", "unknown")
        exception = test_context.get("exception")
        model_id = test_context.get("model_id")
        model_type = test_context.get("model_type", "text")
        
        # Prüfe System-State
        if model_id:
            state = self.check_model_state(model_type, model_id)
            if state.get("error"):
                problems.append({
                    "type": "api_error",
                    "description": f"API-Fehler beim Prüfen des Modell-Status: {state['error']}",
                    "location": "system_checker.check_model_state",
                    "evidence": [state]
                })
            elif not state.get("is_loaded") and not state.get("is_loading"):
                problems.append({
                    "type": "model_not_loaded",
                    "description": f"Modell {model_id} ist nicht geladen",
                    "location": "model_state_check",
                    "evidence": [state]
                })
        
        # Prüfe Logs auf Fehler
        log_errors = self.check_logs_for_errors()
        if log_errors["errors"]:
            problems.append({
                "type": "log_errors",
                "description": f"{len(log_errors['errors'])} Fehler in Logs gefunden",
                "location": "debug.log",
                "evidence": log_errors["errors"][:5]  # Erste 5 Fehler
            })
        
        # Analysiere Exception
        if exception:
            exception_str = str(exception)
            if "model" in exception_str.lower() and "not loaded" in exception_str.lower():
                problems.append({
                    "type": "model_not_loaded_exception",
                    "description": f"Exception: {exception_str}",
                    "location": "test_execution",
                    "evidence": [{"exception": exception_str}]
                })
            elif "api" in exception_str.lower() or "connection" in exception_str.lower():
                problems.append({
                    "type": "api_connection_error",
                    "description": f"API-Verbindungsfehler: {exception_str}",
                    "location": "test_execution",
                    "evidence": [{"exception": exception_str}]
                })
            else:
                problems.append({
                    "type": "unknown_exception",
                    "description": f"Unbekannte Exception: {exception_str}",
                    "location": "test_execution",
                    "evidence": [{"exception": exception_str}]
                })
        
        return problems
    
    def _get_relevant_logs(self, model_type: str, model_id: Optional[str] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Holt relevante Log-Einträge für Modell
        
        Args:
            model_type: Modell-Typ
            model_id: Modell-ID
            limit: Maximale Anzahl von Logs
            
        Returns:
            Liste von relevanten Log-Einträgen
        """
        logs = []
        
        if not self.debug_log_path.exists():
            return logs
        
        try:
            with open(self.debug_log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                
                # Suche rückwärts (neueste zuerst)
                for line in reversed(all_lines[-100:]):  # Letzte 100 Zeilen
                    if not line.strip():
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        location = log_entry.get("location", "")
                        message = log_entry.get("message", "")
                        data = log_entry.get("data", {})
                        
                        # Prüfe ob Log relevant ist
                        is_relevant = False
                        
                        if model_id and model_id in str(data):
                            is_relevant = True
                        
                        if model_type == "text" and ("model_manager" in location or "main.py" in location):
                            is_relevant = True
                        elif model_type == "image" and "image_manager" in location:
                            is_relevant = True
                        elif model_type == "audio" and "whisper_manager" in location:
                            is_relevant = True
                        
                        if is_relevant:
                            logs.append(log_entry)
                            if len(logs) >= limit:
                                break
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            pass  # Ignoriere Fehler beim Lesen
        
        return logs










