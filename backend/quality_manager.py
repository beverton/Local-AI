"""
Quality Manager - Validiert und verbessert Antwort-Qualität
Inspiriert von Perplexity's Quality Management
"""
import logging
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityManager:
    """Verwaltet Qualität von AI-Antworten - nutzt automatisch Web-Search - für ALLE Chat-Modelle"""
    
    def __init__(self, web_search_function, settings_path: str = "data/quality_settings.json"):
        """
        Args:
            web_search_function: Funktion für Web-Suche (aus agent_tools)
            settings_path: Pfad zu Quality Settings JSON
        """
        self.web_search = web_search_function
        self.settings_path = settings_path
        self.settings = self._load_settings()
        self.feedback_history = []  # Nutzerrückmeldungen
        self.source_cache = {}  # Cache für Quellen-Validierungen
    
    def _load_settings(self) -> Dict[str, Any]:
        """Lädt Quality Settings"""
        default_settings = {
            "web_validation": True,  # Web-Search Validierung
            "contradiction_check": True,  # Widerspruchsprüfung
            "hallucination_check": True,  # Halluzinations-Erkennung
            "actuality_check": True,  # Aktualitätsprüfung
            "source_quality_check": True,  # Quellen-Qualitätsbewertung
            "completeness_check": True,  # Vollständigkeitsprüfung
            "auto_web_search": False  # Automatischer Web-Search (Default: False um Blockierung zu vermeiden)
        }
        
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Merge mit Defaults (für neue Optionen)
                    default_settings.update(loaded)
                    return default_settings
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Quality Settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Speichert Quality Settings"""
        try:
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Quality Settings: {e}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Gibt aktuelle Quality Settings zurück"""
        return self.settings.copy()
    
    def update_setting(self, key: str, value: bool):
        """Aktualisiert eine Quality Setting"""
        if key in self.settings:
            self.settings[key] = value
            self.save_settings()
            logger.info(f"Quality Setting '{key}' auf {value} gesetzt")
    
    def validate_response(self, response: str, question: str, auto_search: bool = True) -> Dict[str, Any]:
        """
        Validiert eine Antwort - nutzt automatisch Web-Search wenn auto_search=True
        
        Args:
            response: Die generierte Antwort
            question: Die ursprüngliche Frage
            auto_search: Automatisch Web-Search durchführen (Default: True)
        
        Returns:
            {
                "valid": bool,
                "confidence": float,
                "issues": List[str],
                "sources": List[Dict],  # Quellen aus Web-Search
                "suggestions": List[str]
            }
        """
        sources = []
        
        # AUTOMATISCHER WEB-SEARCH (smart + quality_only) - nur wenn aktiviert
        # 1. Smart: Nur bei Fragen die Web-Search benötigen (z.B. aktuelle Infos, Fakten)
        # 2. Quality-only: Für Quality Management Validierung nach Antwort-Generierung
        if auto_search and self.settings.get("auto_web_search", True):
            # Prüfe ob Frage Web-Search benötigt (aktuelle Infos, Fakten, etc.)
            needs_search = self._needs_web_search(question)
            
            if needs_search:
                try:
                    # Führe Web-Search für die Frage durch (mit kurzem Timeout)
                    # WICHTIG: Web-Search hat kurzes Timeout (3 Sekunden) um Blockierung zu vermeiden
                    search_results = self.web_search(question, max_results=5, timeout=3.0)
                    if search_results and "results" in search_results:
                        sources = search_results["results"]
                        logger.info(f"Web-Search durchgeführt: {len(sources)} Quellen gefunden")
                except Exception as search_error:
                    logger.warning(f"Web-Search fehlgeschlagen oder Timeout: {search_error}")
                    # Weiter ohne Quellen - nicht kritisch, blockiert nicht
                    sources = []
        
        validation = {
            "valid": True,
            "confidence": 1.0,
            "issues": [],
            "sources": sources,
            "suggestions": []
        }
        
        # Quality Checks - nur wenn in Settings aktiviert
        # 1. Prüfe auf Widersprüche (wenn mehrere Quellen vorhanden)
        if self.settings.get("contradiction_check", True) and sources and len(sources) > 1:
            contradictions = self._check_contradictions(response, sources)
            if contradictions:
                validation["issues"].extend(contradictions)
                validation["confidence"] *= 0.7
                validation["valid"] = False
        
        # 2. Prüfe auf Halluzinationen (Fakten die nicht in Quellen stehen)
        if self.settings.get("hallucination_check", True) and sources:
            hallucinations = self._check_hallucinations(response, sources)
            if hallucinations:
                validation["issues"].extend(hallucinations)
                validation["confidence"] *= 0.8
        
        # 3. Prüfe Aktualität (wenn Quellen vorhanden)
        if self.settings.get("actuality_check", True) and sources:
            outdated = self._check_actuality(sources)
            if outdated:
                validation["issues"].append("Einige Quellen könnten veraltet sein")
                validation["suggestions"].append("Bitte prüfen Sie die Aktualität der Quellen")
        
        # 4. Prüfe Quellen-Qualität
        if self.settings.get("source_quality_check", True) and sources:
            quality_score = self._rate_source_quality(sources)
            if quality_score < 0.6:
                validation["issues"].append("Quellen haben niedrige Qualität")
                validation["confidence"] *= quality_score
        
        # 5. Prüfe auf vollständige Antwort
        if self.settings.get("completeness_check", True):
            if not self._check_completeness(response, question):
                validation["issues"].append("Antwort könnte unvollständig sein")
                validation["suggestions"].append("Bitte fragen Sie nach, wenn Sie mehr Details benötigen")
        
        # 6. Web-Validation (wenn aktiviert)
        if self.settings.get("web_validation", True) and sources:
            # Zusätzliche Validierung gegen Web-Quellen
            web_validation_issues = self._validate_against_web_sources(response, sources)
            if web_validation_issues:
                validation["issues"].extend(web_validation_issues)
                validation["confidence"] *= 0.9
        
        return validation
    
    def _needs_web_search(self, question: str) -> bool:
        """
        Prüft ob Frage Web-Search benötigt
        
        Returns:
            True wenn Web-Search sinnvoll ist
        """
        question_lower = question.lower()
        
        # Indikatoren für Web-Search-Bedarf
        search_indicators = [
            "wetter", "weather", "aktuelle", "aktuell", "heute", "morgen",
            "wie viel", "was kostet", "wo ist", "wann ist", "wer ist",
            "definition", "was bedeutet", "erkläre", "was ist",
            "news", "neuigkeiten", "nachrichten"
        ]
        
        return any(indicator in question_lower for indicator in search_indicators)
    
    def _validate_against_web_sources(self, response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Zusätzliche Validierung gegen Web-Quellen"""
        issues = []
        # Implementierung: Prüfe Response gegen Quellen-Inhalte
        # Vereinfachte Implementierung - könnte erweitert werden
        return issues
    
    def _check_contradictions(self, response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Prüft auf Widersprüche zwischen Quellen"""
        # Einfache Implementierung: Prüfe auf widersprüchliche Zahlen/Daten
        contradictions = []
        
        # Beispiel: Prüfe auf widersprüchliche Zahlen
        # (Vereinfacht - echte Implementierung würde NLP verwenden)
        
        return contradictions
    
    def _check_hallucinations(self, response: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Prüft auf Halluzinationen (Fakten nicht in Quellen)"""
        hallucinations = []
        
        # Vereinfachte Implementierung
        # Echte Implementierung würde:
        # - Response in Fakten zerlegen
        # - Jeden Fakt gegen Quellen prüfen
        # - Nicht gefundene Fakten als mögliche Halluzinationen markieren
        
        return hallucinations
    
    def _check_actuality(self, sources: List[Dict[str, Any]]) -> bool:
        """Prüft ob Quellen aktuell sind"""
        # Prüfe Datum in Quellen (falls vorhanden)
        current_year = datetime.now().year
        
        for source in sources:
            source_date = source.get("date")
            if source_date:
                try:
                    year = int(source_date.split("-")[0]) if "-" in source_date else int(source_date)
                    if year < current_year - 3:  # Älter als 3 Jahre
                        return True
                except:
                    pass
        
        return False
    
    def _rate_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Bewertet Qualität der Quellen (0.0 - 1.0)"""
        # Bevorzuge seriöse Quellen (Behörden, Unis, etablierte Medien)
        quality_keywords = {
            "high": ["gov", "edu", "org", "wikipedia", "arxiv"],
            "medium": ["com", "net"],
            "low": ["blog", "forum"]
        }
        
        total_score = 0.0
        for source in sources:
            url = source.get("url", "").lower()
            if any(kw in url for kw in quality_keywords["high"]):
                total_score += 1.0
            elif any(kw in url for kw in quality_keywords["medium"]):
                total_score += 0.7
            else:
                total_score += 0.4
        
        return total_score / len(sources) if sources else 0.0
    
    def _check_completeness(self, response: str, question: str) -> bool:
        """Prüft ob Antwort vollständig ist"""
        # Einfache Heuristik: Antwort sollte mindestens 20% der Frage-Länge haben
        min_length = len(question) * 0.2
        return len(response) >= min_length
    
    def add_feedback(self, response_id: str, feedback: Dict[str, Any]):
        """Fügt Nutzerrückmeldung hinzu"""
        self.feedback_history.append({
            "response_id": response_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Lerne aus Feedback (für zukünftige Verbesserungen)
        logger.info(f"Feedback erhalten für Response {response_id}: {feedback}")
    
    def get_quality_report(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Gibt Quality-Report für eine Response zurück"""
        # Finde Feedback für diese Response
        feedbacks = [f for f in self.feedback_history if f["response_id"] == response_id]
        
        if not feedbacks:
            return None
        
        return {
            "response_id": response_id,
            "feedbacks": feedbacks,
            "average_rating": sum(f["feedback"].get("rating", 0) for f in feedbacks) / len(feedbacks) if feedbacks else 0
        }

