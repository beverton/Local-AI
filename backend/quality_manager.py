"""
Quality Manager - Validiert und verbessert Antwort-Qualität
Inspiriert von Perplexity's Quality Management
"""
import logging
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from logging_utils import get_logger

# Strukturierter Logger
logger = get_logger(__name__, log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "quality_manager.log"))


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
        # WICHTIG: Alle Defaults auf False, damit Features nur aktiviert werden wenn explizit gewünscht
        default_settings = {
            "web_validation": False,  # Web-Search Validierung
            "contradiction_check": False,  # Widerspruchsprüfung
            "hallucination_check": False,  # Halluzinations-Erkennung
            "actuality_check": False,  # Aktualitätsprüfung
            "source_quality_check": False,  # Quellen-Qualitätsbewertung
            "completeness_check": False,  # Vollständigkeitsprüfung
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
    
    def is_coding_question(self, message: str) -> bool:
        """
        Erkennt ob eine Frage Coding-bezogen ist
        
        Args:
            message: Die Frage/Nachricht
            
        Returns:
            True wenn Coding-bezogen, False sonst
        """
        import re
        coding_patterns = [
            # Basis Coding-Keywords
            r"\b(schreibe|erstelle|implementiere|programmiere|code|funktion|klasse|def|class|function)\b",
            # Dateiendungen
            r"\.(py|js|ts|java|cpp|c|go|rs|php|rb|swift|kt|html|css|vue|jsx|tsx)\b",
            # Code-Struktur Keywords
            r"\b(algorithmus|syntax|debug|fehler|exception|try|catch|import|from|modul|datei)\b",
            # API/Web Keywords
            r"\b(api|endpoint|request|response|json|xml|http|rest|graphql)\b",
            # Spiele/Game Keywords
            r"\b(spiel|game|baue|nachbau|pong|snake|tetris|tictactoe|spiele|spieleentwicklung)\b",
            # Projekt Keywords
            r"\b(projekt|app|anwendung|programm|software|application|programmieren)\b",
            # Framework-spezifisch
            r"\b(pygame|flask|django|react|vue|angular|express|fastapi|spring|rails)\b",
            # Code-Struktur erweitert
            r"\b(klasse|funktion|methode|variable|array|liste|dictionary|objekt|interface)\b"
        ]
        message_lower = message.lower()
        return any(re.search(pattern, message_lower, re.IGNORECASE) for pattern in coding_patterns)
    
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
        if auto_search and self.settings.get("auto_web_search", False):
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
        
        # Quality Checks - nur wenn in Settings aktiviert (Default: False)
        # 1. Prüfe auf Widersprüche (wenn mehrere Quellen vorhanden)
        if self.settings.get("contradiction_check", False) and sources and len(sources) > 1:
            contradictions = self._check_contradictions(response, sources)
            if contradictions:
                validation["issues"].extend(contradictions)
                validation["confidence"] *= 0.7
                validation["valid"] = False
        
        # 2. Prüfe auf Halluzinationen (Fakten die nicht in Quellen stehen)
        # Hinweis: _check_hallucinations() prüft selbst den Toggle und kann auch ohne sources laufen
        if self.settings.get("hallucination_check", False):
            hallucinations = self._check_hallucinations(response, sources)
            if hallucinations:
                validation["issues"].extend(hallucinations)
                validation["confidence"] *= 0.8
        
        # 3. Prüfe Aktualität (wenn Quellen vorhanden)
        if self.settings.get("actuality_check", False) and sources:
            outdated = self._check_actuality(sources)
            if outdated:
                validation["issues"].append("Einige Quellen könnten veraltet sein")
                validation["suggestions"].append("Bitte prüfen Sie die Aktualität der Quellen")
        
        # 4. Prüfe Quellen-Qualität
        if self.settings.get("source_quality_check", False) and sources:
            quality_score = self._rate_source_quality(sources)
            if quality_score < 0.6:
                validation["issues"].append("Quellen haben niedrige Qualität")
                validation["confidence"] *= quality_score
        
        # 5. Prüfe auf vollständige Antwort
        if self.settings.get("completeness_check", False):
            if not self._check_completeness(response, question):
                validation["issues"].append("Antwort könnte unvollständig sein")
                validation["suggestions"].append("Bitte fragen Sie nach, wenn Sie mehr Details benötigen")
        
        # 6. Web-Validation (wenn aktiviert)
        if self.settings.get("web_validation", False) and sources:
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
    
    def format_sources_for_context(self, sources: List[Dict]) -> str:
        """
        Formatiert Web-Quellen als Kontext für LLM
        
        Args:
            sources: Liste von Suchergebnissen
            
        Returns:
            Formatierter Kontext-String
        """
        context_parts = []
        
        for idx, source in enumerate(sources[:5], 1):  # Max 5 Quellen
            title = source.get("title", "")
            snippet = source.get("snippet", "")
            url = source.get("url", "")
            
            # Formatiere als nummerierte Quelle
            context_parts.append(f"[{idx}] {title}\n{snippet}\nURL: {url}")
        
        return "\n\n".join(context_parts)
    
    def _is_factual_question(self, question: str) -> bool:
        """
        Prüft ob Frage faktische Informationen benötigt
        
        Erweitert von _needs_web_search() - prüft speziell auf Fakten
        
        Args:
            question: Die Frage
            
        Returns:
            True wenn Frage faktische Informationen benötigt
        """
        factual_keywords = [
            "was ist", "wie viel", "wann", "wo", "wer",
            "fakten", "daten", "statistik", "studie",
            "aktuell", "heute", "2024", "2025",
            "definition", "bedeutung", "erklärung"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in factual_keywords)
    
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
    
    def _check_hallucinations(self, response: str, sources: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Erkennt Halluzinationen in Response
        
        Prüft:
        - URLs (Existenz, Validität)
        - Fakten-Patterns (unbelegte Behauptungen)
        - Spezifische Zahlen/Daten ohne Quelle
        
        Args:
            response: Die zu prüfende Antwort
            sources: Optional - Quellen für erweiterte Validierung
            
        Returns:
            Liste von erkannten Problemen
        """
        issues = []
        
        # Toggle-Check: Ist Feature aktiv?
        if not self.settings.get("hallucination_check", False):
            return issues  # Skip wenn Toggle aus
        
        import re
        
        # 1. URL-Validierung
        urls = re.findall(r'https?://[^\s\)]+', response)
        for url in urls:
            # Bereinige URL (entferne trailing Punkt, etc.)
            url_clean = url.rstrip('.,;:!?')
            
            # Prüfe ob URL erreichbar ist (mit Cache)
            if not self._is_valid_url(url_clean):
                issues.append(f"Ungültige oder nicht erreichbare URL: {url_clean}")
        
        # 2. Hallucinations-Patterns
        hallucination_patterns = [
            (r'laut.*?Studie.*?\d{4}', "Unbestätigte Studien-Referenz"),
            (r'Wikipedia.*?sagt', "Direkte Wikipedia-Zitate ohne Link"),
            (r'Experten.*?(bestätigen|sagen|empfehlen)', "Unbelegte Experten-Aussagen"),
            (r'\d+%.*?der.*?(Menschen|Nutzer|Befragten)', "Unbelegte Statistiken"),
            (r'Forschung.*?zeigt', "Unbelegte Forschungs-Claims"),
        ]
        
        for pattern, issue_desc in hallucination_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(issue_desc)
        
        # 3. Spezifische Zahlen ohne Quelle-Referenz
        # Wenn Response Zahlen enthält, aber keine [1], [2] Referenzen
        has_numbers = bool(re.search(r'\d{1,3}[,.]?\d*\s*(Prozent|%|Jahre|Tage|Stunden)', response))
        has_source_refs = bool(re.search(r'\[\d+\]', response))
        
        if has_numbers and not has_source_refs and len(urls) == 0:
            issues.append("Zahlen/Statistiken ohne Quellen-Referenz")
        
        return issues
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Prüft ob URL erreichbar ist (mit Cache)
        
        Args:
            url: Die zu prüfende URL
            
        Returns:
            True wenn URL valide/erreichbar
        """
        # Cache-Check
        if url in self.source_cache:
            return self.source_cache[url]
        
        try:
            import requests
            from urllib.parse import urlparse
            
            # Basis-Validierung: URL-Format
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                self.source_cache[url] = False
                return False
            
            # HEAD-Request (schneller als GET)
            response = requests.head(url, timeout=3, allow_redirects=True)
            is_valid = response.status_code < 400
            
            # Cache Ergebnis
            self.source_cache[url] = is_valid
            return is_valid
            
        except Exception as e:
            logger.debug(f"URL-Validierung fehlgeschlagen für {url}: {e}")
            self.source_cache[url] = False
            return False
    
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
    
    def generate_retry_prompt(self, original_question: str, failed_response: str, issues: List[str]) -> str:
        """
        Generiert Feedback-Prompt für Retry nach fehlgeschlagener Validierung
        
        Args:
            original_question: Die ursprüngliche Frage
            failed_response: Die fehlerhafte Antwort
            issues: Liste von erkannten Problemen
            
        Returns:
            Feedback-Prompt für Regenerierung
        """
        issues_text = "\n- ".join(issues)
        
        return f"""FEEDBACK zur vorherigen Antwort:

Erkannte Probleme:
- {issues_text}

Bitte antworte ERNEUT auf die Frage: "{original_question}"

WICHTIG:
- Vermeide die oben genannten Probleme
- Keine erfundenen URLs oder Fakten
- Sei präzise und faktisch korrekt
- Wenn du etwas nicht sicher weißt, sage es explizit
- Nutze die bereitgestellten Quellen [1], [2], etc.
"""

