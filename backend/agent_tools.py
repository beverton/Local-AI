"""
Agent Tools - Tool-System für Agenten
"""
import os
import logging
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Globale Referenzen (werden von main.py gesetzt)
_model_manager = None
_image_manager = None
_agent_manager = None
_workspace_root = None


def initialize_tools(model_manager, image_manager, agent_manager, workspace_root: str):
    """Initialisiert die Tools mit den benötigten Managern"""
    global _model_manager, _image_manager, _agent_manager, _workspace_root
    _model_manager = model_manager
    _image_manager = image_manager
    _agent_manager = agent_manager
    _workspace_root = workspace_root


def _validate_path(file_path: str) -> str:
    """Validiert und normalisiert einen Dateipfad (nur innerhalb Workspace)"""
    # Normalisiere Pfad
    normalized = os.path.normpath(file_path)
    
    # Wenn relativer Pfad, mache ihn absolut relativ zum Workspace
    if not os.path.isabs(normalized):
        normalized = os.path.join(_workspace_root, normalized)
    else:
        normalized = os.path.abspath(normalized)
    
    # Prüfe ob Pfad innerhalb Workspace liegt
    workspace_abs = os.path.abspath(_workspace_root)
    if not normalized.startswith(workspace_abs):
        raise ValueError(f"Pfad außerhalb des Workspace: {file_path}")
    
    return normalized


def read_file(file_path: str) -> str:
    """
    Liest eine Datei
    
    Args:
        file_path: Der Pfad zur Datei (relativ zum Workspace oder absolut)
        
    Returns:
        Der Dateiinhalt als String
    """
    if not _workspace_root:
        raise RuntimeError("Tools nicht initialisiert")
    
    validated_path = _validate_path(file_path)
    
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
    
    if not os.path.isfile(validated_path):
        raise ValueError(f"Pfad ist keine Datei: {file_path}")
    
    try:
        with open(validated_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Datei gelesen: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Fehler beim Lesen der Datei {file_path}: {e}")
        raise


def write_file(file_path: str, content: str) -> bool:
    """
    Schreibt eine Datei
    
    Args:
        file_path: Der Pfad zur Datei (relativ zum Workspace oder absolut)
        content: Der Inhalt der Datei
        
    Returns:
        True wenn erfolgreich
    """
    if not _workspace_root:
        raise RuntimeError("Tools nicht initialisiert")
    
    validated_path = _validate_path(file_path)
    
    # Erstelle Verzeichnis falls nötig
    os.makedirs(os.path.dirname(validated_path), exist_ok=True)
    
    try:
        with open(validated_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Datei geschrieben: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Schreiben der Datei {file_path}: {e}")
        raise


def execute_code(code: str, language: str = "python") -> Dict[str, Any]:
    """
    Führt Code aus (sandboxed)
    
    Args:
        code: Der Code
        language: Die Programmiersprache (aktuell nur "python")
        
    Returns:
        Dict mit "output", "error", "success"
    """
    if language != "python":
        return {
            "success": False,
            "error": f"Sprache '{language}' nicht unterstützt",
            "output": ""
        }
    
    # TODO: Implementiere sichere Code-Ausführung
    # Für jetzt: Nur Syntax-Check
    try:
        compile(code, "<string>", "exec")
        return {
            "success": True,
            "error": None,
            "output": "Code-Syntax ist gültig (Ausführung deaktiviert aus Sicherheitsgründen)"
        }
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax-Fehler: {str(e)}",
            "output": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Fehler: {str(e)}",
            "output": ""
        }


def generate_image(prompt: str, model_id: Optional[str] = None, 
                   negative_prompt: str = "", num_inference_steps: int = 20,
                   guidance_scale: float = 7.5, width: int = 1024, height: int = 1024) -> str:
    """
    Generiert ein Bild
    
    Args:
        prompt: Der Prompt für die Bildgenerierung
        model_id: Optional: Spezifisches Modell (sonst aktuelles)
        negative_prompt: Negativer Prompt
        num_inference_steps: Anzahl der Inferenz-Schritte
        guidance_scale: Guidance Scale
        width: Bildbreite
        height: Bildhöhe
        
    Returns:
        Base64-kodiertes Bild
        
    Raises:
        RuntimeError: Wenn ImageManager nicht verfügbar ist
        RuntimeError: Wenn das angeforderte Modell nicht geladen ist (sollte über API geladen werden)
    """
    if not _image_manager:
        raise RuntimeError("ImageManager nicht verfügbar (diffusers/xformers nicht installiert)")
    
    # Prüfe ob Modell geladen ist - KEIN synchrones Laden mehr!
    # Modell-Laden sollte über API-Endpunkt erfolgen (asynchron)
    if model_id:
        current_model = _image_manager.get_current_model()
        if current_model != model_id or not _image_manager.is_model_loaded():
            raise RuntimeError(
                f"Bildmodell {model_id} ist nicht geladen. "
                f"Bitte laden Sie das Modell zuerst über den API-Endpunkt /image/models/load. "
                f"Aktuelles Modell: {current_model}"
            )
    elif not _image_manager.is_model_loaded():
        # Prüfe ob ein Default-Modell verfügbar wäre
        available_models = _image_manager.get_available_models()
        if available_models:
            default_model_id = list(available_models.keys())[0]
            raise RuntimeError(
                f"Kein Bildmodell geladen. "
                f"Bitte laden Sie ein Modell über den API-Endpunkt /image/models/load. "
                f"Verfügbare Modelle: {', '.join(available_models.keys())}"
            )
        else:
            raise RuntimeError("Kein Bildmodell verfügbar")
    
    try:
        # Generiere Bild
        image = _image_manager.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        )
        
        if image is None:
            raise RuntimeError("Bildgenerierung fehlgeschlagen")
        
        # Konvertiere zu Base64
        image_base64 = _image_manager.image_to_base64(image)
        logger.info(f"Bild generiert: {len(prompt)} Zeichen Prompt")
        return image_base64
        
    except Exception as e:
        logger.error(f"Fehler bei der Bildgenerierung: {e}")
        raise


def describe_image(image_base64: str, model_id: Optional[str] = None) -> str:
    """
    Beschreibt ein Bild
    
    Args:
        image_base64: Base64-kodiertes Bild
        model_id: Optional: Spezifisches Modell (sonst aktuelles Text-Modell)
        
    Returns:
        Bildbeschreibung als Text
    """
    if not _model_manager:
        raise RuntimeError("ModelManager nicht verfügbar")
    
    # TODO: Implementiere Vision-Modell-Unterstützung
    # Für jetzt: Verwende Text-Modell mit einfacher Beschreibung
    
    # Prüfe ob Modell geladen ist - KEIN synchrones Laden mehr!
    # Modell-Laden sollte über API-Endpunkt erfolgen (asynchron)
    if model_id:
        current_model = _model_manager.get_current_model()
        if current_model != model_id or not _model_manager.is_model_loaded():
            raise RuntimeError(
                f"Textmodell {model_id} ist nicht geladen. "
                f"Bitte laden Sie das Modell zuerst über den API-Endpunkt /models/load. "
                f"Aktuelles Modell: {current_model}"
            )
    elif not _model_manager.is_model_loaded():
        # Prüfe ob ein Default-Modell verfügbar wäre
        default_model = _model_manager.config.get("default_model")
        if default_model:
            raise RuntimeError(
                f"Kein Textmodell geladen. "
                f"Bitte laden Sie ein Modell über den API-Endpunkt /models/load. "
                f"Default-Modell: {default_model}"
            )
        else:
            raise RuntimeError("Kein Modell geladen und kein Default-Modell konfiguriert")
    
    try:
        # Dekodiere Bild
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Für jetzt: Einfache Beschreibung basierend auf Bildgröße
        # TODO: Implementiere echte Vision-Modell-Integration
        width, height = image.size
        description = f"Ein Bild mit den Abmessungen {width}x{height} Pixeln. "
        description += "Bildbeschreibung wird derzeit noch nicht vollständig unterstützt."
        
        logger.info("Bild beschrieben (vereinfacht)")
        return description
        
    except Exception as e:
        logger.error(f"Fehler bei der Bildbeschreibung: {e}")
        raise


def call_agent(conversation_id: str, agent_id: str, message: str) -> str:
    """
    Ruft einen anderen Agenten auf
    
    Args:
        conversation_id: Die ID der Conversation
        agent_id: Die ID des Ziel-Agenten
        message: Die Nachricht
        
    Returns:
        Die Antwort des Agenten
    """
    if not _agent_manager:
        raise RuntimeError("AgentManager nicht verfügbar")
    
    return _agent_manager.call_agent(conversation_id, None, agent_id, message)


def web_search(query: str, max_results: int = 5, timeout: float = 5.0) -> Dict[str, Any]:
    """
    🔧 VERBESSERTE WEB-SUCHE mit Wikipedia-Integration
    
    Führt eine Websuche durch und gibt strukturierte Ergebnisse zurück.
    Strategie: 1. Wikipedia (für Fakten), 2. DuckDuckGo (allgemein), 3. Google Fallback
    
    Args:
        query: Die Suchanfrage
        max_results: Maximale Anzahl der Ergebnisse (Standard: 5)
        timeout: Timeout in Sekunden (Standard: 5.0)
        
    Returns:
        Dict mit "results" (Liste von Ergebnissen) und "summary"
    """
    results = []
    
    # STRATEGIE 1: Wikipedia API (sehr zuverlässig für faktische Fragen)
    try:
        # Wikipedia Suche (Deutsch)
        wiki_search_url = f"https://de.wikipedia.org/w/api.php?action=opensearch&search={requests.utils.quote(query)}&limit={max_results}&namespace=0&format=json"
        
        headers = {'User-Agent': 'LocalAI/1.0'}
        wiki_response = requests.get(wiki_search_url, headers=headers, timeout=timeout)
        
        if wiki_response.status_code == 200:
            wiki_data = wiki_response.json()
            # Format: [search_term, [titles], [descriptions], [urls]]
            if len(wiki_data) >= 4:
                titles = wiki_data[1]
                descriptions = wiki_data[2]
                urls = wiki_data[3]
                
                for i in range(min(len(titles), max_results)):
                    if titles[i] and urls[i]:
                        results.append({
                            "title": titles[i],
                            "url": urls[i],
                            "snippet": descriptions[i] if i < len(descriptions) else f"Wikipedia-Artikel über {titles[i]}"
                        })
                        
        if results:
            logger.info(f"Wikipedia-Suche erfolgreich: {query} - {len(results)} Ergebnisse")
    except Exception as e:
        logger.debug(f"Wikipedia-Suche fehlgeschlagen: {e}")
    
    # STRATEGIE 2: DuckDuckGo (falls Wikipedia keine Ergebnisse)
    if not results:
        try:
            api_url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1&skip_disambig=1"
            headers = {'User-Agent': 'LocalAI/1.0'}
            
            response = requests.get(api_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Extrahiere Instant Answer
            if data.get('Abstract'):
                results.append({
                    "title": data.get('Heading', query.title()),
                    "url": data.get('AbstractURL', f"https://duckduckgo.com/?q={requests.utils.quote(query)}"),
                    "snippet": data.get('Abstract', '')
                })
            
            # Extrahiere Related Topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        "title": topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                        "url": topic.get('FirstURL', ''),
                        "snippet": topic.get('Text', '')
                    })
                    
            if results:
                logger.info(f"DuckDuckGo-Suche erfolgreich: {query} - {len(results)} Ergebnisse")
        except Exception as e:
            logger.debug(f"DuckDuckGo-Suche fehlgeschlagen: {e}")
    
    # STRATEGIE 3: Fallback mit Google-Suchlink (nur wenn keine Ergebnisse)
    if not results:
        logger.warning(f"Keine Web-Ergebnisse gefunden für: {query}, verwende Google-Fallback")
        results = [{
            "title": f"Suche nach: {query}",
            "url": f"https://www.google.com/search?q={requests.utils.quote(query)}",
            "snippet": f"Keine direkten Ergebnisse gefunden. Suchen Sie auf Google nach '{query}' für mehr Informationen."
        }]
    
    summary = f"Gefunden: {len(results)} Ergebnis(se) für '{query}'"
    logger.info(f"Web-Suche abgeschlossen: {query} - {len(results)} Ergebnisse")
    
    return {
        "results": results,
        "summary": summary,
        "query": query
    }


def list_directory(directory_path: str = ".") -> List[Dict[str, Any]]:
    """
    Listet den Inhalt eines Verzeichnisses auf
    
    Args:
        directory_path: Der Pfad zum Verzeichnis (relativ zum Workspace oder absolut)
        
    Returns:
        Liste von Dicts mit "name", "type" (file/directory), "size" (nur für Dateien)
    """
    if not _workspace_root:
        raise RuntimeError("Tools nicht initialisiert")
    
    validated_path = _validate_path(directory_path)
    
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory_path}")
    
    if not os.path.isdir(validated_path):
        raise ValueError(f"Pfad ist kein Verzeichnis: {directory_path}")
    
    try:
        items = []
        for item_name in os.listdir(validated_path):
            item_path = os.path.join(validated_path, item_name)
            item_info = {
                "name": item_name,
                "type": "directory" if os.path.isdir(item_path) else "file"
            }
            
            if os.path.isfile(item_path):
                item_info["size"] = os.path.getsize(item_path)
            
            items.append(item_info)
        
        # Sortiere: Verzeichnisse zuerst, dann Dateien
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        
        logger.info(f"Verzeichnis aufgelistet: {directory_path} - {len(items)} Einträge")
        return items
        
    except Exception as e:
        logger.error(f"Fehler beim Auflisten des Verzeichnisses {directory_path}: {e}")
        raise


def delete_file(file_path: str) -> bool:
    """
    Löscht eine Datei
    
    Args:
        file_path: Der Pfad zur Datei (relativ zum Workspace oder absolut)
        
    Returns:
        True wenn erfolgreich
    """
    if not _workspace_root:
        raise RuntimeError("Tools nicht initialisiert")
    
    validated_path = _validate_path(file_path)
    
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
    
    if not os.path.isfile(validated_path):
        raise ValueError(f"Pfad ist keine Datei: {file_path}")
    
    try:
        os.remove(validated_path)
        logger.info(f"Datei gelöscht: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Löschen der Datei {file_path}: {e}")
        raise


def file_exists(file_path: str) -> bool:
    """
    Prüft ob eine Datei existiert
    
    Args:
        file_path: Der Pfad zur Datei (relativ zum Workspace oder absolut)
        
    Returns:
        True wenn die Datei existiert, False sonst
    """
    if not _workspace_root:
        raise RuntimeError("Tools nicht initialisiert")
    
    try:
        validated_path = _validate_path(file_path)
        exists = os.path.exists(validated_path) and os.path.isfile(validated_path)
        logger.info(f"Datei-Existenz geprüft: {file_path} - {exists}")
        return exists
    except Exception as e:
        logger.error(f"Fehler beim Prüfen der Datei-Existenz {file_path}: {e}")
        return False

