"""
Tool Detection - Zentrale heuristische Erkennung (Fallback)

Ziel: Keine duplizierte Pattern-Logik zwischen ChatAgent und QualityManager.
Diese Heuristik ist ein Fallback, wenn natives Function/Tool-Calling nicht genutzt werden kann.
"""

from __future__ import annotations

import re
from typing import Optional


def detect_web_search_query(message: str) -> Optional[str]:
    """
    Best-effort: Extrahiert eine sinnvolle Such-Query aus natürlicher Sprache.
    Returns:
      - query string (bereinigt) oder None, falls keine Websuche naheliegt.
    """
    if not message or not message.strip():
        return None

    message_original = message.strip()
    message_lower = message_original.lower()

    # URL-Erkennung: wenn URL erwähnt wird und es nach Suche aussieht, nutze komplette Nachricht
    url_pattern = r"https?://[^\s]+"
    if re.search(url_pattern, message_original):
        if re.search(r"\b(suche|finde|google|web|website|webseite|link|url)\b", message_lower):
            return message_original

    # Fragewörter/Keywords - Priorität (extrahieren Topic direkt)
    web_search_patterns = [
        r"^(.+?)\?.*?(?:suche|finde|google|web|website)",  # "Wer ist X? Suche..." -> "Wer ist X"
        r"wer\s+(?:ist|sind|war|waren)\s+(.+?)(?:\?|$)",
        r"was\s+(?:ist|wird|sind|war|bedeutet)\s+(.+?)(?:\?|$)",
        r"wo\s+(?:ist|sind|finde\s+ich|liegt|befindet\s+sich)\s+(.+?)(?:\?|$)",
        r"wann\s+(?:ist|war|findet|beginnt)\s+(.+?)(?:\?|$)",
        r"wie\s+(?:viel|viele|teuer|hoch)\s+(.+?)(?:\?|$)",
        # Such-Keywords mit spezifischem Kontext
        r"(?:suche|finde|google)\s+(?:nach\s+)?(?:informationen\s+)?(?:über|zu)\s+(.+)",
        r"website\s+(?:von|für|über|zu)\s+(.+)",
        r"webseite\s+(?:von|für|über|zu)\s+(.+)",
        # Wetter
        r"wetter\s+(?:in|für|heute|morgen)\s+(.+)",
        r"wettervorhersage\s+(?:für|in)\s+(.+)",
        # News/Aktualität
        r"(?:aktuelle|neueste)\s+(?:informationen|infos|news|nachrichten)\s+(?:über|zu|von)\s+(.+)",
        # Generisch am Ende
        r"(?:suche|finde|google)\s+(.+)",
    ]

    for pattern in web_search_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue

        query = match.group(1).strip()
        query = re.sub(r"[,\.;]+$", "", query)  # trailing punctuation
        query = re.sub(r"\s+(?:suche|finde|google|web|website|webseite|link|url).*?$", "", query).strip()
        if len(query) > 3:
            return query

    # Fallback Indikatoren (QualityManager-kompatibel)
    indicators = [
        "wetter", "weather", "aktuelle", "aktuell", "heute", "morgen",
        "wie viel", "was kostet", "wo ist", "wann ist", "wer ist",
        "definition", "was bedeutet", "news", "neuigkeiten", "nachrichten",
    ]
    if any(ind in message_lower for ind in indicators):
        return message_original

    return None


def needs_web_search(message: str) -> bool:
    """Boolean wrapper für Quality Checks."""
    return detect_web_search_query(message) is not None

