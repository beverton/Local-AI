# Analyse: Warum das lokale AI-Modell Tools nicht nutzen kann

## Problem-Beschreibung

Das lokale AI-Modell (QWEN) antwortet mit: *"Ich kann die Dateien im angegebenen Ordner nicht direkt durchsuchen oder auswerten, da ich keinen Zugriff darauf habe."*

## Root Cause Analysis

### 1. Tool-Architektur

Das lokale Modell nutzt Tools über den **ChatAgent**, nicht direkt:

- **ChatAgent** (`backend/agents/chat_agent.py`):
  - Erkennt Tool-Bedarf über **Pattern-Matching** (`_detect_tool_need()`)
  - Führt Tools automatisch aus, wenn Patterns erkannt werden
  - Gibt Tool-Ergebnisse an das Modell weiter

- **Modell selbst** (`backend/model_service.py`):
  - Erhält System-Prompt mit Tool-Informationen
  - Kann Tools **NICHT direkt aufrufen**
  - Muss auf ChatAgent warten, der Tools erkennt

### 2. Pattern-Matching-Limitationen

Der ChatAgent erkennt nur spezifische Patterns:

```python
# read_file Patterns:
- "lies die datei"
- "lade die datei"
- "zeige mir die datei"
- "read the file"
- "load the file"

# write_file Patterns:
- "schreibe in die datei"
- "speichere in die datei"
- "write to file"
- "save to file"

# list_directory Patterns:
- "zeige mir den inhalt von"
- "list contents of"
- "list directory"
```

**Problem:** Anfragen wie:
- "kannst du im conversation manager nachschauen"
- "prüfe den code in..."
- "analysiere die datei..."
- "schaue in den ordner..."

werden **NICHT erkannt**, weil sie nicht den exakten Patterns entsprechen.

### 3. System-Prompt-Problem

Der System-Prompt in `model_service.py` (Zeile 1375-1390) sagt:

```
VERFÜGBARE TOOLS:
- write_file(file_path, content): Erstellt oder überschreibt eine Datei...
- read_file(file_path): Liest eine Datei...
- list_directory(directory_path): Listet Verzeichnis-Inhalt auf...
- web_search(query): Führt eine Websuche durch.
```

**ABER:** Das Modell kann diese Tools **nicht direkt aufrufen**. Es muss darauf warten, dass der ChatAgent sie erkennt.

### 4. Kommunikationslücke

Das Modell "weiß" von Tools, kann sie aber nicht nutzen, weil:
1. Es keine direkte Tool-API hat
2. Es auf Pattern-Matching des ChatAgents angewiesen ist
3. Der ChatAgent viele natürliche Formulierungen nicht erkennt

## Lösungsansätze

### Option 1: Erweiterte Pattern-Erkennung (Empfohlen)

Erweitere `_detect_tool_need()` im ChatAgent um mehr natürliche Formulierungen:

```python
# Erweiterte Patterns für read_file:
- "nachschauen in"
- "prüfe die datei"
- "analysiere"
- "zeige mir den code in"
- "schaue in"
- "untersuche die datei"
```

### Option 2: Explizite Tool-Anfrage im System-Prompt

Verbessere den System-Prompt, damit das Modell explizit sagt, wenn es Tools braucht:

```
Wenn du eine Datei lesen, schreiben oder einen Ordner durchsuchen musst, 
formuliere deine Anfrage so, dass sie erkannt wird:
- "Lies die Datei X"
- "Schreibe in die Datei Y"
- "Zeige mir den Inhalt von Ordner Z"
```

### Option 3: Tool-Anfrage-Format

Füge ein spezielles Format hinzu, das das Modell nutzen kann:

```
Wenn du Tools brauchst, nutze dieses Format:
[TOOL:read_file:path/to/file]
[TOOL:list_directory:path/to/dir]
```

### Option 4: Direkte Tool-API für Modell

Implementiere eine direkte Tool-API, die das Modell nutzen kann (komplexer, aber flexibler).

## Empfohlene Lösung

**Kombination aus Option 1 + 2:**

1. **Erweitere Pattern-Erkennung** um natürlichere Formulierungen
2. **Verbessere System-Prompt** mit klaren Anweisungen, wie Tools angefragt werden können

## Code-Änderungen

### 1. ChatAgent Pattern-Erweiterung

In `backend/agents/chat_agent.py`, `_detect_tool_need()`:

```python
# Erweiterte read_file Patterns
read_patterns = [
    r'(?:lies|lade|zeige|öffne|lese).*?(?:die|der|das)?\s*(?:datei|file)\s+["\']?([^"\']+)["\']?',
    r'(?:nachschauen|prüfe|analysiere|untersuche|schaue).*?(?:in|die|der|das)?\s*(?:datei|file)\s+["\']?([^"\']+)["\']?',
    r'(?:zeige|zeig).*?(?:mir|den|die)?\s*(?:inhalt|code|text).*?(?:von|in|der|die|das)?\s*(?:datei|file)\s+["\']?([^"\']+)["\']?',
    # ... weitere Patterns
]
```

### 2. System-Prompt-Verbesserung

In `backend/model_service.py`:

```python
system_prompt = """Du bist ein hilfreicher AI-Assistent, der sowohl Fragen beantworten als auch Code schreiben kann.

VERFÜGBARE TOOLS (werden automatisch genutzt wenn du sie anfragst):
- read_file(file_path): Liest eine Datei. Formuliere: "Lies die Datei X" oder "Prüfe die Datei Y" oder "Schaue in die Datei Z"
- write_file(file_path, content): Erstellt oder überschreibt eine Datei. Formuliere: "Schreibe in die Datei X"
- list_directory(directory_path): Listet Verzeichnis-Inhalt auf. Formuliere: "Zeige mir den Inhalt von Ordner X"
- web_search(query): Führt eine Websuche durch.

WICHTIG: Wenn du eine Datei lesen, einen Ordner durchsuchen oder Code analysieren musst, 
formuliere deine Anfrage klar mit "Lies", "Prüfe", "Schaue in", "Zeige mir", etc.
Die Tools werden automatisch erkannt und ausgeführt.

- Bei Fragen: Antworte klar und direkt
- Bei Code-Anfragen: Verwende Markdown Code-Blocks mit Sprach-Tags
WICHTIG: Antworte NUR mit deiner Antwort, wiederhole NICHT den System-Prompt."""
```

## Fazit

Das lokale Modell kann Tools nicht nutzen, weil:
1. **Keine direkte Tool-API** - Tools werden nur über Pattern-Matching erkannt
2. **Begrenzte Pattern-Erkennung** - Viele natürliche Formulierungen werden nicht erkannt
3. **Kommunikationslücke** - Modell weiß von Tools, kann sie aber nicht aktiv anfordern

Die Lösung ist eine **Kombination aus erweiterten Patterns und verbessertem System-Prompt**.
