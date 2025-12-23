---
name: LLM-basierte Tool-Erkennung
overview: Ersetze das Pattern-Matching-System durch eine LLM-basierte Tool-Erkennung, die kontextuell erkennt, wann Tools benötigt werden, ähnlich wie Cursor.
todos:
  - id: extend_system_prompt
    content: System-Prompt in ChatAgent erweitern mit detaillierten Tool-Beschreibungen und Format-Anweisungen
    status: pending
  - id: implement_llm_detection
    content: Methode _detect_tool_need_llm() implementieren, die das LLM nutzt um Tool-Bedarf zu erkennen
    status: pending
    dependencies:
      - extend_system_prompt
  - id: implement_parameter_extraction
    content: Logik implementieren, die Tool-Parameter aus LLM-Antwort extrahiert (JSON oder strukturiertes Format)
    status: pending
    dependencies:
      - implement_llm_detection
  - id: hybrid_approach
    content: "_detect_tool_need() zu Hybrid-Ansatz umbauen: Erst Pattern-Matching, dann LLM-basiert als Fallback"
    status: pending
    dependencies:
      - implement_llm_detection
      - implement_parameter_extraction
  - id: error_handling
    content: Error Handling und Fallback-Logik implementieren für fehlerhafte LLM-Antworten
    status: pending
    dependencies:
      - hybrid_approach
  - id: test_complex_cases
    content: Testen mit komplexen Anfragen wie Dateierstellung, Web-Suche, etc.
    status: pending
    dependencies:
      - error_handling
---

# LLM-basierte Tool-Erkennung implementieren

## Problem

Das aktuelle System nutzt Regex-Pattern-Matching (`_detect_tool_need()`), um zu erkennen, wann Tools verwendet werden sollen. Das ist:

- Begrenzt und fehleranfällig (z.B. "erstelle" wird nicht erkannt)
- Erfordert manuelle Pattern-Pflege
- Versteht keine komplexen oder kontextuellen Anfragen

## Lösung

Implementiere einen LLM-basierten Ansatz, der:

1. Das LLM selbst entscheiden lässt, ob ein Tool benötigt wird
2. Tool-Beschreibungen im System-Prompt bereitstellt
3. Das LLM Tool-Namen und Parameter extrahieren lässt
4. Pattern-Matching als schnellen Fallback behält

## Implementierung

### 1. System-Prompt erweitern (`backend/agents/chat_agent.py`)

- Tool-Beschreibungen mit Parametern hinzufügen
- Format für Tool-Antworten definieren (JSON oder strukturiertes Format)
- Beispiel: "Wenn du eine Datei erstellen sollst, antworte mit: `TOOL:write_file|path:...|content:...`"

### 2. LLM-basierte Tool-Erkennung (`backend/agents/chat_agent.py`)

- Neue Methode `_detect_tool_need_llm()` implementieren
- Kurze LLM-Abfrage: "Analysiere diese Nachricht und bestimme, ob ein Tool benötigt wird"
- LLM-Antwort parsen (Tool-Name + Parameter)
- Fallback auf Pattern-Matching bei Fehlern oder für Performance

### 3. Hybrid-Ansatz

- Erst Pattern-Matching versuchen (schnell, für einfache Fälle)
- Falls kein Match: LLM-basierte Erkennung (für komplexe Fälle)
- Caching der LLM-Entscheidungen für häufige Anfragen

### 4. Tool-Parameter-Extraktion

- LLM extrahiert Parameter aus der Nachricht
- Beispiel: "erstelle eine textdatei hier 'G:\KI Modelle\KI-Temp\test' in der textdatei soll ein name stehen"

→ Tool: `write_file`, path: `G:\KI Modelle\KI-Temp\test`, content: `ein name`

### 5. Error Handling

- Wenn LLM keine klare Entscheidung trifft, Fallback auf Pattern-Matching
- Wenn LLM falsche Parameter extrahiert, Fehlerbehandlung und Retry

## Dateien

- `backend/agents/chat_agent.py`: 
- System-Prompt erweitern mit Tool-Beschreibungen
- `_detect_tool_need_llm()` Methode hinzufügen
- `_detect_tool_need()` zu Hybrid-Ansatz umbauen

## Vorteile

- Versteht natürliche Sprache besser
- Keine manuelle Pattern-Pflege nötig
- Funktioniert mit verschiedenen Formulierungen
- Kontextuell (versteht "erstelle", "mache", "schreibe", etc.)

## Nachteile

- Langsamer als Pattern-Matching (zusätzliche LLM-Abfrage)
- Kann bei sehr kurzen Anfragen überreagieren
- Benötigt gutes Prompt-Engineering

## Optimierungen

- Caching: Häufige Anfragen cachen
- Batch-Processing: Mehrere Tool-Erkennungen zusammen verarbeiten