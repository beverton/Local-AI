---
name: Qwen Integration & Quality Management Testing
overview: Systematische Integration und Optimierung von Qwen-2.5-7b-instruct mit vollständigem Quality Management Testing, Coding-Optimierungen und MCP Server Integration für Cursor.
todos:
  - id: qwen-eos-token-fix
    content: "Fix EOS-Token-Handling für Qwen: Liste beibehalten statt zu Integer konvertieren (model_manager.py Zeile 848-862)"
    status: completed
  - id: qwen-context-limit
    content: "Qwen zu model_limits hinzufügen: 32k Kontext statt 2048 (model_manager.py Zeile 790-799)"
    status: completed
  - id: qwen-basic-test
    content: "Test-Script erstellen: test_qwen_basic.py - einfache, komplexe und Coding-Fragen testen"
    status: completed
    dependencies:
      - qwen-eos-token-fix
      - qwen-context-limit
  - id: git-commit-phase1
    content: "Git Commit nach Phase 1: Qwen Basis-Fixes und Tests (Commit-Message: 'fix: Qwen EOS-Token und Kontext-Limit, Basis-Tests hinzugefügt')"
    status: completed
    dependencies:
      - qwen-basic-test
      - debug-logging
  - id: debug-logging
    content: "Debug-Logging erweitern: Nach generate(), decode(), clean(), return() (model_manager.py)"
    status: completed
  - id: quality-test-script
    content: "Test-Script erstellen: test_quality_options.py für systematisches Testing aller Quality-Optionen"
    status: completed
    dependencies:
      - qwen-basic-test
  - id: test-auto-web-search
    content: "Test auto_web_search: Web-Search vor Generierung, Quellen in Antwort"
    status: completed
    dependencies:
      - quality-test-script
  - id: test-web-validation
    content: "Test web_validation: Response-Validierung und Retry-Logik"
    status: completed
    dependencies:
      - quality-test-script
  - id: test-hallucination-check
    content: "Test hallucination_check: URL- und Zahlen-Validierung"
    status: completed
    dependencies:
      - quality-test-script
  - id: test-other-quality-options
    content: Test contradiction_check, actuality_check, source_quality_check, completeness_check
    status: completed
    dependencies:
      - quality-test-script
  - id: test-quality-combinations
    content: "Test Quality-Optionen in Kombination: auto_web_search + web_validation, alle aktiviert, etc."
    status: completed
    dependencies:
      - test-auto-web-search
      - test-web-validation
      - test-hallucination-check
  - id: git-commit-phase2
    content: "Git Commit nach Phase 2: Quality Management Einzeltests (Commit-Message: 'test: Quality Management Optionen einzeln getestet')"
    status: completed
    dependencies:
      - test-other-quality-options
  - id: git-commit-phase3
    content: "Git Commit nach Phase 3: Quality Management Kombinationen (Commit-Message: 'test: Quality Management Kombinationen und Performance getestet')"
    status: completed
    dependencies:
      - test-quality-combinations
  - id: qwen-coding-prompt
    content: "Coding-spezifischer System-Prompt für Qwen: Code-Formatierung, Best Practices"
    status: completed
    dependencies:
      - qwen-basic-test
  - id: qwen-coding-parameters
    content: "Coding-spezifische Parameter: temperature=0.2, höhere max_new_tokens, repetition_penalty=1.1"
    status: completed
    dependencies:
      - qwen-basic-test
  - id: code-detection
    content: "Code-Erkennung implementieren: Regex-Patterns für Coding-Fragen"
    status: completed
    dependencies:
      - qwen-coding-prompt
  - id: git-commit-phase4
    content: "Git Commit nach Phase 4: Coding-Optimierungen (Commit-Message: 'feat: Qwen Coding-Optimierungen (System-Prompt, Parameter, Code-Erkennung)')"
    status: completed
    dependencies:
      - code-detection
      - qwen-coding-parameters
  - id: mcp-server-check
    content: "MCP Server prüfen: Funktioniert er? Alle Tools verfügbar? Model Service Verbindung?"
    status: completed
  - id: mcp-model-tools
    content: "MCP Server erweitern: list_models, load_model, unload_model, model_status, chat Tools"
    status: completed
    dependencies:
      - mcp-server-check
  - id: mcp-server-test
    content: "MCP Server testen: Server startet, Tools registriert, Chat funktioniert"
    status: completed
    dependencies:
      - mcp-model-tools
  - id: cursor-mcp-config
    content: "Cursor MCP Konfiguration prüfen und aktualisieren: Pfade korrekt, Dokumentation aktualisieren"
    status: completed
    dependencies:
      - mcp-server-test
  - id: git-commit-phase5
    content: "Git Commit nach Phase 5: MCP Server Integration (Commit-Message: 'feat: MCP Server Model Service Integration (list_models, load_model, chat, etc.)')"
    status: completed
    dependencies:
      - cursor-mcp-config
  - id: documentation
    content: "Dokumentation erstellen: Test-Ergebnisse, Performance-Metriken, bekannte Probleme (docs/qwen_integration_results.md)"
    status: completed
    dependencies:
      - test-quality-combinations
      - mcp-server-test
  - id: code-cleanup
    content: "Code-Cleanup: Debug-Logs entfernen/setzen, Kommentare aktualisieren, unbenutzte Code-Pfade entfernen"
    status: completed
    dependencies:
      - documentation
  - id: final-tests
    content: "Finale Tests: Smoke-Tests, Integration-Tests, Performance-Tests"
    status: completed
    dependencies:
      - code-cleanup
  - id: git-commit-final
    content: "Finaler Git Commit: Dokumentation und Cleanup (Commit-Message: 'docs: Qwen Integration abgeschlossen - Dokumentation und Cleanup')"
    status: completed
    dependencies:
      - final-tests
      - documentation
---

# Qwen Integration & Quality Management - Vollständiger Plan

## Phase 1: Qwen Basis-Funktionalität prüfen und reparieren

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein Git Commit als Backup.

### 1.1 Qwen-Laden prüfen

- **Datei**: `backend/model_manager.py`
- **Problem**: EOS-Token-Handling für Qwen ist defekt (Zeile 849 konvertiert Liste zu Integer)
- **Fix**: EOS-Token-Liste für Qwen beibehalten (model.generate() unterstützt Listen)
- **Test**: Modell laden und Status prüfen
- **KI-Aktion**: Code-Fix implementieren, testen, validieren

### 1.2 Qwen-Kontext-Limit hinzufügen

- **Datei**: `backend/model_manager.py` (Zeile 790-799)
- **Problem**: Qwen fehlt in `model_limits` - verwendet Default 2048 statt 32k
- **Fix**: `"qwen": 32768` und `"qwen2": 32768` zu `model_limits` hinzufügen
- **Erkennung**: Prüfe auf "qwen" im gesamten Modell-Namen, nicht nur Prefix
- **KI-Action**: Code-Fix implementieren, testen, validieren

### 1.3 Qwen-Antworten testen

- **Test-Script**: Erstelle `test_qwen_basic.py`
- **Tests**:
  - Einfache Frage: "Was ist 2+3?"
  - Komplexe Frage: "Erkläre Python Decorators"
  - Coding-Frage: "Schreibe eine Fibonacci-Funktion"
- **Erwartung**: Antworten sollten vollständig sein, keine hängenden Generierungen
- **KI-Aktion**: Test-Script erstellen, ausführen, Ergebnisse validieren

### 1.4 Debug-Logging erweitern

- **Datei**: `backend/model_manager.py`
- **Hinzufügen**:
  - Log nach `model.generate()`: "Generierung abgeschlossen"
  - Log nach Decoding: "Decoding abgeschlossen, Länge: X"
  - Log nach Cleaning: "Cleaning abgeschlossen, finale Länge: Y"
  - Log vor Return: "Response fertig, Länge: Z"
- **KI-Aktion**: Logging hinzufügen, testen

### 1.5 Git Commit - Phase 1 abgeschlossen

- **Commit-Message**: "fix: Qwen EOS-Token und Kontext-Limit, Basis-Tests hinzugefügt"
- **Dateien**: `backend/model_manager.py`, `test_qwen_basic.py`
- **KI-Aktion**: Git commit durchführen nach erfolgreichem Test

## Phase 2: Quality Management - Einzelne Optionen testen

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein Git Commit als Backup.

### 2.1 Test-Script erstellen

- **Datei**: `test_quality_options.py`
- **Funktionalität**:
  - Jede Quality-Option einzeln aktivieren
  - Test-Fragen für jede Option
  - Logging der Ergebnisse
  - Vergleich mit/ohne Option
- **KI-Aktion**: Test-Script erstellen, strukturiert aufbauen für alle Optionen

### 2.2 Option 1: `auto_web_search`

- **Datei**: `data/quality_settings.json`
- **Test-Fragen**:
  - "Was ist die aktuelle Python-Version?"
  - "Wann wurde Qwen 2.5 veröffentlicht?"
- **Erwartung**: Web-Search wird durchgeführt, Quellen werden in Antwort integriert
- **Prüfung**: `sources` Array in Response vorhanden

### 2.3 Option 2: `web_validation`

- **Test-Fragen**:
  - "Was ist die Hauptstadt von Deutschland?" (einfach, sollte validieren)
  - "Wie funktioniert Quantum Computing?" (komplex, sollte Retry auslösen wenn unvollständig)
- **Erwartung**: Response wird validiert, Retry bei Problemen
- **Prüfung**: `validation` Objekt in Response, `retry_count` loggen

### 2.4 Option 3: `hallucination_check`

- **Test-Fragen**:
  - "Was ist die URL zu Qwen auf HuggingFace?" (sollte URLs validieren)
  - "Wie viele Parameter hat Qwen-2.5-7B?" (sollte Zahlen validieren)
- **Erwartung**: Halluzinationen werden erkannt
- **Prüfung**: `hallucination_issues` Array in Response

### 2.5 Option 4: `contradiction_check`

- **Test-Fragen**: Fragen mit widersprüchlichen Informationen
- **Erwartung**: Widersprüche werden erkannt
- **Prüfung**: `contradiction_issues` in Response

### 2.6 Option 5: `actuality_check`

- **Test-Fragen**: Fragen zu aktuellen Ereignissen
- **Erwartung**: Aktualität wird geprüft
- **Prüfung**: `actuality_issues` in Response

### 2.7 Option 6: `source_quality_check`

- **Test-Fragen**: Fragen die Web-Search auslösen
- **Erwartung**: Quellen-Qualität wird bewertet
- **Prüfung**: `source_quality_score` in Response

### 2.8 Option 7: `completeness_check`

- **Test-Fragen**: Komplexe Fragen die vollständige Antworten benötigen
- **Erwartung**: Vollständigkeit wird geprüft
- **Prüfung**: `completeness_score` in Response

### 2.9 Git Commit - Phase 2 abgeschlossen

- **Commit-Message**: "test: Quality Management Optionen einzeln getestet"
- **Dateien**: `test_quality_options.py`, `data/quality_settings.json` (falls geändert)
- **KI-Aktion**: Git commit nach erfolgreichem Test aller Optionen

## Phase 3: Quality Management - Kombinationen testen

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein Git Commit als Backup.

### 3.1 Häufige Kombinationen

- **Kombination 1**: `auto_web_search` + `web_validation`
  - RAG vor Generierung + Validierung nach Generierung
- **Kombination 2**: `auto_web_search` + `hallucination_check`
  - RAG + Halluzinations-Erkennung
- **Kombination 3**: Alle Optionen aktiviert
  - Vollständiges Quality Management
- **Test-Script**: Erweitere `test_quality_options.py` für Kombinationen

### 3.2 Performance-Tests

- **Metriken**: Antwort-Zeit, Token-Verbrauch, GPU-Speicher
- **Vergleich**: Mit/ohne Quality Management
- **Dokumentation**: Ergebnisse in `docs/quality_performance.md`
- **KI-Aktion**: Performance-Tests durchführen, Ergebnisse dokumentieren

### 3.3 Git Commit - Phase 3 abgeschlossen

- **Commit-Message**: "test: Quality Management Kombinationen und Performance getestet"
- **Dateien**: `test_quality_options.py` (erweitert), `docs/quality_performance.md`
- **KI-Aktion**: Git commit nach erfolgreichem Test aller Kombinationen

## Phase 4: Qwen für Coding optimieren

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein Git Commit als Backup.

### 4.1 Coding-spezifischer System-Prompt

- **Datei**: `backend/main.py` oder `backend/agents/chat_agent.py`
- **Erkennung**: Wenn Modell-ID "qwen" enthält UND Frage Coding-bezogen ist
- **System-Prompt**: 
  - Code-Formatierung (Markdown Code-Blocks)
  - Kommentare in Code
  - Best Practices erwähnen
  - Fehlerbehandlung

### 4.2 Coding-spezifische Parameter

- **Datei**: `backend/model_manager.py`
- **Parameter für Coding**:
  - `temperature`: 0.2 (niedriger für präziseren Code)
  - `repetition_penalty`: 1.1 (weniger Repetition)
  - `max_new_tokens`: Höher für längeren Code (4096 statt 2048)

### 4.3 Code-Erkennung

- **Datei**: `backend/quality_manager.py` oder neue Funktion
- **Erkennung**: Regex-Patterns für Coding-Fragen
  - "Schreibe", "Erstelle", "Implementiere"
  - "Code", "Funktion", "Klasse"
  - Datei-Endungen: ".py", ".js", ".ts", etc.

### 4.4 Code-Validierung (optional)

- **Tool**: Code-Syntax prüfen (Python AST, etc.)
- **Integration**: In Quality Management als zusätzliche Validierung
- **KI-Aktion**: Code-Erkennung implementieren, Coding-Parameter anwenden

### 4.5 Git Commit - Phase 4 abgeschlossen

- **Commit-Message**: "feat: Qwen Coding-Optimierungen (System-Prompt, Parameter, Code-Erkennung)"
- **Dateien**: `backend/main.py` oder `backend/agents/chat_agent.py`, `backend/model_manager.py`, `backend/quality_manager.py`
- **KI-Aktion**: Git commit nach erfolgreicher Implementierung

## Phase 5: MCP Server für Model Service

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein Git Commit als Backup.

### 5.1 MCP Server prüfen

- **Datei**: `backend/mcp_server.py`
- **Prüfung**: 
  - Funktioniert der Server?
  - Sind alle Tools verfügbar?
  - Verbindung zum Model Service (Port 8001)?

### 5.2 Model Service Integration

- **Datei**: `backend/mcp_server.py`
- **Hinzufügen**:
  - Tool: `list_models` - Liste aller verfügbaren Modelle
  - Tool: `load_model` - Modell laden
  - Tool: `unload_model` - Modell entladen
  - Tool: `model_status` - Status des aktuellen Modells
  - Tool: `chat` - Chat mit lokalem Modell (über Model Service)

### 5.3 MCP Server testen

- **Test-Script**: `test_mcp_server.py` (existiert bereits, erweitern)
- **Tests**:
  - Server startet korrekt
  - Tools werden registriert
  - Model Service Verbindung funktioniert
  - Chat über MCP funktioniert

### 5.4 Cursor-Konfiguration

- **Datei**: `config/cursor_mcp_config.json`
- **Prüfung**: Pfade sind korrekt
- **Dokumentation**: `docs/CURSOR_MCP_SETUP.md` aktualisieren
- **Test**: MCP Server in Cursor aktivieren und testen
- **KI-Aktion**: Konfiguration prüfen, Dokumentation aktualisieren

### 5.5 Git Commit - Phase 5 abgeschlossen

- **Commit-Message**: "feat: MCP Server Model Service Integration (list_models, load_model, chat, etc.)"
- **Dateien**: `backend/mcp_server.py`, `config/cursor_mcp_config.json`, `docs/CURSOR_MCP_SETUP.md`
- **KI-Aktion**: Git commit nach erfolgreicher Integration und Tests

## Phase 6: Dokumentation & Abschluss

**WICHTIG**: Diese Phase wird vollständig von der KI selbstständig implementiert. Nach Abschluss erfolgt ein finaler Git Commit.

### 6.1 Test-Ergebnisse dokumentieren

- **Datei**: `docs/qwen_integration_results.md`
- **Inhalt**:
  - Alle Test-Ergebnisse
  - Performance-Metriken
  - Bekannte Probleme
  - Empfehlungen

### 6.2 Code-Cleanup

- Debug-Logs entfernen oder auf DEBUG-Level setzen
- Kommentare aktualisieren
- Unbenutzte Code-Pfade entfernen

### 6.3 Finale Tests

- **Smoke-Tests**: Alle Haupt-Funktionen
- **Integration-Tests**: End-to-End mit allen Quality-Optionen
- **Performance-Tests**: Vergleich vor/nach Optimierungen
- **KI-Aktion**: Alle Tests durchführen, Ergebnisse dokumentieren

### 6.4 Finaler Git Commit

- **Commit-Message**: "docs: Qwen Integration abgeschlossen - Dokumentation und Cleanup"
- **Dateien**: Alle geänderten Dateien, `docs/qwen_integration_results.md`
- **KI-Aktion**: Finaler Git commit nach erfolgreichem Abschluss aller Phasen

## Abhängigkeiten zwischen Phasen

```
Phase 1 (Qwen Basis) → Phase 2 (Quality Einzeln)
Phase 2 (Quality Einzeln) → Phase 3 (Quality Kombinationen)
Phase 1 (Qwen Basis) → Phase 4 (Coding Optimierung)
Phase 1 (Qwen Basis) → Phase 5 (MCP Server)
Phase 5 (MCP Server) → Phase 6 (Dokumentation)
```

## Risiken & Mitigation

1. **Qwen hängt bei Generierung**

   - Mitigation: Timeout in `model.generate()` hinzufügen
   - Fallback: Reduziere `max_new_tokens` wenn Timeout

2. **Quality Management zu langsam**

   - Mitigation: Web-Search Timeout (bereits 5s)
   - Option: Quality Management asynchron machen

3. **MCP Server Verbindungsprobleme**

   - Mitigation: Retry-Logic für Model Service Verbindung
   - Fallback: Lokale Manager wenn Model Service nicht verfügbar

## Erfolgs-Kriterien

- [ ] Qwen lädt korrekt ohne "meta device" Fehler
- [ ] Qwen generiert vollständige Antworten ohne Hängen
- [ ] Alle 7 Quality-Optionen funktionieren einzeln
- [ ] Quality-Optionen funktionieren in Kombination
- [ ] Qwen ist für Coding optimiert (niedrigere Temperature, etc.)
- [ ] MCP Server stellt Model Service Tools bereit
- [ ] MCP Server funktioniert in Cursor
- [ ] Alle Tests bestehen
- [ ] Dokumentation ist vollständig
- [ ] Git Commits nach jeder Phase erfolgreich

## Implementierungs-Strategie

**Selbstständige KI-Implementierung**: Die KI führt alle Phasen selbstständig durch:

1. Code-Änderungen implementieren
2. Tests durchführen und validieren
3. Bei Fehlern: Debugging und Fixes
4. Nach erfolgreicher Phase: Git Commit als Backup
5. Nächste Phase starten

**Git Commit Strategie**:

- Commit nach Phase 1: Qwen Basis-Fixes
- Commit nach Phase 2: Quality Management Einzeltests
- Commit nach Phase 3: Quality Management Kombinationen
- Commit nach Phase 4: Coding-Optimierungen
- Commit nach Phase 5: MCP Server Integration
- Finaler Commit nach Phase 6: Dokumentation und Cleanup

**Bei Fehlern**:

- Git Status prüfen
- Fehlerhafte Änderungen rückgängig machen (git reset/revert)
- Fix implementieren
- Erneut testen
- Commit nach erfolgreichem Fix