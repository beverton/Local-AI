# Response Cleaning - Analyse und Plan

## Aktueller Ansatz - Analyse

### Probleme identifiziert:

1. **Zu aggressive Bereinigung**
   - Zeichen-basierte Filterung entfernt legitime Inhalte
   - CJK-Bereinigung kann auch normale Zeichen entfernen
   - Code-Blocks werden beschädigt

2. **Fehlende Hierarchie**
   - Keine klare Priorisierung der Cleaning-Schritte
   - Fallback-Mechanismen greifen zu spät
   - Keine Validierung nach jedem Schritt

3. **Inkonsistente Marker-Erkennung**
   - Chat-Marker werden nicht zuverlässig erkannt
   - Case-sensitivity Probleme
   - Keine Unterscheidung zwischen legitimen und problematischen Markern

4. **Fehlende Modell-spezifische Behandlung**
   - Qwen hat spezielle Token-Strukturen
   - Verschiedene Chat-Templates benötigen unterschiedliche Behandlung

## Best Practices aus der Industrie

### ChatGPT/OpenAI Ansatz:
- Minimales Post-Processing
- Fokus auf Token-Level (nicht Zeichen-Level)
- Modell-spezifische Stop-Sequenzen
- Validierung vor Cleaning

### Perplexity Ansatz:
- Strukturierte Response-Validierung
- Code-Block-Erhaltung hat höchste Priorität
- Fallback auf Original bei Problemen

### Transformers Library Best Practices:
- Verwendung von `skip_special_tokens` Parameter
- Chat-Template-spezifische Bereinigung
- EOS-Token-basierte Trimming

## Robuster Cleaning-Plan

### Phase 1: Token-Level Cleaning (KRITISCH - IMMER)
1. **EOS-Token Handling**
   - Entferne alles nach dem ersten EOS-Token
   - Modell-spezifische EOS-Token-Listen
   - Validierung: Response sollte nach EOS enden

2. **Special Token Removal**
   - Entferne Chat-Template-Tokens (`<|im_start|>`, `<|im_end|>`, etc.)
   - Modell-spezifische Token-Listen
   - Validierung: Keine Special Tokens in finaler Response

### Phase 2: Struktur-Level Cleaning (WICHTIG)
3. **Chat-Marker Detection**
   - Erkenne Chat-Marker (Human:, Assistant:, User:, System:)
   - Trimme bei Markern die auf neuen Chat-Turn hinweisen
   - Validierung: Response sollte keine Chat-Marker enthalten

4. **Code-Block Protection**
   - Identifiziere Code-Blocks (```...```)
   - Schütze Code-Blocks während aller Cleaning-Schritte
   - Validierung: Code-Blocks sollten intakt bleiben

### Phase 3: Content-Level Cleaning (OPTIONAL - NUR WENN NÖTIG)
5. **Artifact Removal**
   - Entferne HTML-Tags (nur wenn nicht in Code-Blocks)
   - Entferne CJK-Zeichen (nur wenn Response noch Buchstaben hat)
   - Validierung: Response sollte noch Buchstaben enthalten

6. **Whitespace Normalization**
   - Normalisiere mehrfache Leerzeichen
   - Normalisiere mehrfache Zeilenumbrüche
   - Validierung: Response sollte lesbar bleiben

### Phase 4: Validation & Fallback
7. **Response Validation**
   - Prüfe ob Response Buchstaben enthält
   - Prüfe ob Response zu kurz ist
   - Prüfe ob Response nur Sonderzeichen enthält

8. **Fallback Mechanism**
   - Wenn Validierung fehlschlägt: Verwende Original
   - Minimal Cleaning auf Original (nur HTML, CJK)
   - Logging für Debugging

## Implementierungs-Strategie

### Prinzipien:
1. **Fail-Safe**: Immer Fallback auf Original
2. **Validierung nach jedem Schritt**: Prüfe ob Response noch gültig ist
3. **Modell-spezifisch**: Unterschiedliche Behandlung für Qwen, Mistral, etc.
4. **Code-Block-Priorität**: Code-Blocks haben höchste Priorität
5. **Minimal Invasiv**: Nur notwendige Bereinigung

### Struktur:
```python
def clean_response(response, model_type, messages):
    # Phase 1: Token-Level (KRITISCH)
    response = clean_tokens(response, model_type)
    if not validate_response(response):
        return fallback_clean(response)
    
    # Phase 2: Struktur-Level (WICHTIG)
    response = clean_structure(response, messages)
    if not validate_response(response):
        return fallback_clean(response)
    
    # Phase 3: Content-Level (OPTIONAL)
    if needs_content_cleaning(response):
        response = clean_content(response)
        if not validate_response(response):
            return fallback_clean(response)
    
    # Phase 4: Final Validation
    if not validate_response(response):
        return fallback_clean(response)
    
    return response
```

## Nächste Schritte

1. ✅ Analyse abgeschlossen
2. ⏳ Implementierung des neuen Cleaning-Systems
3. ⏳ Tests mit verschiedenen Modellen
4. ⏳ Performance-Optimierung
5. ⏳ Dokumentation
