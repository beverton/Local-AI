# Robuster Response Cleaning Plan

## Analyse des aktuellen Problems

### Identifizierte Probleme:

1. **Zu aggressive Zeichen-Filterung**
   - Regex-basierte Filterung entfernt legitime Inhalte
   - Code-Blocks werden beschädigt
   - Deutsche Umlaute und Sonderzeichen werden entfernt

2. **Fehlende Validierung**
   - Keine Prüfung nach jedem Cleaning-Schritt
   - Fallback greift zu spät
   - Keine Qualitätskontrolle

3. **Inkonsistente Marker-Erkennung**
   - Chat-Marker werden nicht zuverlässig erkannt
   - Keine Unterscheidung zwischen Content und Markern

4. **Fehlende Modell-spezifische Behandlung**
   - Qwen hat spezielle Token-Strukturen
   - Verschiedene Chat-Templates benötigen unterschiedliche Behandlung

## Robuster Cleaning-Plan (4-Phasen-Ansatz)

### Phase 1: Token-Level Cleaning (KRITISCH - IMMER)
**Ziel:** Entferne Modell-spezifische Tokens und Stop-Sequenzen

1. **EOS-Token Handling**
   - Entferne alles nach dem ersten EOS-Token (bereits in Decoding)
   - Modell-spezifische EOS-Token-Listen (Qwen: eos_token_id + im_end_id)
   - ✅ Validierung: Response sollte nach EOS enden

2. **Special Token Removal**
   - Entferne Chat-Template-Tokens: `<|im_start|>`, `<|im_end|>`
   - Entferne andere Special Tokens: `</s>`, `<|end|>`, `<|endoftext|>`
   - ✅ Validierung: Keine Special Tokens in finaler Response

3. **Assistant Prefix Removal**
   - Entferne führendes "assistant:" oder "assistant "
   - Case-insensitive
   - ✅ Validierung: Response sollte nicht mit "assistant" beginnen

### Phase 2: Struktur-Level Cleaning (WICHTIG)
**Ziel:** Entferne Chat-Marker und schütze Code-Blocks

4. **Code-Block Protection**
   - Identifiziere Code-Blocks: ```...```
   - Temporär ersetzen: `__CODE_BLOCK_0__`, `__CODE_BLOCK_1__`, etc.
   - ✅ Validierung: Code-Blocks sollten intakt bleiben

5. **Chat-Marker Detection & Trimming**
   - Erkenne Chat-Marker: `Human:`, `Assistant:`, `User:`, `System:`
   - Case-insensitive Suche
   - Trimme bei Markern die auf neuen Chat-Turn hinweisen:
     - Marker muss > 50 Zeichen vom Anfang entfernt sein
     - Nach Marker < 50 Zeichen ODER keine Buchstaben in ersten 100 Zeichen
   - ✅ Validierung: Response sollte keine Chat-Marker enthalten

6. **Code-Blocks wieder einfügen**
   - Ersetze `__CODE_BLOCK_X__` zurück mit Original
   - ✅ Validierung: Code-Blocks sollten intakt sein

### Phase 3: Content-Level Cleaning (OPTIONAL - NUR WENN NÖTIG)
**Ziel:** Entferne Artefakte ohne Content zu beschädigen

7. **HTML-Tag Removal**
   - Entferne HTML-Tags: `<div>`, `<a href>`, etc.
   - ✅ Validierung: Response sollte noch Buchstaben enthalten

8. **CJK-Zeichen Removal (NUR wenn Response Buchstaben hat)**
   - Entferne Chinesisch, Japanisch, Koreanisch
   - NUR wenn Response bereits Buchstaben hat
   - ✅ Validierung: Response sollte noch Buchstaben enthalten

9. **Whitespace Normalization**
   - Normalisiere mehrfache Leerzeichen: `  ` → ` `
   - Normalisiere mehrfache Zeilenumbrüche: `\n\n\n` → `\n\n`
   - ✅ Validierung: Response sollte lesbar bleiben

### Phase 4: Validation & Fallback (SICHERHEITSNETZ)
**Ziel:** Garantiere dass Response gültig ist

10. **Response Validation**
    - Prüfe ob Response Buchstaben enthält
    - Prüfe ob Response zu kurz ist (< 5 Zeichen)
    - Prüfe ob Response nur Sonderzeichen enthält

11. **Fallback Mechanism**
    - Wenn Validierung fehlschlägt: Verwende Original
    - Minimal Cleaning auf Original:
      - Nur HTML entfernen
      - Nur CJK entfernen (wenn Buchstaben vorhanden)
      - Chat-Marker trimmen
    - ✅ Validierung: Fallback-Response sollte gültig sein

## Implementierungs-Struktur

```python
def clean_response(response: str, model_type: str, messages: List[Dict]) -> str:
    """
    Robuste Response-Bereinigung mit 4-Phasen-Ansatz
    
    Prinzipien:
    1. Fail-Safe: Immer Fallback auf Original
    2. Validierung nach jedem Schritt
    3. Modell-spezifisch
    4. Code-Block-Priorität
    5. Minimal Invasiv
    """
    original_response = response
    response_before_step = response
    
    # ============================================================
    # PHASE 1: Token-Level Cleaning (KRITISCH)
    # ============================================================
    response = clean_tokens(response, model_type)
    if not validate_basic(response):
        logger.warning("[Clean] Token-Cleaning fehlgeschlagen, verwende Fallback")
        return fallback_clean(original_response)
    
    # ============================================================
    # PHASE 2: Struktur-Level Cleaning (WICHTIG)
    # ============================================================
    response = clean_structure(response, messages)
    if not validate_basic(response):
        logger.warning("[Clean] Struktur-Cleaning fehlgeschlagen, verwende Fallback")
        return fallback_clean(original_response)
    
    # ============================================================
    # PHASE 3: Content-Level Cleaning (OPTIONAL)
    # ============================================================
    if needs_content_cleaning(response):
        response = clean_content(response)
        if not validate_basic(response):
            logger.warning("[Clean] Content-Cleaning fehlgeschlagen, verwende Fallback")
            return fallback_clean(original_response)
    
    # ============================================================
    # PHASE 4: Final Validation
    # ============================================================
    if not validate_final(response):
        logger.warning("[Clean] Finale Validierung fehlgeschlagen, verwende Fallback")
        return fallback_clean(original_response)
    
    return response
```

## Validierungs-Funktionen

```python
def validate_basic(response: str) -> bool:
    """Basis-Validierung: Prüft ob Response gültig ist"""
    if not response or len(response.strip()) == 0:
        return False
    if len(response.strip()) < 5:
        return False
    # Prüfe ob Response Buchstaben enthält
    if not re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
        return False
    return True

def validate_final(response: str) -> bool:
    """Finale Validierung: Prüft ob Response vollständig ist"""
    if not validate_basic(response):
        return False
    # Prüfe ob Response nicht nur Sonderzeichen enthält
    words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', response)
    if len(words) == 0:
        return False
    return True
```

## Modell-spezifische Behandlung

### Qwen-Modelle:
- EOS-Tokens: `[eos_token_id, im_end_id]`
- Special Tokens: `<|im_start|>`, `<|im_end|>`
- Chat-Template: Qwen-ChatML-Format

### Mistral-Modelle:
- EOS-Tokens: `[eos_token_id]`
- Special Tokens: `</s>`
- Höhere Repetition Penalty nötig

### Phi-3-Modelle:
- EOS-Tokens: `[eos_token_id]`
- Chat-Template: ChatML-Format
- Kein separater System-Prompt nötig

## Qualitäts-Garantien

1. **Code-Blocks bleiben intakt**
   - Code-Blocks werden vor allen Cleaning-Schritten geschützt
   - Werden am Ende wieder eingefügt

2. **Fallback auf Original**
   - Wenn Validierung fehlschlägt, wird Original verwendet
   - Minimal Cleaning auf Original (nur HTML, CJK, Marker)

3. **Validierung nach jedem Schritt**
   - Jeder Cleaning-Schritt wird validiert
   - Bei Fehlschlag: Fallback

4. **Modell-spezifische Behandlung**
   - Unterschiedliche Behandlung für verschiedene Modelle
   - Anpassung an Modell-spezifische Token-Strukturen

5. **Minimal Invasiv**
   - Nur notwendige Bereinigung
   - Keine aggressive Zeichen-Filterung

## Nächste Schritte

1. ✅ Analyse abgeschlossen
2. ✅ Plan erstellt
3. ⏳ Implementierung des neuen Cleaning-Systems
4. ⏳ Tests mit verschiedenen Modellen
5. ⏳ Performance-Optimierung
6. ⏳ Dokumentation
