# Response Cleaning - Implementierungs-Plan

## Übersicht

Dieses Dokument beschreibt die Implementierung des robusten 4-Phasen-Cleaning-Systems für LLM-Responses.

## Architektur

### Hauptfunktion: `clean_response()`

```python
def clean_response(
    response: str, 
    model_type: str, 
    messages: List[Dict[str, str]],
    original_prompt: str = ""
) -> str:
    """
    Robuste Response-Bereinigung mit 4-Phasen-Ansatz
    
    Args:
        response: Rohe Response vom Modell
        model_type: Modell-Typ (z.B. "qwen", "mistral", "phi-3")
        messages: Message-History
        original_prompt: Originaler Prompt (optional)
    
    Returns:
        Bereinigte Response
    """
```

### Phase 1: Token-Level Cleaning

```python
def clean_tokens(response: str, model_type: str) -> str:
    """
    Phase 1: Entferne Modell-spezifische Tokens
    
    Schritte:
    1. Entferne Chat-Template-Tokens
    2. Entferne Special Tokens
    3. Entferne Assistant-Prefix
    """
    # 1. Chat-Template-Tokens
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
    
    # 2. Special Tokens
    special_tokens = ["</s>", "<|end|>", "<|endoftext|>"]
    for token in special_tokens:
        response = response.replace(token, "")
    
    # 3. Assistant Prefix
    response = response.strip()
    if response.lower().startswith('assistant:'):
        response = response[10:].strip()
    elif response.lower().startswith('assistant '):
        response = response[9:].strip()
    
    return response
```

### Phase 2: Struktur-Level Cleaning

```python
def clean_structure(response: str, messages: List[Dict[str, str]]) -> str:
    """
    Phase 2: Entferne Chat-Marker und schütze Code-Blocks
    
    Schritte:
    1. Schütze Code-Blocks
    2. Entferne Chat-Marker
    3. Code-Blocks wieder einfügen
    """
    import re
    
    # 1. Code-Blocks schützen
    code_block_pattern = r'```[\s\S]*?```'
    code_blocks = re.findall(code_block_pattern, response)
    for i, code_block in enumerate(code_blocks):
        response = response.replace(code_block, f"__CODE_BLOCK_{i}__", 1)
    
    # 2. Chat-Marker entfernen
    markers = ['Human:', 'Assistant:', 'User:', 'System:']
    for marker in markers:
        positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
        for pos in positions:
            if pos > 50:  # Mindestens 50 Zeichen vom Anfang
                after_marker = response[pos + len(marker):].strip()
                # Wenn nach Marker wenig Content ODER keine Buchstaben
                if len(after_marker) < 50 or not re.search(r'[a-zA-ZäöüßÄÖÜ]', after_marker[:100]):
                    response = response[:pos].strip()
                    break
        if len(response) < len(response_before_marker) * 0.9:
            break
    
    # 3. Code-Blocks wieder einfügen
    for i, code_block in enumerate(code_blocks):
        response = response.replace(f"__CODE_BLOCK_{i}__", code_block)
    
    return response
```

### Phase 3: Content-Level Cleaning

```python
def clean_content(response: str) -> str:
    """
    Phase 3: Entferne Artefakte (OPTIONAL)
    
    Schritte:
    1. HTML-Tags entfernen
    2. CJK-Zeichen entfernen (nur wenn Buchstaben vorhanden)
    3. Whitespace normalisieren
    """
    import re
    
    # 1. HTML-Tags
    html_pattern = r'<[^>]+>'
    response = re.sub(html_pattern, '', response)
    
    # 2. CJK-Zeichen (nur wenn Buchstaben vorhanden)
    if re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]+'
        response = re.sub(cjk_pattern, '', response)
    
    # 3. Whitespace normalisieren
    response = re.sub(r' +', ' ', response)  # Mehrfache Leerzeichen
    response = re.sub(r'\n\s*\n+', '\n\n', response)  # Mehrfache Zeilenumbrüche
    
    return response.strip()
```

### Phase 4: Validation & Fallback

```python
def validate_basic(response: str) -> bool:
    """Basis-Validierung"""
    if not response or len(response.strip()) == 0:
        return False
    if len(response.strip()) < 5:
        return False
    if not re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
        return False
    return True

def validate_final(response: str) -> bool:
    """Finale Validierung"""
    if not validate_basic(response):
        return False
    # Prüfe ob Response Wörter enthält (nicht nur Sonderzeichen)
    words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', response)
    if len(words) == 0:
        return False
    return True

def fallback_clean(original_response: str) -> str:
    """
    Fallback: Minimal Cleaning auf Original
    
    Nur:
    - HTML entfernen
    - CJK entfernen (wenn Buchstaben vorhanden)
    - Chat-Marker trimmen
    - Whitespace normalisieren
    """
    import re
    
    response = original_response
    
    # HTML entfernen
    html_pattern = r'<[^>]+>'
    response = re.sub(html_pattern, '', response)
    
    # CJK entfernen (nur wenn Buchstaben vorhanden)
    if re.search(r'[a-zA-ZäöüßÄÖÜ]', response):
        cjk_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]+'
        response = re.sub(cjk_pattern, '', response)
    
    # Chat-Marker trimmen
    markers = ['Human:', 'Assistant:', 'User:', 'System:']
    for marker in markers:
        positions = [m.start() for m in re.finditer(re.escape(marker), response, re.IGNORECASE)]
        for pos in positions:
            if pos > 50:
                after_marker = response[pos + len(marker):].strip()
                if len(after_marker) < 50:
                    response = response[:pos].strip()
                    break
    
    # Whitespace normalisieren
    response = re.sub(r'\n\s*\n+', '\n\n', response)
    
    return response.strip()
```

## Implementierungs-Reihenfolge

1. **Erstelle neue Funktionen** (Phase 1-4)
2. **Ersetze alte `_clean_response_minimal()`** mit neuem System
3. **Tests mit verschiedenen Modellen**
4. **Performance-Optimierung**
5. **Dokumentation aktualisieren**

## Qualitäts-Garantien

✅ Code-Blocks bleiben intakt
✅ Fallback auf Original bei Problemen
✅ Validierung nach jedem Schritt
✅ Modell-spezifische Behandlung
✅ Minimal invasiv
