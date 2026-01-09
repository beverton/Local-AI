# Logging-System & Modell-Laden Optimierungen

## Zusammenfassung

### 1. Strukturiertes Logging-System ✅

**Implementiert:**
- Zentrales Logging-System (`backend/logging_utils.py`)
- Strukturierte Logs mit Tags: `[MODEL_LOAD]`, `[MODEL_GEN]`, `[CHAT]`, `[QUALITY]`, `[WEB_SEARCH]`, `[VALIDATION]`, `[ERROR]`
- Log-Dateien in `logs/` für jeden Bereich
- Konsistentes Format: `TIMESTAMP | LEVEL | [TAG] MESSAGE`

**Vorteile:**
- Schnelle Fehlerlokalisierung durch Tags
- Vollständige Tracebacks bei Exceptions
- Strukturierte Ausgabe für einfaches Filtern
- Datei-Logging für spätere Analyse

### 2. Modell-Laden Optimierungen ✅

**Änderungen:**
- **Nur ein Modell beim Start:** Model Service lädt nur Qwen Text-Modell beim Start
- **Keine gleichzeitigen Ladevorgänge:** Verhindert Konflikte beim Laden mehrerer Modelle
- **Automatisches Entladen:** Beim Laden eines neuen Text-Modells wird das alte automatisch entladen
- **Memory-Informationen im Status:** `/status` Endpoint zeigt jetzt GPU-Memory-Usage

**Model Service Startup:**
```python
# Lädt nur Qwen Text-Modell beim Start
# Audio- und Image-Modelle werden NICHT beim Start geladen (nur bei Bedarf)
```

**Memory-Informationen:**
```json
{
  "text_model": {
    "loaded": true,
    "model_id": "qwen-2.5-7b-instruct",
    "memory": {
      "gpu_allocated_gb": 13.5,
      "gpu_reserved_gb": 14.2,
      "gpu_total_gb": 15.9,
      "gpu_free_gb": 1.7,
      "gpu_usage_percent": 89.3
    }
  }
}
```

### 3. CUDA OOM Problem

**Problem:**
- `caching_allocator_warmup` versucht Speicher zu reservieren, bevor `device_map="auto"` greift
- Führt zu OOM auch mit `max_memory` und CPU-Offloading

**Lösung implementiert:**
- `device_map="auto"` mit `max_memory=8GB` und CPU-Offloading erlaubt
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` gesetzt (wird aber auf Windows nicht unterstützt)

**Bekannte Einschränkungen:**
- Qwen-2.5-7B benötigt ~13-14GB VRAM
- Mit 15.9GB total VRAM ist das knapp
- Empfehlung: Quantisierung (8-bit/4-bit) oder kleineres Modell verwenden

### 4. Nächste Schritte (für später)

**Service Manager UI für Memory-Management:**
- Anzeige: Wie viel Platz geladene Modelle beanspruchen
- Steuerung: Manuelles CPU-Offloading aktivieren/deaktivieren
- Visualisierung: GPU/CPU Memory-Usage Charts
- Warnungen: Bei hoher Memory-Usage

**Implementierung:**
- Frontend: Memory-Display in Service Manager
- Backend: Endpoint für Memory-Management (`/models/text/memory/offload_to_cpu`)
- Real-time Updates: WebSocket oder Polling für Memory-Status
