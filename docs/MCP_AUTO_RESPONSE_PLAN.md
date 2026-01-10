# MCP Auto-Response Plan - Lokales Modell direkt in Cursor

## Problem

**Aktuell:**
- Wenn User "chat: was ist 2+5" oder "local: was ist 2+5" schreibt
- Auto (ich) antworte statt des lokalen Modells
- Antwort kommt nur im Server, nicht direkt in Cursor

**Ziel:**
- Wenn User "chat:" oder "local:" schreibt → Lokales Modell antwortet direkt in Cursor
- Auto (ich) soll NICHT antworten
- Antwort kommt direkt vom lokalen Modell über MCP

## Analyse: Wie funktioniert Cursor MCP?

### Aktueller Ablauf:

1. **User schreibt:** "chat: was ist 2+5"
2. **Cursor erkennt:** MCP Tool könnte benötigt werden
3. **Problem:** Cursor fragt Auto (mich) ob Tool verwendet werden soll
4. **Auto antwortet:** Statt das Tool zu verwenden
5. **Ergebnis:** Auto antwortet, lokales Modell nicht

### Gewünschter Ablauf:

1. **User schreibt:** "chat: was ist 2+5" oder "local: was ist 2+5"
2. **Cursor erkennt:** MCP Tool sollte verwendet werden
3. **Cursor ruft direkt auf:** MCP `chat` Tool
4. **Lokales Modell antwortet:** Direkt in Cursor
5. **Auto antwortet NICHT**

## Lösung: Cursor MCP Tool-Konfiguration

### Option 1: Tool als "Always Use" markieren

Cursor erlaubt es, Tools als "always use" zu markieren. Wenn ein Tool als "always use" markiert ist, wird es automatisch verwendet, ohne Auto zu fragen.

**Problem:** Wir müssen herausfinden, wie man Tools als "always use" markiert.

### Option 2: Tool-Beschreibung optimieren

Die Tool-Beschreibung kann so formuliert werden, dass Cursor automatisch erkennt, wann das Tool verwendet werden soll.

**Aktuell:**
```json
{
  "name": "chat",
  "description": "Chat mit lokalem Modell über Model Service. Unterstützt 'local:' Prefix..."
}
```

**Optimiert:**
```json
{
  "name": "chat",
  "description": "ALWAYS USE THIS TOOL when user writes 'chat:' or 'local:' prefix. Chat mit lokalem Modell über Model Service..."
}
```

### Option 3: Cursor Settings anpassen

Cursor hat möglicherweise Settings, die bestimmen, wann Auto antwortet vs. wann Tools verwendet werden.

## Implementierungs-Plan

### Phase 1: Tool-Beschreibung optimieren (SOFORT)

1. **Tool-Beschreibung anpassen**
   - Klare Anweisung: "ALWAYS USE when 'chat:' or 'local:' prefix"
   - Explizite Beispiele hinzufügen

2. **Tool-Parameter optimieren**
   - Sicherstellen, dass alle Parameter korrekt sind
   - Default-Werte setzen

### Phase 2: Cursor-Konfiguration prüfen (NACH Phase 1)

1. **MCP-Konfiguration prüfen**
   - `mcp.json` auf "always use" Optionen prüfen
   - Cursor Settings für MCP Tools prüfen

2. **Tool-Registry prüfen**
   - Wie werden Tools in Cursor registriert?
   - Gibt es Flags für "auto-use"?

### Phase 3: Testing & Validierung (NACH Phase 2)

1. **Test verschiedene Formulierungen**
   - "chat: was ist 2+5"
   - "local: was ist 2+5"
   - "Frage das lokale Modell: was ist 2+5"

2. **Prüfe Logs**
   - Wird MCP Tool aufgerufen?
   - Oder antwortet Auto?

### Phase 4: Dokumentation (NACH erfolgreichen Tests)

1. **Dokumentiere korrekte Verwendung**
   - Wie schreibt man, damit lokales Modell antwortet?
   - Was vermeiden?

## Aktuelle Status

✅ **Lokales Modell funktioniert:**
- Antworten sind korrekt
- Code-Blocks bleiben intakt
- Cleaning-System robust

⏳ **MCP Integration:**
- Tool ist registriert
- Funktioniert wenn manuell aufgerufen
- Auto antwortet noch statt lokales Modell

## Nächste Schritte

1. **Tool-Beschreibung optimieren** (sofort)
2. **Cursor MCP-Konfiguration prüfen** (nach Tool-Optimierung)
3. **Tests durchführen** (nach Konfiguration)
4. **Dokumentation aktualisieren** (nach erfolgreichen Tests)

## Wichtige Hinweise

⚠️ **Nur implementieren wenn Responses gut sind:**
- Aktuell sind Responses gut (✅ getestet)
- Aber noch nicht perfekt (Chat-Marker-Problem teilweise gelöst)
- Warten bis Responses 100% korrekt sind

⚠️ **Auto soll NICHT antworten wenn:**
- User schreibt "chat:" oder "local:"
- User explizit lokales Modell anfragt
- MCP Tool verfügbar ist

✅ **Auto SOLL antworten wenn:**
- User fragt allgemeine Fragen (ohne "chat:" oder "local:")
- MCP Tool nicht verfügbar ist
- User explizit Auto anfragt
