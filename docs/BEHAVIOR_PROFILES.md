# Verhaltensprofile für Local AI

## Übersicht

Verhaltensprofile ermöglichen es, verschiedene Parameter-Konfigurationen für unterschiedliche Use Cases zu verwenden. Das Standard-Verhalten bleibt erhalten, während spezialisierte Profile (z.B. für Coding) optimierte Einstellungen bieten.

## Verfügbare Profile

### 1. `default` (Standard)
- **Beschreibung**: Standard-Verhalten für allgemeine Anfragen (Chat, Fragen, etc.)
- **Parameter**:
  - `temperature`: 0.3
  - `max_length`: 2048
  - `repetition_penalty`: 1.15
  - `top_p`: 0.9
  - `is_coding`: false

### 2. `coding` (Coding)
- **Beschreibung**: Optimiert für Code-Generierung mit Qwen - niedrige Temperature für präzisen Code
- **Parameter**:
  - `temperature`: 0.1 (niedriger = deterministischer, präziser Code)
  - `max_length`: 4096 (höher für längeren Code)
  - `repetition_penalty`: 1.1 (niedriger, erlaubt Wiederholungen in Code)
  - `top_p`: 0.95 (konservativ)
  - `top_k`: 40 (konservativ)
  - `is_coding`: true
- **System-Prompt**: Spezialisierter Coding-Prompt für Qwen

### 3. `creative` (Kreativ)
- **Beschreibung**: Höhere Temperature für kreativere Antworten
- **Parameter**:
  - `temperature`: 0.7
  - `max_length`: 2048
  - `repetition_penalty`: 1.2
  - `top_p`: 0.9
  - `is_coding`: false

## Verwendung

### Über API

**Chat-Request mit Profil:**
```json
{
  "message": "Erstelle eine Pong-Spiel in Python",
  "profile": "coding"
}
```

**Ohne Profil (verwendet default):**
```json
{
  "message": "Was ist 2+2?",
  "temperature": 0.3,
  "max_length": 2048
}
```

### Profil-Parameter überschreiben

Sie können Profil-Parameter explizit überschreiben:

```json
{
  "message": "Erstelle eine Pong-Spiel in Python",
  "profile": "coding",
  "temperature": 0.05,  // Überschreibt Profil-Parameter
  "max_length": 8192   // Überschreibt Profil-Parameter
}
```

## Profil-Konfiguration

Profile werden in `data/profiles.json` gespeichert:

```json
{
  "profiles": {
    "coding": {
      "name": "Coding",
      "description": "Optimiert für Code-Generierung",
      "parameters": {
        "temperature": 0.1,
        "max_length": 4096,
        "repetition_penalty": 1.1,
        "top_p": 0.95,
        "top_k": 40,
        "is_coding": true
      },
      "system_prompt_modifications": {
        "qwen": "Spezialisierter Coding-Prompt..."
      }
    }
  },
  "default_profile": "default",
  "model_specific_profiles": {
    "qwen-2.5-7b-instruct": {
      "default": "default",
      "coding": "coding"
    }
  }
}
```

## API-Endpoints

### GET `/profiles`
Gibt alle verfügbaren Profile zurück.

### GET `/profiles/{profile_name}`
Gibt ein spezifisches Profil zurück.

## Best Practices für Coding mit Qwen

Basierend auf Recherche und Tests:

1. **Temperature**: 0.1-0.2 für präzisen Code
   - Niedriger = deterministischer, präziser Code
   - Höher = kreativer, aber weniger konsistent

2. **max_length**: 4096+ für längeren Code
   - Ermöglicht vollständige Code-Generierung ohne Abschneiden

3. **repetition_penalty**: 1.1 für Code
   - Niedriger erlaubt Wiederholungen (z.B. in Schleifen, Variablennamen)

4. **top_p**: 0.95 (konservativ)
   - Fokussiert auf wahrscheinlichste Tokens

5. **top_k**: 40 (konservativ)
   - Begrenzt Auswahl auf Top-K Tokens

## Standard-Verhalten beibehalten

Das `default` Profil entspricht dem bisherigen Verhalten:
- `temperature`: 0.3
- `max_length`: 2048
- `repetition_penalty`: 1.15

Wenn kein Profil angegeben wird, wird automatisch `default` verwendet.

## Nächste Schritte

1. ✅ Profile-System implementiert
2. ✅ Coding-Profil mit optimierten Parametern erstellt
3. ✅ Standard-Verhalten als `default` Profil beibehalten
4. ⏳ Testen: `local: erstelle eine Version von dem Spiel "pong"` mit `profile: "coding"`
