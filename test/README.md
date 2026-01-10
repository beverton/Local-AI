# Pong - Klassisches Arcade-Spiel

Eine vollständige Implementierung des klassischen Pong-Spiels in Python mit pygame.

## Installation

1. Installieren Sie die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Spielen

Starten Sie das Spiel:
```bash
python pong.py
```

## Steuerung

- **Linkes Paddle:**
  - `W` - Nach oben bewegen
  - `S` - Nach unten bewegen

- **Rechtes Paddle:**
  - `Pfeiltaste hoch` - Nach oben bewegen
  - `Pfeiltaste runter` - Nach unten bewegen

- **Sonstiges:**
  - `ESC` - Spiel beenden
  - `R` - Neustart (nach Game Over)

## Spielregeln

- Der Ball bewegt sich automatisch über den Bildschirm
- Wenn der Ball an einem Paddle abprallt, ändert er die Richtung
- Wenn der Ball links oder rechts aus dem Bildschirm geht, erhält der Gegner einen Punkt
- Der erste Spieler, der 10 Punkte erreicht, gewinnt

## Technische Details

- **Auflösung:** 800x600 Pixel
- **FPS:** 60
- **Paddle-Größe:** 15x100 Pixel
- **Ball-Größe:** 15x15 Pixel
- **Gewinnpunktzahl:** 10 Punkte

## Features

- ✅ Zwei Spieler (lokaler Multiplayer)
- ✅ Punktestand-Anzeige
- ✅ Kollisionserkennung
- ✅ Game Over Screen
- ✅ Neustart-Funktion
- ✅ Flüssige Bewegung (60 FPS)
