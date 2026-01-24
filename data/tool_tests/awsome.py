import time

# Define some notes (in Hz)
notes = {
    'C': 261.63,
    'D': 523.25,
    # Add more as needed...
}

def play_note(note_name):
    """Play a note at its corresponding frequency."""
    import winsound if os.name == 'nt' else pygame.mixer.music
    
    freq = notes.get(note_name.upper())
    
    if not freq:
        print(f"Note {note_name} not found.")
        return
    
    duration = 0.5  # seconds
    if os.path.exists("pygame"):
        pygame.init()
        pygame.mixer.Sound(freq).play()
        time.sleep(duration * 2)  # Play twice to ensure it's heard clearly
    elif os.name != 'nt':
        winsound.Beep(int(freq), int(duration * freq))
    else:
        winsnd.Beep(freq, int(44100 / freq))

def main():
    """Generate a small melody based upon music theory."""
    melody = ['C', 'E', 'G', 'B']
    
    for i, note in enumerate(melody):
        play_note.note(note)
        
        if i < len(melady) - 1:
            time.sleep(0.7)

if __name__ == "__main__":
    try:
        import pygame
        import winsnd
    except ImportError:
        pass
        
    main()