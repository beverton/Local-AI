"""
Pong - Klassisches Arcade-Spiel
Steuerung:
- Links: W (hoch) / S (runter)
- Rechts: Pfeiltaste hoch / Pfeiltaste runter
"""

import pygame
import sys
import random

# Initialisierung
pygame.init()

# Konstanten
WIDTH, HEIGHT = 800, 600
BALL_SIZE = 15
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 100
PADDLE_SPEED = 5
BALL_SPEED = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WIN_SCORE = 10

# Fenster erstellen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)


class Paddle:
    """Paddle-Klasse für die Spieler"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.speed = PADDLE_SPEED
        self.score = 0
    
    def move_up(self):
        """Bewegt Paddle nach oben"""
        if self.y > 0:
            self.y -= self.speed
    
    def move_down(self):
        """Bewegt Paddle nach unten"""
        if self.y < HEIGHT - self.height:
            self.y += self.speed
    
    def draw(self):
        """Zeichnet das Paddle"""
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
    
    def get_rect(self):
        """Gibt das Rechteck des Paddles zurück"""
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Ball:
    """Ball-Klasse"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Setzt den Ball in die Mitte zurück"""
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.size = BALL_SIZE
        # Zufällige Richtung
        self.dx = random.choice([-1, 1]) * BALL_SPEED
        self.dy = random.choice([-1, 1]) * BALL_SPEED
    
    def move(self, left_paddle, right_paddle):
        """Bewegt den Ball und prüft Kollisionen"""
        # Bewegung
        self.x += self.dx
        self.y += self.dy
        
        # Kollision mit oberem/unterem Rand
        if self.y <= 0 or self.y >= HEIGHT - self.size:
            self.dy = -self.dy
        
        # Kollision mit linkem Paddle
        ball_rect = pygame.Rect(self.x, self.y, self.size, self.size)
        left_rect = left_paddle.get_rect()
        if ball_rect.colliderect(left_rect):
            if self.dx < 0:  # Ball bewegt sich nach links
                self.dx = -self.dx
                # Leichte Richtungsänderung basierend auf Trefferposition
                hit_pos = (self.y - left_paddle.y) / left_paddle.height
                self.dy = (hit_pos - 0.5) * 2 * BALL_SPEED
        
        # Kollision mit rechtem Paddle
        right_rect = right_paddle.get_rect()
        if ball_rect.colliderect(right_rect):
            if self.dx > 0:  # Ball bewegt sich nach rechts
                self.dx = -self.dx
                # Leichte Richtungsänderung basierend auf Trefferposition
                hit_pos = (self.y - right_paddle.y) / right_paddle.height
                self.dy = (hit_pos - 0.5) * 2 * BALL_SPEED
        
        # Punkt für rechten Spieler (Ball geht links raus)
        if self.x < 0:
            right_paddle.score += 1
            self.reset()
            return "right_score"
        
        # Punkt für linken Spieler (Ball geht rechts raus)
        if self.x > WIDTH:
            left_paddle.score += 1
            self.reset()
            return "left_score"
        
        return None
    
    def draw(self):
        """Zeichnet den Ball"""
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.size, self.size))


def draw_scores(left_score, right_score):
    """Zeichnet die Punktestände"""
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (WIDTH // 4, 50))
    screen.blit(right_text, (3 * WIDTH // 4, 50))


def draw_center_line():
    """Zeichnet die Mittellinie"""
    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 2, y, 4, 10))


def show_game_over(winner):
    """Zeigt Game Over Nachricht"""
    screen.fill(BLACK)
    if winner == "left":
        text = font.render("Spieler Links gewinnt!", True, WHITE)
    else:
        text = font.render("Spieler Rechts gewinnt!", True, WHITE)
    
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(text, text_rect)
    
    restart_text = small_font.render("Drücken Sie R zum Neustart oder ESC zum Beenden", True, WHITE)
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    screen.blit(restart_text, restart_rect)
    
    pygame.display.flip()


def main():
    """Hauptspiel-Loop"""
    # Spielobjekte erstellen
    left_paddle = Paddle(50, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    right_paddle = Paddle(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()
    
    game_over = False
    winner = None
    
    # Haupt-Loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                if game_over and event.key == pygame.K_r:
                    # Spiel zurücksetzen
                    left_paddle.score = 0
                    right_paddle.score = 0
                    ball.reset()
                    game_over = False
                    winner = None
        
        if not game_over:
            # Steuerung
            keys = pygame.key.get_pressed()
            
            # Linkes Paddle: W/S
            if keys[pygame.K_w]:
                left_paddle.move_up()
            if keys[pygame.K_s]:
                left_paddle.move_down()
            
            # Rechtes Paddle: Pfeiltasten
            if keys[pygame.K_UP]:
                right_paddle.move_up()
            if keys[pygame.K_DOWN]:
                right_paddle.move_down()
            
            # Ball bewegen
            result = ball.move(left_paddle, right_paddle)
            
            # Prüfe auf Gewinner
            if left_paddle.score >= WIN_SCORE:
                game_over = True
                winner = "left"
            elif right_paddle.score >= WIN_SCORE:
                game_over = True
                winner = "right"
            
            # Zeichnen
            screen.fill(BLACK)
            draw_center_line()
            left_paddle.draw()
            right_paddle.draw()
            ball.draw()
            draw_scores(left_paddle.score, right_paddle.score)
        else:
            show_game_over(winner)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
