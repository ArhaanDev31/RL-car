import math
import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3 import SAC

MODEL_PATH = "pendulum_sac"        # change this to compare seeds/models
model = SAC.load(MODEL_PATH)
env = gym.make("Pendulum-v1")

W, H = 1000, 560
CX, CY, L = 280, 300, 170          # pivot position, rod length (px)
PANEL_X = 580
BG = (20, 28, 55)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Pendulum SAC - episode viewer")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)
small = pygame.font.SysFont("consolas", 15)

def rew_color(r):
    if r > -200: return (90, 220, 130)     # good
    if r > -400: return (255, 190, 80)     # mediocre
    return (240, 90, 90)                   # bad

def draw(obs, action, ep_rew, step, episode, start_deg, history, speed, paused):
    screen.fill(BG)
    theta = math.atan2(obs[1], obs[0])     # 0 = upright, +-pi = hanging down

    # pendulum
    tip = (CX + L * math.sin(theta), CY - L * math.cos(theta))
    pygame.draw.circle(screen, (45, 55, 90), (CX, CY), L, 1)          # reachable circle
    pygame.draw.line(screen, (90, 100, 140), (CX - 8, CY - L), (CX + 8, CY - L), 2)  # goal mark
    up = abs(theta) < 0.2
    col = (90, 220, 130) if up else (80, 170, 255)
    pygame.draw.line(screen, col, (CX, CY), tip, 10)
    pygame.draw.circle(screen, col, (int(tip[0]), int(tip[1])), 16)
    pygame.draw.circle(screen, (240, 240, 250), (CX, CY), 7)

    # torque bar (action is -2..2)
    a = float(action[0])
    pygame.draw.rect(screen, (50, 60, 90), (CX - 100, CY + L + 40, 200, 16), border_radius=6)
    bar_w = int(100 * a / 2)
    pygame.draw.rect(screen, (255, 210, 80),
                     (CX + min(0, bar_w), CY + L + 40, abs(bar_w), 16), border_radius=6)
    screen.blit(small.render(f"torque {a:+.2f}", True, (200, 200, 220)), (CX - 40, CY + L + 62))

    # HUD
    screen.blit(font.render(f"Episode {episode}   Step {step}/200", True, (230, 230, 240)), (20, 20))
    screen.blit(font.render(f"Reward so far {ep_rew:.0f}", True, rew_color(ep_rew * 200 / max(step, 1))), (20, 48))
    screen.blit(small.render(f"start angle {start_deg:+.0f} deg (180 = hanging down)", True, (180, 190, 220)), (20, 78))
    screen.blit(small.render(f"speed x{speed:.2f}   SPACE pause   UP/DOWN speed   ESC quit", True, (140, 150, 180)), (20, H - 28))
    if paused:
        screen.blit(font.render("PAUSED", True, (255, 120, 120)), (CX - 35, 100))

    # history panel
    screen.blit(font.render("Episode rewards (last 15)", True, (230, 230, 240)), (PANEL_X, 20))
    for i, (r, sd) in enumerate(history[-15:]):
        y = 60 + i * 28
        w = int(min(-r, 1000) / 1000 * 220)
        pygame.draw.rect(screen, rew_color(r), (PANEL_X, y, max(w, 3), 20), border_radius=4)
        screen.blit(small.render(f"{r:7.0f}   start {sd:+4.0f} deg", True, (230, 230, 240)), (PANEL_X + 235, y + 2))
    if history:
        rs = [r for r, _ in history]
        screen.blit(small.render(f"mean {np.mean(rs):.0f}  std {np.std(rs):.0f}  worst {min(rs):.0f}",
                                 True, (255, 210, 80)), (PANEL_X, H - 60))
    pygame.display.flip()

def start_angle(obs):
    return math.degrees(math.atan2(obs[1], obs[0]))

obs, _ = env.reset()
sd = start_angle(obs)
ep_rew, step, episode, speed, paused = 0.0, 0, 1, 1.0, False
history = []
action = np.array([0.0])
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            elif e.key == pygame.K_SPACE: paused = not paused
            elif e.key == pygame.K_UP: speed = min(speed * 2, 8)
            elif e.key == pygame.K_DOWN: speed = max(speed / 2, 0.25)

    if not paused:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_rew += reward
        step += 1
        if terminated or truncated:
            history.append((ep_rew, sd))
            print(f"episode {episode:3d}  reward {ep_rew:8.1f}  start {sd:+7.1f} deg")
            episode += 1
            obs, _ = env.reset()
            sd, ep_rew, step = start_angle(obs), 0.0, 0

    draw(obs, action, ep_rew, step, episode, sd, history, speed, paused)
    clock.tick(20 * speed)       # Pendulum's real rate is 20 steps/sec

pygame.quit()