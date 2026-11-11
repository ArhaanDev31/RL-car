import numpy as np
import pygame
from stable_baselines3 import SAC
from reach_env import ReachEnv

model = SAC.load("reach_sac_momentum")
env = ReachEnv()

SIZE = 640                     # arena in pixels
PANEL = 260
W, H = SIZE + PANEL, SIZE
BG = (20, 28, 55)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Reach env - SAC")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)
small = pygame.font.SysFont("consolas", 15)

def to_px(p):
    # env coords -1..1 -> pixels (y flipped so up is up)
    return int((p[0] + 1) / 2 * SIZE), int((1 - (p[1] + 1) / 2) * SIZE)

def draw(pos, target, trail, ep, steps, reward, wins, total, last_steps, mode, paused):
    screen.fill(BG)
    pygame.draw.rect(screen, (30, 40, 75), (0, 0, SIZE, SIZE))
    pygame.draw.rect(screen, (90, 100, 140), (0, 0, SIZE, SIZE), 2)

    # target and goal radius
    tx, ty = to_px(target)
    r_px = int(env.goal_radius / 2 * SIZE)
    pygame.draw.circle(screen, (90, 220, 130), (tx, ty), r_px, 2)
    pygame.draw.circle(screen, (90, 220, 130), (tx, ty), 5)

    # trail (older = dimmer)
    n = len(trail)
    for i in range(1, n):
        a = i / n
        col = tuple(int(BG[k] * (1 - a) + (80, 170, 255)[k] * a) for k in range(3))
        pygame.draw.line(screen, col, to_px(trail[i - 1]), to_px(trail[i]), 3)

    # dot
    px, py = to_px(pos)
    pygame.draw.circle(screen, (255, 210, 80), (px, py), 9)

    # panel
    x0 = SIZE + 20
    screen.blit(font.render(f"Episode {ep}", True, (230, 230, 240)), (x0, 20))
    screen.blit(small.render(f"step {steps}/{env.max_steps}", True, (180, 190, 220)), (x0, 55))
    screen.blit(small.render(f"reward {reward:.1f}", True, (180, 190, 220)), (x0, 78))
    rate = wins / total if total else 0
    screen.blit(font.render(f"success {wins}/{total}", True, (255, 210, 80)), (x0, 130))
    screen.blit(small.render(f"({rate:.0%})", True, (255, 210, 80)), (x0, 158))
    if last_steps is not None:
        screen.blit(small.render(f"last episode: {last_steps} steps", True, (180, 190, 220)), (x0, 190))
    screen.blit(small.render(f"policy: {mode}", True, (140, 220, 255)), (x0, 230))
    screen.blit(small.render("B  swap policy", True, (140, 150, 180)), (x0, H - 90))
    screen.blit(small.render("SPACE  pause", True, (140, 150, 180)), (x0, H - 68))
    screen.blit(small.render("UP/DOWN  speed", True, (140, 150, 180)), (x0, H - 46))
    screen.blit(small.render("ESC  quit", True, (140, 150, 180)), (x0, H - 24))
    if paused:
        screen.blit(font.render("PAUSED", True, (255, 120, 120)), (x0, 270))
    pygame.display.flip()

def act(obs, mode):
    if mode == "SAC":
        return model.predict(obs, deterministic=True)[0]
    return np.clip((obs[4:6] - obs[0:2]) * 10, -1, 1)

obs, _ = env.reset()
trail = [env.pos.copy()]
ep, steps, ep_reward, wins, total, last_steps = 1, 0, 0.0, 0, 0, None
mode, speed, paused, running = "SAC", 1.0, False, True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            elif e.key == pygame.K_SPACE: paused = not paused
            elif e.key == pygame.K_UP: speed = min(speed * 2, 16)
            elif e.key == pygame.K_DOWN: speed = max(speed / 2, 0.25)
            elif e.key == pygame.K_b:
                mode = "STRAIGHT" if mode == "SAC" else "SAC"
                wins, total, last_steps = 0, 0, None      # reset stats for a fair comparison

    if not paused:
        obs, r, term, trunc, info = env.step(act(obs, mode))
        steps += 1
        ep_reward += r
        trail.append(env.pos.copy())
        if term or trunc:
            total += 1
            wins += int(info["reached"])
            last_steps = steps
            print(f"episode {ep:3d}  {mode:8s}  steps {steps:3d}  reached {info['reached']}")
            ep += 1
            obs, _ = env.reset()
            trail, steps, ep_reward = [env.pos.copy()], 0, 0.0

    draw(env.pos, env.target, trail, ep, steps, ep_reward, wins, total, last_steps, mode, paused)
    clock.tick(30 * speed)

pygame.quit()