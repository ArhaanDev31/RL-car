import math
import numpy as np
import torch
import gymnasium as gym
import pygame
from stable_baselines3 import PPO

model = PPO.load("cartpole_ppo")
env = gym.make("CartPole-v1")

# Network layers (PPO MlpPolicy: Linear-Tanh-Linear-Tanh, then action head)
pn = model.policy.mlp_extractor.policy_net
head = model.policy.action_net
W1 = pn[0].weight.detach().numpy()   # (64, 4)
W2 = pn[2].weight.detach().numpy()   # (64, 64)
W3 = head.weight.detach().numpy()    # (2, 64)

def forward(obs):
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        h1 = pn[1](pn[0](x))
        h2 = pn[3](pn[2](h1))
        probs = torch.softmax(head(h2), dim=1)
    return x[0].numpy(), h1[0].numpy(), h2[0].numpy(), probs[0].numpy()

W_CART, W_NET, H = 800, 700, 500
GROUND_Y, POLE_LEN = 380, 190
SCALE = (W_CART - 200) / 4.8
SHOW = 16                      # neurons displayed per hidden layer
BG = (20, 28, 55)

pygame.init()
screen = pygame.display.set_mode((W_CART + W_NET, H))
pygame.display.set_caption("CartPole RL + Neural Network")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)
small = pygame.font.SysFont("consolas", 14)

def blend(color, a):
    return tuple(int(BG[i] * (1 - a) + color[i] * a) for i in range(3))

def draw_cart(state, steps, episode, best):
    x, _, theta, _ = state
    pygame.draw.rect(screen, (60, 70, 90), (0, GROUND_Y + 30, W_CART, H))
    for lim in (-2.4, 2.4):
        lx = int(W_CART / 2 + lim * SCALE)
        pygame.draw.line(screen, (220, 80, 80), (lx, GROUND_Y - 10), (lx, GROUND_Y + 30), 3)
    cx = int(W_CART / 2 + x * SCALE)
    cart = pygame.Rect(0, 0, 90, 45)
    cart.center = (cx, GROUND_Y)
    pygame.draw.rect(screen, (80, 170, 255), cart, border_radius=10)
    tip = (cx + POLE_LEN * math.sin(theta), GROUND_Y - POLE_LEN * math.cos(theta))
    danger = min(abs(theta) / 0.2095, 1.0)
    color = (int(80 + 175 * danger), int(230 - 150 * danger), 100)
    pygame.draw.line(screen, color, (cx, GROUND_Y), tip, 12)
    pygame.draw.circle(screen, color, (int(tip[0]), int(tip[1])), 9)
    screen.blit(font.render(f"Episode {episode}  Steps {steps}  Best {best}",
                            True, (230, 230, 240)), (20, 20))

def node_y(n, i):
    return int(90 + (H - 150) * (i + 0.5) / n)

def draw_edges(W, src_act, dst_act, x0, n_src, n_dst, y_src_n, y_dst_n, x1):
    for j in range(n_src):
        for i in range(n_dst):
            v = W[i, j] * src_act[j]
            a = min(abs(v) / 1.5, 1.0)
            if a < 0.08:
                continue
            col = (90, 170, 255) if v > 0 else (255, 150, 70)
            pygame.draw.line(screen, blend(col, a),
                             (x0, node_y(y_src_n, j)), (x1, node_y(y_dst_n, i)), 1)

def draw_network(inp, h1, h2, probs):
    X = [W_CART + 90, W_CART + 270, W_CART + 450, W_CART + 620]
    a1, a2 = h1[:SHOW], h2[:SHOW]
    iv = np.tanh(inp)                      # squash inputs into -1..1 for display

    draw_edges(W1[:SHOW, :], iv, a1, X[0], 4, SHOW, 4, SHOW, X[1])  # note: W1 rows=hidden
    # (W1 is [hidden, input]; transpose view handled below)
    draw_edges(W2[:SHOW, :SHOW], a1, a2, X[1], SHOW, SHOW, SHOW, SHOW, X[2])
    draw_edges(W3[:, :SHOW], a2, probs, X[2], SHOW, 2, SHOW, 2, X[3])

    def nodes(vals, x, n, labels=None):
        for i, v in enumerate(vals[:n]):
            t = (float(v) + 1) / 2         # -1..1 -> 0..1
            c = (int(40 + 200 * t), int(60 + 150 * t), int(120 + 100 * (1 - t) * 0 + 100 * t))
            pygame.draw.circle(screen, c, (x, node_y(n, i)), 8 if n <= 4 else 6)
            if labels:
                screen.blit(small.render(labels[i], True, (200, 200, 220)), (x - 80, node_y(n, i) - 8))

    nodes(iv, X[0], 4, ["pos", "vel", "angle", "ang_vel"])
    nodes(a1, X[1], SHOW)
    nodes(a2, X[2], SHOW)
    for i, (p, name) in enumerate(zip(probs, ["LEFT", "RIGHT"])):
        y = node_y(2, i)
        pygame.draw.rect(screen, (50, 60, 90), (X[3] - 10, y - 12, 80, 24), border_radius=4)
        pygame.draw.rect(screen, (255, 210, 80) if p == probs.max() else (120, 130, 160),
                         (X[3] - 10, y - 12, int(80 * p), 24), border_radius=4)
        screen.blit(small.render(f"{name} {p:.0%}", True, (240, 240, 250)), (X[3] - 8, y - 8))
    for x, t in zip(X, ["Input", "Hidden 1", "Hidden 2", "Output"]):
        screen.blit(small.render(t, True, (180, 190, 220)), (x - 25, 55))

obs, _ = env.reset()
steps, best, episode = 0, 0, 1
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
    inp, h1, h2, probs = forward(obs)
    action = int(np.argmax(probs))          # same as deterministic predict
    obs, _, terminated, truncated, _ = env.step(action)
    steps += 1

    screen.fill(BG)
    draw_cart(obs, steps, episode, best)
    draw_network(inp, h1, h2, probs)
    pygame.display.flip()

    if terminated or truncated:
        best = max(best, steps)
        steps, episode = 0, episode + 1
        obs, _ = env.reset()
    clock.tick(50)
pygame.quit()