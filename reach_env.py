import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # obs: pos(2), velocity(2), target(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.accel = 0.02         # acceleration per step at full action
        self.drag = 0.98          # slight friction; try 1.0 for a frictionless puck
        self.goal_radius = 0.1
        self.max_speed_to_finish = 0.05   # must be slow to "land" on the target
        self.max_steps = 200

    def _obs(self):
        return np.concatenate([self.pos, self.vel, self.target]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1, 1, size=2)
        self.vel = np.zeros(2)
        self.target = self.np_random.uniform(-1, 1, size=2)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.vel = (self.vel + self.accel * action) * self.drag
        self.pos = self.pos + self.vel
        # hitting a wall stops motion on that axis
        for i in range(2):
            if abs(self.pos[i]) > 1:
                self.pos[i] = np.clip(self.pos[i], -1, 1)
                self.vel[i] = 0.0
        self.steps += 1

        dist = float(np.linalg.norm(self.pos - self.target))
        speed = float(np.linalg.norm(self.vel))
        terminated = dist < self.goal_radius and speed < self.max_speed_to_finish
        truncated = self.steps >= self.max_steps
        reward = -dist + (10.0 if terminated else 0.0)
        return self._obs(), reward, terminated, truncated, {"reached": terminated}