import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

for steps in (20_000, 40_000, 80_000):
    model = SAC("MlpPolicy", "Pendulum-v1", seed=0)
    model.learn(total_timesteps=steps)
    env = gym.make("Pendulum-v1")
    env.reset(seed=123)  # same eval conditions each time
    rewards, _ = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True)
    print(f"{steps:>6} steps  mean {np.mean(rewards):7.1f}  std {np.std(rewards):6.1f}  worst {min(rewards):7.1f}")