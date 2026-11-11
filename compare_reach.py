import numpy as np
from stable_baselines3 import SAC
from reach_env import ReachEnv

model = SAC.load("reach_sac")
straight = lambda obs: np.clip((obs[2:] - obs[:2]) * 10, -1, 1)
sac = lambda obs: model.predict(obs, deterministic=True)[0]

def run(policy, n=200):
    env, steps_list, wins = ReachEnv(), [], 0
    for i in range(n):
        obs, _ = env.reset(seed=5000 + i)   # same starts for both policies
        done, steps = False, 0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            steps += 1
            done = term or trunc
        wins += info["reached"]
        steps_list.append(steps)
    return wins / n, np.mean(steps_list), np.std(steps_list)

for name, p in (("straight", straight), ("SAC", sac)):
    w, m, s = run(p)
    print(f"{name:9s} success {w:.0%}  steps {m:.1f} +/- {s:.1f}")