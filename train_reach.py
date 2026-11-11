import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from reach_env import ReachEnv

check_env(ReachEnv())   # catches shape/dtype mistakes in your env

def success_rate(policy_fn, n=100):
    env, wins = ReachEnv(), 0
    for i in range(n):
        obs, _ = env.reset(seed=1000 + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            done = term or trunc
        wins += info["reached"]
    return wins / n

random_policy = lambda obs: np.random.uniform(-1, 1, 2)
print("random policy success:", success_rate(random_policy))

model = SAC("MlpPolicy", ReachEnv(), seed=0, verbose=1)
model.learn(total_timesteps=100_000)
model.save("reach_sac_momentum")

trained = lambda obs: model.predict(obs, deterministic=True)[0]
print("trained policy success:", success_rate(trained))