from stable_baselines3 import SAC

model = SAC("MlpPolicy", "Pendulum-v1", verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=80000)
model.save("pendulum_sac")