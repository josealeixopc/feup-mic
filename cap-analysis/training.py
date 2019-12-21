import gym
from envs import cpa

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2, ACKTR, DQN, A2C, DDPG

# TODO: Logging and graphics

# env = cpa.CPAEnvDense()
env = cpa.CPAEnvSparse()

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = Monitor(env, filename=None, allow_early_resets=True)
env = DummyVecEnv([lambda: env])


# Train the agent
model = ACKTR('MlpPolicy', env, verbose=1)
# model = PPO2('MlpPolicy', env, verbose=1)
# model = A2C('MlpPolicy', env, verbose=1)
# model = DQN('MlpPolicy', env, verbose=1)
# model = DDPG('MlpPolicy', env, verbose=1) # Only spaces.Box is supported

model.learn(total_timesteps=10000)

# model.save("models/")

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
