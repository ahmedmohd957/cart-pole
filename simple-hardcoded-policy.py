# Hard code a simple strategy: 
# if the pole is tilting to the left, then push the cart to the left, and vice versa.

import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode='human')

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        if done or truncated:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), min(totals), max(totals))
# Output: 41.698 8.389445512070509 24.0 63.0
# the best it did was to keep the poll up for only 63 steps. 
# This environment is considered solved when the agent keeps the poll up for 200 steps.

env.close()
