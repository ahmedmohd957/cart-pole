import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from tqdm import tqdm
from src.utils.state_discretizer import StateDiscretizer

def evaluate_agent(env, episodes, max_steps, q_table_file):
    q_table = np.load(q_table_file)
    discretizer = StateDiscretizer(env)
    episode_rewards = []
    average_rewards = []
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        cumulative_reward = 0.0
        for _ in range(max_steps):
            discretized_state = discretizer.discretize_state(state)
            action = np.argmax(q_table[discretized_state])
            next_state, reward, terminated, _, _ = env.step(action)
            cumulative_reward += reward
            state = next_state
            if terminated:
                break
        episode_rewards.append(cumulative_reward)
        average_rewards.append(np.mean(episode_rewards))
    return episode_rewards, average_rewards
