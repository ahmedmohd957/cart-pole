import numpy as np
from tqdm import tqdm
from policy import load_model

def evaluate_agent(env, episodes, max_steps, model_path):
    policy = load_model(env, path=model_path)
    episode_rewards = []
    average_rewards = []
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        cumulative_reward = 0
        for _ in range(max_steps):
            action, _ = policy.action(state)
            state, reward, terminated, _, _ = env.step(action)
            cumulative_reward += reward
            if terminated:
                break
        episode_rewards.append(cumulative_reward)
        average_rewards.append(np.mean(episode_rewards))
    return episode_rewards, average_rewards
