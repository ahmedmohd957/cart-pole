import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from QLearningAgent import QLearningAgent

# Set up environment and configuration
env = gym.make('CartPole-v1')
observation = env.reset(seed=74)

config = {
    'env': env,
    'learning_rate': 0.2,
    'gamma': 1.0,
    'epsilon': 0.2,
    'epsilon_decay': 0.670,
    'episodes': 2000,
    'max_steps': 1000,
    'bins': (38, 38, 38, 38),
    'seed': 1,
}

# Train Q-Learning agent
agent = QLearningAgent(config)
episode_rewards, average_reward, solved_episode, training_time = agent.train()

# Print metrics
print(f"Average Reward (last 100): {np.mean(episode_rewards[-100:])}")
print(f"Solved at episode: {solved_episode}")
print(f"Training Time (seconds): {training_time}")

# Plotting the original reward and average reward
data = {
    'Episode': range(config['episodes']),
    'Reward': episode_rewards,
    'Average Reward': average_reward
}

df = pd.DataFrame(data)

sns.set(style="darkgrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Episode', y='Reward', label='Reward')
sns.lineplot(data=df, x='Episode', y='Average Reward', label='Average Reward')
plt.axhline(y=475, color='r', linestyle='--', label='475 Steps')
plt.title('Q-learning Training Result', fontsize=16)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.legend(loc='upper left')
plt.savefig(f'Q-learning.png')
plt.show()