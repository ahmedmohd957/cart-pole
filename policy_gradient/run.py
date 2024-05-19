import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from reinforce import REINFORCE

# Set up environment and configuration
env = gym.make('CartPole-v1')

config = {
    'env': env,
    'episodes': 800,
    'gamma': 0.99,
    'learning_rate': 0.01,
    'seed': 1,
}

# Train REINFORCE agent
reinforce = REINFORCE(config)
episode_rewards, average_reward, convergence_episode, training_time = reinforce.train()

# Print metrics
print(f'Average Cumulative Reward (last 100): {np.mean(episode_rewards[-100:])}')
print(f'Convergence Episode: {convergence_episode}')
print(f'Training Time (seconds): {training_time}')

plt.figure().set_figwidth(15)
plt.plot(np.arange(config['episodes']), episode_rewards, label='Reward')
plt.plot(np.arange(config['episodes']), average_reward, label='Average Reward')
plt.axhline(y=475, color='r', linestyle='--', label='475 Steps')
plt.legend(loc='upper left')
plt.title(f'REINFORCE Training Result')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'Reinforce.png')
plt.show()

env.close()
