import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning import Q_Learning

env = gym.make('CartPole-v1')

config = {
    'env': env,
    'episodes': 350,
    'epsilon': float,
    'min_epsilon': 0.005,
    'gamma': 0.95,
    'learning_rate': 0.5,
    'min_learning_rate': 0.01,
    'seed': 1,
    'bins': (1, 1, 6, 3),
}

q = Q_Learning(config)
episode_rewards, average_reward, convergence_episode, training_time = q.train()

# Print metrics
print(
    f"Average Cumulative Reward (last 100): {np.mean(episode_rewards[-100:])}")
print(f"Convergence Episode: {convergence_episode}")
print(f"Training Time (seconds): {training_time}")

plt.figure().set_figwidth(15)
plt.plot(range(config['episodes']), episode_rewards, label="Reward")
plt.plot(range(config['episodes']), average_reward, label="Average Reward")
plt.axhline(y=475, color='r', linestyle='--', label='475 Steps')
plt.legend(loc='upper left')
plt.title(f"Q-learning Training Result")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'Q-learning.png')
plt.show()
