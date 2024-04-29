import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning import Q_Learning
from plotter import plot_graph_1, plot_graph_2

env = gym.make('CartPole-v1')
(state, _) = env.reset()

config = {
    'env': env,
    'learning_rate': 0.1,
    'discount_factor': 1,
    'epsilon': 0.2,
    "epsilon_decay": 0.999,
    'bins': [30, 30, 30, 30],  # position, velocity, angle, angle velocity
    'num_action_space': env.action_space.n
}

Q1 = Q_Learning(config)
episodes = 100
Q1.train(episodes)

# Calculate the mean total reward after each episode
mean_total_rewards = np.cumsum(Q1.total_rewards) / (np.arange(episodes) + 1)

print(mean_total_rewards)

plot_graph_1(
    title='Cartpole Q Learning Training Result', 
    xlabel="Episode",
    ylabel="Reward", 
    rewards=Q1.total_rewards,
    mean_rewards=mean_total_rewards,
    episodes=episodes, 
    file_name="test_1"
)

plot_graph_2(
    title="Cartpole Q Learning Training Result",
    xlabel="Episode",
    ylabel="Reward", 
    rewards=Q1.total_rewards, 
    file_name="test_2"
)
