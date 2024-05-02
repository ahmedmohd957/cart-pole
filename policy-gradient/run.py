from reinforce import Reinforce
import numpy as np
from plotter import plot_graph_1, plot_graph_2, plot_graph_3
import matplotlib.pyplot as plt

config = {
    'learning_rate': 0.01,
    'discount_factor': 0.99,
    'seed': 543,
    'n_episodes': 1000,
    'n_max_steps': 500,
    'log_interval': 10,
}

reinforce = Reinforce(config)
reinforce.train()

# Calculate the mean total reward after each episode
mean_total_rewards = np.cumsum(reinforce.total_rewards) / (np.arange(config["n_episodes"]) + 1)

plot_graph_1(
    title='Cartpole Policy Gradient Training Result', 
    xlabel="Episode",
    ylabel="Sum of rewards", 
    rewards=reinforce.total_rewards,
    mean_rewards=mean_total_rewards,
    episodes=config["n_episodes"], 
    file_name="test_1"
)

# plt.figure(figsize=(10, 6))
# plt.plot(reinforce.running_rewards)
# plt.title('Running Average of Rewards')
# plt.xlabel('Episode')
# plt.ylabel('Running Average Reward')
# plt.grid(True)
# plt.show()
