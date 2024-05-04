from reinforce import Reinforce
import numpy as np
from plotter import plot_graph_1, plot_graph_2, plot_graph_3, plot_test_rewards_with_average
import matplotlib.pyplot as plt

config = {
    'learning_rate': 0.005, #'learning_rate': 0.005 best that the grid search showed
    'discount_factor': 0.95, #'discount_factor': 0.95 best that the grid search showed
    'seed': 543,
    'n_episodes': 10000,
    'n_max_steps': 500,
    'log_interval': 10,
}


reinforce = Reinforce(config)
reinforce.train()

mean_total_rewards = np.cumsum(reinforce.total_rewards) / (np.arange(config["n_episodes"]) + 1)


# Test the trained model
#test_rewards = reinforce.test(n_test_episodes=100)
test_rewards = reinforce.test(n_test_episodes=5, render=True)

# Calculate average test reward
average_test_reward = np.mean(test_rewards)

print(f"Average Test Reward: {average_test_reward}")

plot_graph_1(
    title='Cartpole Policy Gradient Training Result', 
    xlabel="Episode",
    ylabel="Sum of rewards", 
    rewards=reinforce.total_rewards,
    mean_rewards=mean_total_rewards,
    episodes=config["n_episodes"], 
    file_name="test_1"
)

plot_graph_2(
    title='Policy Loss', 
    xlabel='Episode', 
    ylabel='Policy Loss', 
    rewards=reinforce.policy_losses, 
    file_name='policy_loss'
)

# Plot the test results
plot_graph_3(
    title='Test Performance of Trained Model', 
    xlabel='Test Episode', 
    ylabel='Reward per Episode', 
    rewards=test_rewards, 
    file_name='test_performance'
)

# Plot the test results with average test reward
plot_test_rewards_with_average(
    title='Test Performance of Trained Model', 
    xlabel='Test Episode', 
    ylabel='Reward per Episode', 
    rewards=test_rewards, 
    average_reward=average_test_reward,
    file_name='test_performance_average'
)
