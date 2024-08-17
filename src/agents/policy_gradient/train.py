import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from env.cartpole import CustomCartPoleEnv
from reinforce import ReinforceAgent
from src.utils.helper_functions import print_metrics
from src.utils.plots import plot_training_result

# Set up environment and configuration
env = CustomCartPoleEnv()
env.reset(seed=540)

config = {
    'env': env,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'hidden_size': 128,
    'episodes': 2000,
    'max_steps': 1000,
}

# Train Reinforce agent
agent = ReinforceAgent(config)
episode_rewards, average_rewards, solved_episode, training_time = agent.train()

# Print metrics
print_metrics(episode_rewards, solved_episode, training_time)

# Plot training result
plot_training_result("REINFORCE", config["episodes"], episode_rewards, average_rewards)
