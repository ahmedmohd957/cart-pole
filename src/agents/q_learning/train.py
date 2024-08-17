import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from env.cartpole import CustomCartPoleEnv
from src.agents.q_learning.qlearning import QLearningAgent
from src.utils.helper_functions import print_metrics
from src.utils.plots import plot_training_result

# Set up environment and configuration
env = CustomCartPoleEnv()
env.reset(seed=42)

config = {
    'env': env,
    'learning_rate': 0.421798,
    'gamma': 1.0,
    'epsilon': 0.1,
    'epsilon_decay': 0.4,
    'episodes': 2000,
    'max_steps': 1000,
    'bins': (40, 40, 40, 40)
}

# Train Q-learning agent
agent = QLearningAgent(config)
episode_rewards, average_rewards, solved_episode, training_time = agent.train()

# Print metrics
print_metrics(episode_rewards, solved_episode, training_time)

# Plot training result
plot_training_result("Q-learning", config["episodes"], episode_rewards, average_rewards)
