import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from env.cartpole import CustomCartPoleEnv
from evaluate_agent import evaluate_agent
from src.utils.plots import plot_robustness_performance
from src.utils.helper_functions import print_robustness_metrics

variants = ['Deterministic', 'Random', 'Extreme']
episode_rewards_data = []
average_rewards_data = []

for variant in variants:
    env = CustomCartPoleEnv(variant=variant[0])
    episode_rewards_over_seeds = []
    average_rewards_over_seeds = []
    for seed in [1, 2, 3, 5, 8]:
        env.reset(seed=seed)
        episode_rewards, average_rewards = evaluate_agent(env=env, episodes=200, max_steps=1000, q_table_file="data/q_table.npy")
        episode_rewards_over_seeds.append(episode_rewards)
        average_rewards_over_seeds.append(average_rewards)
    episode_rewards_data.append(episode_rewards_over_seeds)
    average_rewards_data.append(average_rewards_over_seeds)

# Print metrics
print_robustness_metrics(variants, episode_rewards_data)

# Plot results
plot_robustness_performance("Q-learning", variants, average_rewards_data)
