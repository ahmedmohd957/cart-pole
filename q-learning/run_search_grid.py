import gymnasium as gym
import numpy as np
from q_learning import Q_Learning
from itertools import product

def run_grid_search():
    env = gym.make('CartPole-v1')

    # Define hyperparameters to test
    learning_rates = [0.01, 0.1, 0.5]
    discount_factors = [0.9, 0.99, 1]
    epsilons = [0.1, 0.2, 0.3]
    epsilon_decays = [0.99, 0.999, 0.9999]
    episodes = 1000  # Reduced for quicker testing

    # Grid of all possible hyperparameter combinations
    grid = list(product(learning_rates, discount_factors, epsilons, epsilon_decays))

    best_reward = -float('inf')
    best_params = None

    # Test each combination
    for params in grid:
        lr, df, eps, ed = params
        config = {
            'env': env,
            'learning_rate': lr,
            'discount_factor': df,
            'epsilon': eps,
            'epsilon_decay': ed,
            'bins': [30, 30, 30, 30],
            'num_action_space': env.action_space.n
        }

        q_learning = Q_Learning(config)
        q_learning.train(episodes)

        # Assuming that higher rewards are better
        average_reward = np.mean(q_learning.total_rewards)
        print(f"Tested {params} with average reward: {average_reward}")

        # Update the best parameters
        if average_reward > best_reward:
            best_reward = average_reward
            best_params = params

    print(f"Best parameters: {best_params} with reward: {best_reward}")

if __name__ == "__main__":
    run_grid_search()
