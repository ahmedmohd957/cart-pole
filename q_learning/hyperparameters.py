import gymnasium as gym
import numpy as np
from q_learning import Q_Learning
from itertools import product

def main():
    env = gym.make('CartPole-v1')

    learning_rates = [0.5, 0.7, 1.0]
    discount_factors = [0.8, 0.95, 0.99]

    params_list = list(product(learning_rates, discount_factors))

    best_reward = -float('inf')
    best_params = None

    for params in params_list:
        lr, df = params
        config = {
            'env': env,
            'episodes': 350,
            'epsilon': float,
            'min_epsilon': 0.005,
            'gamma': df,
            'learning_rate': lr,
            'min_learning_rate': 0.01,
            'seed': 1,
            'bins': (1,1,6,3),
            'num_action_space': env.action_space.n
        }

        q_learning = Q_Learning(config)
        episode_rewards, convergence_episode = q_learning.train()

        average_reward = np.mean(episode_rewards)
        print(f"Tested {params} with average reward: {average_reward}")

        if average_reward > best_reward:
            best_reward = average_reward
            best_params = params

    print(f"Best parameters: {best_params} with reward: {best_reward}")

if __name__ == "__main__":
    main()
