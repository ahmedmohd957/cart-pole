import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from env.cartpole import CustomCartPoleEnv
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from qlearning import QLearningAgent

# Set up environment
env = CustomCartPoleEnv()

# Define the parameter space
param_space = [
    Real(0.1, 1.0, name='learning_rate'),
    Real(0.7, 1.0, name='gamma'),
    Real(0.1, 1.0, name='epsilon'),
    Real(0.4, 1.0, name='epsilon_decay'),
    Integer(30, 60, name='bins')
]

# Define the objective function
@use_named_args(param_space)
def objective(**params):
    env.reset(seed=42)
    config = {
        'env': env,
        'learning_rate': params['learning_rate'],
        'gamma': params['gamma'],
        'epsilon': params['epsilon'],
        'epsilon_decay': params['epsilon_decay'],
        'episodes': 2000,
        'max_steps': 1000,
        'bins': (params['bins'], params['bins'], params['bins'], params['bins']),
    }
    agent = QLearningAgent(config)
    episode_rewards, _, _, _ = agent.train()    
    mean_reward = np.mean(episode_rewards)

    print(f"{params} - mean_reward: {mean_reward}")
    return -mean_reward

# Run Bayesian optimization
result = gp_minimize(objective, param_space, n_calls=2500)

# Extract the best parameters
best_params = result.x
best_score = -result.fun

# Print best parameters
print("Best parameters found:")
print("Learning rate:", best_params[0])
print("Gamma:", best_params[1])
print("Epsilon:", best_params[2])
print("Epsilon decay:", best_params[3])
print("Bins:", best_params[4])
print("Best score:", best_score)
