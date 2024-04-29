import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 
from q_learning import Q_Learning

env = gym.make('CartPole-v1')
(state, _) = env.reset()

config = {
    'env': env,
    'alpha': 0.1,
    'gamma': 1,
    'epsilon': 0.2,
    'bins': [30, 30, 30, 30],  # position, velocity, angle, angle velocity
    'num_action_space': env.action_space.n
}

Q1 = Q_Learning(config)
Q1.train(15000)