# A Comparative Study of Q-Learning and Policy Gradient Algorithms in Reinforcement Learning Applied to an Inverted Pendulum

## Overview

This repository contains the implementation of Q-Learning and Policy Gradient algorithms to solve the Cart-Pole problem using the OpenAI Gym environment. The goal is to balance a pole on a moving cart by applying appropriate forces to the cart.

- **policy_gradient/**: Contains the implementation of the Policy Gradient algorithm.
  - `policy.py`: Defines the policy network.
  - `reinforce.py`: Contains the REINFORCE algorithm for training the policy network.
  - `run.py`: Script to run the Policy Gradient algorithm.
  
- **q_learning/**: Contains the implementation of the Q-Learning algorithm.
  - `hyperparameters.py`: Defines the hyperparameters for Q-Learning.
  - `q_learning.py`: Contains the Q-Learning algorithm.
  - `run.py`: Script to run the Q-Learning algorithm.
  