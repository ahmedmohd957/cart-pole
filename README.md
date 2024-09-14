# A Comparative Study of Q-Learning and Policy Gradient Algorithms in Reinforcement Learning Applied to an Inverted Pendulum

## Overview

This repository contains the implementation of Q-Learning and Policy Gradient (REINFORCE) algorithms to solve the Cart-Pole problem using the OpenAI Gym environment.

<img src="data/cart_pole.png" alt="CartPole Environment" width="400" height="300"/>

## Setting Up the Environment

To set up the environment needed to run this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/ahmedmohd957/cart-pole.git
   ```

   ```bash
   cd cart-pole
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate cartpole
   ```

4. Run the Q-learning agent:

   ```bash
   python src/agents/q_learning/train.py
   ```

5. Run the REINFORCE agent:

   ```bash
   python src/agents/policy_gradient/train.py
   ```

<br/>
