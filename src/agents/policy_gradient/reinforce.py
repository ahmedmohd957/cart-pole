import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
from tqdm import tqdm
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from policy import Policy
from src.utils.helper_functions import update_rolling_average

class ReinforceAgent():
    def __init__(self, config):
        self.env = config["env"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.hidden_size = config["hidden_size"]
        self.episodes = config["episodes"]
        self.max_steps = config["max_steps"]

        # Set the seed
        torch.manual_seed(540)
        
        self.saved_log_probs = []
        self.rewards = []
        self.policy_losses = []

        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n, self.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

    # Compute discounted returns
    def compute_returns(self, rewards, gamma):
        R = 0
        returns = deque()
        for r in reversed(rewards):
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    # Compute Policy Loss
    def compute_policy_loss(self, saved_log_probs, returns):
        policy_loss = [-log_prob * R for log_prob, R in zip(saved_log_probs, returns)]
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    # Backpropagation
    def backpropagate(self, policy_loss, optimizer):
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    # Update Policy after episode
    def update(self):    
        returns = self.compute_returns(self.rewards, self.gamma)
        policy_loss = self.compute_policy_loss(self.saved_log_probs, returns)
        self.policy_losses.append(policy_loss.item())
        self.backpropagate(policy_loss, self.optimizer)
        del self.rewards[:]
        del self.saved_log_probs[:]

    # Train agent
    def train(self):
        episode_rewards = np.zeros(self.episodes)
        average_rewards = []
        solved_episode = None

        start = time.time()
        for episode in tqdm(range(self.episodes)):
            state, _ = self.env.reset()
            cumulative_reward = 0.0
            for _ in range(self.max_steps):
                action, log_probs = self.policy.action(state)
                self.saved_log_probs.append(log_probs)
                state, reward, terminated, _, _ = self.env.step(action)
                self.rewards.append(reward)
                cumulative_reward += reward
                if terminated:
                    break

            episode_rewards[episode] = cumulative_reward

            # Update policy
            self.update()

            solved_episode = update_rolling_average(episode, episode_rewards, average_rewards, solved_episode)

        end = time.time()
        training_time = end - start
        return episode_rewards, average_rewards, solved_episode, training_time
