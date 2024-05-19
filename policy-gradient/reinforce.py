import time
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy import Policy

class REINFORCE:
    def __init__(self, config):
        self.env = config["env"]
        self.episodes = config["episodes"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.seed = config["seed"]
        self.log_interval = config["log_interval"]

        # Set the seed
        self.env.reset(seed=self.seed)
        torch.manual_seed(self.seed)

        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()
        
        # self.total_rewards = []
        self.running_rewards = []
        self.policy_losses = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        returns = deque()
        
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        self.policy_losses.append(policy_loss.item())

        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        episode_rewards = np.zeros(self.episodes)
        average_reward = []
        convergence_episode = None

        start = time.time()
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            cumulative_reward = 0.0

            done = False
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                self.policy.rewards.append(reward)
                cumulative_reward += reward
                if terminated or truncated:
                    break
            
            episode_rewards[episode] = cumulative_reward
            self.update()

            # Check for convergence
            if episode >= 100:
                avg_reward = sum(episode_rewards[episode-100:episode]) / 100
                average_reward.append(avg_reward)
                if avg_reward >= 475 and convergence_episode is None:
                    convergence_episode = episode
            else:
                average_reward.append(sum(episode_rewards[:episode]) / 100)

            if ((episode % 10) == 0):
                print(episode, cumulative_reward, sep=',')

        end = time.time()
        training_time = end - start
        return episode_rewards, average_reward, convergence_episode, training_time
