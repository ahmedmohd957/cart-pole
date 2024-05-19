import time
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from policy import Policy

class REINFORCE:
    def __init__(self, config):
        self.env = config['env']
        self.episodes = config['episodes']
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.seed = config['seed']

        # Set the seed
        torch.manual_seed(self.seed)

        observations = self.env.observation_space.shape[0]
        actions = self.env.action_space.n
        self.policy = Policy(observations, actions)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate)

        self.probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state)
        c = Categorical(probs)
        action = c.sample()
        self.probs.append(c.log_prob(action))
        return action.item()

    def update(self):
        g = 0
        returns = []

        for r in self.rewards[::-1]:
            g = r + self.gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns)

        loss = 0
        for log_prob, delta in zip(self.probs, returns):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

    def train(self):
        episode_rewards = np.zeros(self.episodes)
        average_reward = []
        convergence_episode = None

        start = time.time()
        for episode in range(self.episodes):
            state, _ = self.env.reset(seed=self.seed)
            cumulative_reward = 0.0

            done = False
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                cumulative_reward += reward
                self.rewards.append(reward)
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
