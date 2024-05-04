import gym
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy import Policy



class Reinforce:
    def __init__(self, config):
        self.learning_rate = config["learning_rate"]
        self.discount_factor = config["discount_factor"]
        self.seed = config["seed"]
        self.n_episodes = config["n_episodes"]
        self.n_max_steps = config["n_max_steps"]
        self.log_interval = config["log_interval"]



        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=self.seed)
        torch.manual_seed(self.seed)

        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()
        self.total_rewards = []
        self.running_rewards = []
        self.policy_losses = []


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.policy.rewards[::-1]:
            R = r + self.discount_factor * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        self.policy_losses.append(policy_loss.item())  # Store policy loss after each episode

        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        running_reward = 10
        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            episode_rewards = []
            
            while range(self.n_max_steps):
                action = self.select_action(state)
                state, reward, done, truncated, _ = self.env.step(action)
                self.policy.rewards.append(reward)
                episode_rewards.append(reward)
                if done or truncated:
                    break
            
            self.total_rewards.append(np.sum(episode_rewards))
            running_reward = 0.05 * np.sum(episode_rewards) + (1 - 0.05) * running_reward
            self.running_rewards.append(running_reward)
            self.finish_episode()
            

            if episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    episode, np.sum(episode_rewards), running_reward))
                #if running_reward > self.env.spec.reward_threshold:
                    #print("Solved! Running reward is now {}".format(running_reward))
                    #break
        return np.mean(self.running_rewards[-10:])  # Return the average of the last 10 episodes' rewards
    
    def test(self, n_test_episodes=100, render=False):
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.env.reset(seed=self.seed)

        test_rewards = []
        for episode in range(n_test_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            while True:
                if render:
                    self.env.render()  # Specify rendering mode
                action = self.select_action(state)
                state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                if done or truncated:
                    break
            test_rewards.append(episode_reward)
        if render:
            self.env.close()  # Ensure the environment is closed after rendering
        return test_rewards



    