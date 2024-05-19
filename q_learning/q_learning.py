import math
import random
import time
import numpy as np
from collections import defaultdict


class Q_Learning:
    def __init__(self, config):
        self.env = config["env"]
        self.episodes = config["episodes"]
        self.epsilon = config["epsilon"]
        self.min_epsilon = config["min_epsilon"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.min_learning_rate = config["min_learning_rate"]
        self.seed = config["seed"]
        self.bins = config["bins"]
        self.n_actions = self.env.action_space.n

        # Set the seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize q_table
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    # Discretize state
    def discretize_state(self, state, bounds):
        # Number of bins for each dimension (x, x_dot, theta, theta_dot)
        bins = (1, 1, 6, 3)
        bin_indices = []
        for i in range(len(state)):
            if state[i] <= bounds[i][0]:
                bin_index = 0
            elif state[i] >= bounds[i][1]:
                bin_index = bins[i] - 1
            else:
                bound_width = bounds[i][1] - bounds[i][0]
                offset = (bins[i] - 1) * bounds[i][0] / bound_width
                scale_factor = (bins[i] - 1) / bound_width
                bin_index = int(round(scale_factor * state[i] - offset))
            bin_indices.append(bin_index)
        return tuple(bin_indices)

    # SELECT ACTION
    def select_action(self, state):
        if np.random.random() > self.epsilon:
            return int(np.argmax(self.q_table[state]))
        else:
            return random.randint(0, self.n_actions - 1)

    # UPDATE THE Q-TABLE
    def update_q_table(self, state, action, reward, next_state, terminated):
        q_max_next = np.max(self.q_table[next_state])
        if not terminated:
            td_error = reward + self.gamma * q_max_next - self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * td_error
        else:
            td_error = reward - self.q_table[state][action]
            self.q_table[state][action] += self.learning_rate * td_error

    # DECAY EPSILON
    def decay_epsilon(self, step):
        self.epsilon = max(self.min_epsilon, min(1.0, 1.0 - math.log10((step + 1) / 25)))

    # DECAY LEARNING RATE
    def decay_learning_rate(self, step):
        self.learning_rate = max(self.min_learning_rate, min(1.0, 1.0 - math.log10((step + 1) / 25)))

    # Train
    def train(self):
        obs_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        episode_rewards = np.zeros(self.episodes)
        average_reward = []
        convergence_episode = None

        start = time.time()
        for episode in range(self.episodes):
            observation, _ = self.env.reset(seed=self.seed)
            state = self.discretize_state(observation, obs_bounds)

            self.decay_epsilon(episode)
            self.decay_learning_rate(episode)

            cumulative_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_obs, obs_bounds)
                self.update_q_table(state, action, reward, next_state, terminated)
                cumulative_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            episode_rewards[episode] = cumulative_reward

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
