import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import numpy as np
from tqdm import tqdm
from src.utils.state_discretizer import StateDiscretizer
from src.utils.helper_functions import update_rolling_average

class QLearningAgent:
    def __init__(self, config):
        self.env = config["env"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.episodes = config["episodes"]
        self.max_steps = config["max_steps"]
        self.bins = config["bins"]
        self.n_actions = self.env.action_space.n

        # Set the seed
        np.random.seed(seed=1)
        
        # Initialize Q-Table
        self.q_table = np.random.uniform(low=0, high=1, size=(*self.bins, self.n_actions))
        self.discretizer = StateDiscretizer(self.env, self.bins)

    # Select action
    def select_action(self, state, index):
        if index < 100:
            return np.random.choice(self.n_actions)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    # Update the Q-Table
    def update_q_table(self, state, action, reward, next_state, terminated):
        q_max_next = np.max(self.q_table[next_state])
        state_action = state + (action,)

        if not terminated:
            td_error = reward + self.gamma * q_max_next - self.q_table[state_action]
            self.q_table[state_action] += self.learning_rate * td_error
        else:
            td_error = reward - self.q_table[state_action]
            self.q_table[state_action] += self.learning_rate * td_error

    # Decay epsilon
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    # Train the agent
    def train(self):
        episode_rewards = np.zeros(self.episodes)
        average_rewards = []
        solved_episode = None

        start = time.time()
        for episode in tqdm(range(self.episodes)):    
            state, _ = self.env.reset()
            cumulative_reward = 0.0
            for _ in range(self.max_steps):
                discretized_state = self.discretizer.discretize_state(state)
                action = self.select_action(discretized_state, episode)
                next_state, reward, terminated, _, _ = self.env.step(action)
                discretized_next_state = self.discretizer.discretize_state(next_state)
                self.update_q_table(discretized_state, action, reward, discretized_next_state, terminated)
                cumulative_reward += reward
                state = next_state
                if terminated:
                    break
            
            episode_rewards[episode] = cumulative_reward

            if episode > 700:
                self.decay_epsilon()
            
            solved_episode = update_rolling_average(episode, episode_rewards, average_rewards, solved_episode)
            
        end = time.time()
        training_time = end - start
        return episode_rewards, average_rewards, solved_episode, training_time
