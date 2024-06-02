import time
import numpy as np

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
        self.seed = config["seed"]
        self.n_actions = self.env.action_space.n

        # Set the seed
        np.random.seed(seed=self.seed)
        
        # Initialize Q-Table
        self.q_table = np.random.uniform(low=0, high=1, size=(*self.bins, self.n_actions))
    
    # Discretize state
    def discretize_state(self, state):
        low = [-4.8, -3, self.env.observation_space.low[2], -10]
        high = [4.8, 3, self.env.observation_space.high[2], 10]

        state_bins = [
            np.linspace(low[0], high[0], self.bins[0]),
            np.linspace(low[1], high[1], self.bins[1]),
            np.linspace(low[2], high[2], self.bins[2]),
            np.linspace(low[3], high[3], self.bins[3])
        ]

        state_index = []
        for i in range(len(state)):
            index = np.maximum(np.digitize(state[i], state_bins[i]) - 1, 0)
            state_index.append(index)
        return tuple(state_index)

    # SELECT ACTION
    def select_action(self, state, index):
        if index < 100:
            return np.random.choice(self.n_actions)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    # UPDATE THE Q-TABLE
    def update_q_table(self, state, action, reward, next_state, terminated):
        q_max_next = np.max(self.q_table[next_state])
        state_action = state + (action,)

        if not terminated:
            td_error = reward + self.gamma * q_max_next - self.q_table[state_action]
            self.q_table[state_action] += self.learning_rate * td_error
        else:
            td_error = reward - self.q_table[state_action]
            self.q_table[state_action] += self.learning_rate * td_error

    # DECAY EPSILON
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    # Train
    def train(self):
        episode_rewards = np.zeros(self.episodes)
        average_reward = []
        solved_episode = None

        start = time.time()
        for episode in range(self.episodes):    
            state, _ = self.env.reset()
            cumulative_reward = 0.0
            for _ in range(self.max_steps):
                discretized_state = self.discretize_state(state)
                action = self.select_action(discretized_state, episode)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                discretized_next_state = self.discretize_state(next_state)
                self.update_q_table(discretized_state, action, reward, discretized_next_state, terminated)
                cumulative_reward += reward
                state = next_state
                if terminated:
                    break
            
            episode_rewards[episode] = cumulative_reward

            if episode > 700:
                self.decay_epsilon()
            
            if episode >= 100:
                avg_reward = sum(episode_rewards[episode-100:episode]) / 100
                average_reward.append(avg_reward)
                if avg_reward >= 475 and solved_episode is None:
                    solved_episode = episode
            else:
                average_reward.append(sum(episode_rewards[:episode]) / 100)

            print(f"Episode: {episode}, Rewards: {cumulative_reward}")
            
        end = time.time()
        training_time = end - start
        return episode_rewards, average_reward, solved_episode, training_time
