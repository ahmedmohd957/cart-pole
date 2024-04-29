import numpy as np
from discretize_state import StateDiscretizer

class Q_Learning:
    def __init__(self, config):
        self.env = config["env"]
        self.learning_rate = config["learning_rate"]
        self.discount_factor = config["discount_factor"] 
        self.epsilon = config['epsilon']
        self.epsilon_decay = config["epsilon_decay"]
        self.num_action_space = config['num_action_space']
        self.bins = config["bins"]
        
        self.total_rewards = []
        self.q_table = np.random.uniform(low=0, high=1, size=(self.bins[0], self.bins[1], self.bins[2], self.bins[3], self.num_action_space))
        self.discretizer = StateDiscretizer(self.env)

    def random_action(self):
        return np.random.choice(self.num_action_space)
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return self.random_action()        
        else:
            return np.random.choice(np.where(self.q_table[self.discretizer.discretize(state)]==np.max(self.q_table[self.discretizer.discretize(state)]))[0])

    def choose_action(self, state, index):
        if index < 500:
            return self.random_action()
        
        if index > 7000:
            self.decay_epsilon()
        
        return self.epsilon_greedy_policy(state)
    
    def update_q_table(self, state, action, reward, next_state, done):
        q_max_next = np.max(self.q_table[next_state])

        if not done:
            error = reward + self.discount_factor * q_max_next - self.q_table[state + (action,)]
            self.q_table[state + (action,)] = self.q_table[state + (action,)] + self.learning_rate * error
        else:
            error = reward - self.q_table[state + (action,)]
            self.q_table[state + (action,)] = self.q_table[state + (action,)] + self.learning_rate * error

    def train(self, episodes):
        for indexEpisode in range(episodes):
            episode_rewards = []
            
            (state, _) = self.env.reset()
            state = list(state)
           
            print("Simulating episode {}".format(indexEpisode))
            
            done = False
            while not done:
                discrete_state = self.discretizer.discretize(state)
                action = self.choose_action(state, indexEpisode)
                (next_state, reward, done, _, _) = self.env.step(action)          
                episode_rewards.append(reward)
                next_state = list(next_state)
                discrete_next_state = self.discretizer.discretize(next_state)

                self.update_q_table(discrete_state, action, reward, discrete_next_state, done)
                
                state = next_state
            
            np.save('trained_q_table.npy', self.q_table)
            print("Sum of rewards {}".format(np.sum(episode_rewards)))        
            self.total_rewards.append(np.sum(episode_rewards))
