import gymnasium as gym
import numpy as np
from discretize_state import StateDiscretizer

class QLearningSimulator:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.q_table = self.load_q_table('trained_q_table.npy')
        self.discretizer = StateDiscretizer(self.env)

    def load_q_table(self, file_name):
        q_table = np.load(file_name)
        return q_table

    def choose_action(self, state):
        return np.random.choice(np.where(self.q_table[self.discretizer.discretize(state)]==np.max(self.q_table[self.discretizer.discretize(state)]))[0])

    def simulate(self, episodes):
        (state, _) = self.env.reset()
        self.env.render()
        rewards = []
        
        for episode in range(episodes):
            print(episode)
            action = self.choose_action(state)
            state, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)
            if (done):
                break
            
        return rewards, self.env

if __name__ == "__main__":
    simulator = QLearningSimulator()
    simulator.simulate(1000)
        