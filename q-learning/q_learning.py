import numpy as np
from discretize_state import StateDiscretizer

class Q_Learning:
    def __init__(self, config):
        import numpy as np
        
        self.env = config["env"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"] 
        self.epsilon = config['epsilon']
        self.actionNumber = config['num_action_space']
        self.bins = config["bins"]
        
        self.total_rewards = []
        self.q_table = np.random.uniform(low=0, high=1, size=(self.bins[0], self.bins[1], self.bins[2], self.bins[3], self.actionNumber))
        self.discretizer = StateDiscretizer(self.env)

    def choose_action(self,state,index):
        if index < 500:
            return np.random.choice(self.actionNumber)   
             
        random_number = np.random.random()
         
        if index > 7000:
            self.epsilon = 0.999*self.epsilon
         
        if random_number < self.epsilon:
            return np.random.choice(self.actionNumber)            
        else:
            return np.random.choice(np.where(self.q_table[self.discretizer.discretize(state)]==np.max(self.q_table[self.discretizer.discretize(state)]))[0])

    def train(self, episodes):
        import numpy as np
        
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

                q_max_next = np.max(self.q_table[discrete_next_state])     
                
                if not done:
                    error = reward+self.gamma*q_max_next-self.q_table[discrete_state+(action,)]
                    self.q_table[discrete_state+(action,)]=self.q_table[discrete_state+(action,)]+self.alpha*error
                else:
                    error = reward-self.q_table[discrete_state+(action,)]
                    self.q_table[discrete_state+(action,)]=self.q_table[discrete_state+(action,)]+self.alpha*error
                 
                state = next_state
            
            np.save('trained_q_table.npy', self.q_table)
            print("Sum of rewards {}".format(np.sum(episode_rewards)))        
            self.total_rewards.append(np.sum(episode_rewards))
