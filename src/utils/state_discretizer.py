import numpy as np

class StateDiscretizer:
    def __init__(self, env, bins=(40, 40, 40, 40)):
        """
        Initialize the StateDiscretizer with the environment and bins.

        Parameters:
        - env: The environment object with an observation_space attribute.
        - bins: A list or tuple specifying the number of bins for each dimension.
        """
        self.env = env
        self.bins = bins
    
    def discretize_state(self, state):
        """
        Discretizes the continuous state into a discrete state index.

        Parameters:
        - state: A list or array representing the continuous state.

        Returns:
        - A tuple representing the discrete state index.
        """
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
