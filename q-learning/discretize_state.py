import numpy as np

class StateDiscretizer:
    def __init__(self, env):
        self.bins = [30, 30, 30, 30]
        self.lower_bounds = [env.observation_space.low[0], -3, env.observation_space.low[2], -10]
        self.upper_bounds = [env.observation_space.high[0], 3, env.observation_space.high[2], 10]

    def discretize(self, state):
        cart_position_bins = np.linspace(self.lower_bounds[0], self.upper_bounds[0], self.bins[0])
        cart_velocity_bins = np.linspace(self.lower_bounds[1], self.upper_bounds[1], self.bins[1])
        pole_angle_bins = np.linspace(self.lower_bounds[2], self.upper_bounds[2], self.bins[2])
        pole_angular_velocity_bins = np.linspace(self.lower_bounds[3], self.upper_bounds[3], self.bins[3])
        
        index_position = np.maximum(np.digitize(state[0], cart_position_bins)-1, 0)
        index_velocity = np.maximum(np.digitize(state[1], cart_velocity_bins)-1, 0)
        index_angle = np.maximum(np.digitize(state[2], pole_angle_bins)-1, 0)
        index_angular_velocity = np.maximum(np.digitize(state[3], pole_angular_velocity_bins)-1, 0)

        return tuple([index_position, index_velocity, index_angle, index_angular_velocity])