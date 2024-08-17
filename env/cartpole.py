import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, render_mode=None, variant='D'):
        super(CustomCartPoleEnv, self).__init__(render_mode=render_mode)
        self.variant = variant
        self._set_parameters()
    
    def _set_parameters(self):
        if self.variant == 'D':
            self.force_mag = 10.0
            self.length = 0.5
            self.masspole = 0.1
        elif self.variant == 'R':
            self.force_mag = np.random.uniform(5, 15)
            self.length = np.random.uniform(0.25, 0.75)
            self.masspole = np.random.uniform(0.05, 0.5)
        elif self.variant == 'E':
            self.force_mag = np.random.choice([np.random.uniform(1, 5), np.random.uniform(15, 20)])
            self.length = np.random.choice([np.random.uniform(0.05, 0.25), np.random.uniform(0.75, 1.0)])
            self.masspole = np.random.choice([np.random.uniform(0.01, 0.05), np.random.uniform(0.5, 1.0)])

        # Update dependent parameters
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
    
    def reset(self, **kwargs):
        self._set_parameters()
        return super(CustomCartPoleEnv, self).reset(**kwargs)
