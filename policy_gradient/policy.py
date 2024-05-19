import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer = nn.Linear(n_observations, n_actions)

    def forward(self, state):
        state = self.layer(state)
        return F.softmax(state, dim=1)