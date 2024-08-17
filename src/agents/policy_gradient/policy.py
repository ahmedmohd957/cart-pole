import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(p=0.6)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        return F.softmax(self.layer2(x), dim=1)
    
    def action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


# Save the trained policy
def save_model(path):
    torch.save(Policy.state_dict(), path)

# Load the saved policy
def load_model(env, path):
    model = Policy(env.observation_space.shape[0], env.action_space.n, 128)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
