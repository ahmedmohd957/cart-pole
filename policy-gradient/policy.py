import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)  # First layer
        self.affine2 = nn.Linear(128, 64)  # New second layer
        self.dropout = nn.Dropout(p=0.6)  # Dropout can be applied here
        self.affine3 = nn.Linear(64, 2)  # Output layer

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.affine2(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)  
