import torch
import torch.nn as nn
import torch.nn.functional as F
from mancala_model import mancala_model

class updated_fancy_model(mancala_model):
    model_name = "updated_fancy"

    def __init__(self):
        super(updated_fancy_model, self).__init__()

        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 16)

        # Bottleneck aligned to actions
        self.fc3 = nn.Linear(16, 6)

        # Re-expansion for move comparison
        self.fc4 = nn.Linear(6, 16)
        self.fc5 = nn.Linear(16, 6)

        self.fitness = 0
        self.wins = 0
        self.special = False
        self.eval()

    def forward(self, x):
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0,:6]])

        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        masked_x = x.clone()
        masked_x[0, legal_mask == 0] = -float('inf')

        return torch.argmax(masked_x, dim=1)
