"""
Purpose:
    Give a basic neural network that we could use to solve this problem.
    I am trying to keep it as simple as possible to run test code and such, but to still be technically viable.
    
Internal layers:
    1 hidden layer with 16 nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from mancala_model import mancala_model

class updated_three_layer_model(mancala_model):
    model_name = "Updated_3_layer"

    def __init__(self):
        super(updated_three_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 32) 
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 8) 
        self.fc4 = nn.Linear(8, 6)
        self.fitness = 0
        self.special = False
        self.eval()
        self.wins = 0

    def forward(self, x):
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0,:6]]) 
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        masked_x = x.clone()
        masked_x[0,legal_mask == 0] = -float('inf')
        move = torch.argmax(masked_x, dim=1)
        return move

    
