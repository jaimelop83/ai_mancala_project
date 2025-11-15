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

class updated_single_layer_model(mancala_model):
    model_name = "Updated_1_layer"

    def __init__(self):
        super(updated_single_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16) 
        self.fc2 = nn.Linear(16, 6)
        self.fitness = 0
        self.special = False
        self.eval()
        self.wins = 0

    def forward(self, x):
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0,:6]]) 
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        masked_x = x.clone()
        masked_x[0,legal_mask == 0] = -float('inf')
        move = torch.argmax(masked_x, dim=1)
        return move
    

    
