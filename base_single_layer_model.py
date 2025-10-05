"""
Purpose:
    Give a basic neural network that we could use to solve this problem.
    I am trying to keep it as simple as possible to run test code and such, but to still be technically viable.
    
Input layer:
    Standard 14 nodes, ints in [0,48], representing the board state.

Internal layers:
    1 hidden layer with 16 nodes

Output layer:
    1 value in [0,5] representing the hole to choose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class base_single_layer_model(nn.Module):
    def __init__(self):
        super(base_single_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16) 
        self.fc2 = nn.Linear(16, 6)


    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.argmax(x, dim=1)
    
