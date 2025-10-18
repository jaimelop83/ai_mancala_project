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

class three_layer_model(mancala_model):
    model_name = "Basic_3_layer"

    def __init__(self):
        super(three_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 32) 
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 8) 
        self.fc4 = nn.Linear(8, 6)
        self.fitness = 0
        self.flattened_parameters = None

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #return torch.argmax(x, dim=1)

        #For batched runs. 
        if x.ndim == 1:
            return torch.argmax(x)
        return torch.argmax(x, dim=1)
    

    
