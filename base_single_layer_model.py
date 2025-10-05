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
from torch.nn.utils import parameters_to_vector

class base_single_layer_model(nn.Module):
    def __init__(self):
        super(base_single_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16) 
        self.fc2 = nn.Linear(16, 6)
        self.fitness = 0
        self.flattened_parameters = None

    def get_flattened_parameters(self):
        #They need to be recomputed a bit, this will cache them.
        if self.flattened_parameters is not None:
            return self.flattened_parameters
        else:
            #self.flattened_parameters = torch.cat([param.data.view(-1) for param in self.parameters()]) #faster than flatten because it skips autograd... don't think that'll cause issues?
            self.flattened_parameters = parameters_to_vector(self.parameters()) #Almost surely better... but is it slower?
        return self.flattened_parameters

    def set_flattened_parameters(self, flat_params):
        """
        Purpose:
            Set the parameters of the model from a flattened tensor.
        Input:
            flat_params: A 1D tensor containing the new parameter values.
        """
        pointer = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = flat_params[pointer:pointer + numel].view_as(param).clone()
            pointer += numel

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.argmax(x, dim=1)
    
