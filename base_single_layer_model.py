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

class base_single_layer_model(mancala_model):
    model_name = "Basic_1_layer"

    def __init__(self):
        super(base_single_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16) 
        self.fc2 = nn.Linear(16, 6)
        self.fitness = 0
        self.flattened_parameters = None

    #Not currently used
    def get_flattened_parameters(self):
        #They need to be recomputed a bit, this will cache them.
        if self.flattened_parameters is not None:
            return self.flattened_parameters
        else:
            #self.flattened_parameters = torch.cat([param.data.view(-1) for param in self.parameters()]) #faster than flatten because it skips autograd... don't think that'll cause issues?
            self.flattened_parameters = parameters_to_vector(self.parameters()) #Almost surely better... but is it slower?
        return self.flattened_parameters

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.argmax(x, dim=1)
    

    
