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
import random

class two_layer_model(mancala_model):
    model_name = "Basic_2_layer"

    def __init__(self):
        super(two_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16)  
        self.fc2 = nn.Linear(16, 8) 
        self.fc3 = nn.Linear(8, 6)
        self.fitness = 0
        self.flattened_parameters = None

        #The ratio of moves to choose randomly
        #self.random_tendency = random.random()
        if random.random() < 0.01:
            self.faker = True
        else:
            self.faker = False

    def forward(self, x):
        # print(f"{x=}")
        #print(f"{type(x)=}")
        #print(f"{x[0,2]=}")
        #print(f"{x[0,2]!=0=}")
        #print(f"{type(x[0,2])=}")
        #if self.random_tendency > 0.5 and random.random() < self.random_tendency:
        if self.faker:
            #print(".", end="")
            legal_moves = [i for i in range(6) if x[0,i] != 0]
            #print(f"{torch.tensor(random.choice(legal_moves))}", end="")
            if legal_moves:
                    return torch.tensor(random.choice(legal_moves))
            else:
                print(f"{x=}")
                print("This should have never occured!")
                assert(False)
                return torch.tensor(0)
        #print("*", end="")
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #return torch.argmax(x, dim=1)

        #For batched runs. 
        if x.ndim == 1:
            return torch.argmax(x)
        return torch.argmax(x, dim=1)
    

    
