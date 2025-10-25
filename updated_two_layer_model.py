"""
Purpose:
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mancala_model import mancala_model
import random
from network import PERCENT_RANDOM

class updateded_two_layer_model(mancala_model):
    model_name = "Updated_2_layer"

    def __init__(self):
        super(updateded_two_layer_model, self).__init__()
        self.fc1 = nn.Linear(14, 16)  
        self.fc2 = nn.Linear(16, 8) 
        self.fc3 = nn.Linear(8, 6)
        self.fitness = 0
        self.special = False
        self.eval()
        self.wins = 0


    def forward(self, x):
        # print(f"{x=}")
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0,:6]]) 
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(f"!{x=}")
        
        # print(f"{legal_mask=}")
        masked_x = x.clone()
        masked_x[0,legal_mask == 0] = -float('inf')
        move = torch.argmax(masked_x, dim=1)
        # print(f"{move=}")
        return move
    

    
