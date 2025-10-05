"""
Purpose:
    Standardizes the model reference
"""

import torch
seed = 49
torch.manual_seed(seed)
print(f"Setting manual seed as {seed}")

#Changing this model will change it everywhere else: nothing should import models directly, use from network import model. 
from base_single_layer_model import base_single_layer_model as model

#Hyperparameters
DEATH_RATE = 0.20 #This percent die every generation
REPRODUCTIVE_FLOOR = 0.50 #Above this precent in fitness, reproduces
MUTATION_RATE = 0.00 #Not yet implemented, unclear if even needed

MODEL_SIMILARITY = 0.50 #Percent of parameters to take from model1 when reproducing
