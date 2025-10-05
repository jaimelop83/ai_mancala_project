"""
Purpose:
    Standardizes the model reference
"""

import torch
seed = 49
torch.manual_seed(seed)

#Changing this model will change it everywhere else: nothing should import models directly, use from network import model. 
from base_single_layer_model import base_single_layer_model as model

#Hyperparameters
DEATH_RATE = 0.20 #This percent die every generation
REPRODUCTIVE_FLOOR = 0.50 #Above this precent in fitness, reproduces
MUTATION_RATE = 0.00 #Not yet implemented, unclear if even needed

MODEL_SIMILARITY = 0.50 #Percent of parameters to take from model1 when reproducing

FOOD_SIZE = 3 #Number of boards/games each generation will play.
POPULATION_SIZE = 10000 #Number of models in the population #Around 10,000 is what starts to take nontrivial time on my cpu.
NUMBER_OF_GENERATIONS = 3 #Number of generations to evolve

NUMBER_OF_THREADS = 8 

print("Hyperparameters:")
print(f"\t{DEATH_RATE=}")
print(f"\t{REPRODUCTIVE_FLOOR=}")
print(f"\t{MUTATION_RATE=}")
print(f"\t{MODEL_SIMILARITY=}")
print(f"\t{FOOD_SIZE=}")
print(f"\t{POPULATION_SIZE=}")
print(f"\t{NUMBER_OF_GENERATIONS=}")
print(f"\t{seed=}")

