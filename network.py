"""
Purpose:
    Standardizes the model reference
"""

import torch
import re
import subprocess
import psutil

seed = 49
torch.manual_seed(seed)
torch.set_grad_enabled(False) #.backward is never used


#Changing this model will change it everywhere else: nothing should import models directly, use from network import model. 
"""
Input layer:
    Standard 14 nodes, ints in [0,48], representing the board state.

Internal layers:
    These may vary. 

Output layer:
    1 value in [0,5] representing the hole to choose.
"""
#from base_single_layer_model import base_single_layer_model as model
#from transformer_model import TinyTransformerModel as model
from three_layer_model import three_layer_model as model

#Hyperparameters
DEATH_RATE = 0.50 #This percent die every generation
REPRODUCTIVE_FLOOR = .9 #Above this precent in fitness, reproduces (0.85 = top 15% can reproduce)
MUTATION_RATE = 0.5 #Mutants to make each generation 
MUTATION_AMOUNT = 0.05 #Weights to randomize change each mutant
#NUMBER_PROTECTED = 25 #The number of models that are protected from generation to generation - The best ones.
FITNESS_DECAY_RATE = 0.9

MODEL_SIMILARITY = 0.50 #Percent of parameters to take from model1 when reproducing

FOOD_SIZE = 5 #Number of boards/games each generation will play.
POPULATION_SIZE = 1000 #Number of models in the population #Around 10,000 is what starts to take nontrivial time on my cpu.
NUMBER_OF_GENERATIONS = 20000 #Number of generations to evolve

NUMBER_OF_THREADS = 8 #Currently unused
SAVE_DIR = "recent_models" #The top 10 will be saved here AND OVERWRITTEN EVERY TIME IT IS RUN

print("Hyperparameters:")
print(f"\t{model.model_name=}")
print(f"\t{DEATH_RATE=}")
print(f"\t{REPRODUCTIVE_FLOOR=}")
print(f"\t{MUTATION_RATE=}")
print(f"\t{MUTATION_AMOUNT=}")
print(f"\t{FITNESS_DECAY_RATE=}")
print(f"\t{MODEL_SIMILARITY=}")
print(f"\t{FOOD_SIZE=}")
print(f"\t{POPULATION_SIZE=}")
print(f"\t{NUMBER_OF_GENERATIONS=}")
print(f"\t{NUMBER_OF_THREADS=}")
print(f"\t{SAVE_DIR=}")
print(f"\t{seed=}")


print("System Info:")
if torch.cuda.is_available():
    print("\tCUDA is available. PyTorch can use the GPU.")
    print(f"\tGPU Name: {torch.cuda.get_device_name(0)}")
    print(f"\tCurrent device: {torch.cuda.current_device()}")
    ram = torch.cuda.get_device_properties(0).total_memory
    print(f"\tAvailable GPU RAM: { ram / 1e9:.2f} GB")
else:
    print("\tUsing CPU because CUDA is not available.")
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True)
        print("\tA GPU was found but unused: ", end="")
        print(re.sub(r'\s+', ' ', result.stdout.strip()))
        ram = psutil.virtual_memory().available
        print(f"\tAvailable CPU RAM: {ram / 1e9:.2f} GB")
    except Exception as e:
        print("Failed to determine if the system has a GPU")

print("Evolution Info:")
test_model = model()
model_size = sum(p.numel() * p.element_size() for p in test_model.parameters())
print(f"\t{model_size=:,} bytes")
print(f"\tLargest reasonable population is thus: {ram // model_size:,}") #But keep it a good bit smaller! Perhaps max should be 1/2 or 1/3? TODO look at memory storage and make sure population is the dominant thing.