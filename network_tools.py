from base_single_layer_model import base_single_layer_model
import torch
import numpy as np
import network

#This should probably be moved to the game or board infrastructure
#Perhaps return a board object, that can convert to torch on its own?
def random_board():
    out = np.random.multinomial(48, [1/14.]*14, size=1)[0]
    sum = out.sum()
    if sum != 48:
        print(f"Error: invalid randomly generated board with {sum=} stones. Discarding and trying again.")
        return random_board()
    #print(f"{out=}")
    return out

def run_model_on_random_inputs(model):
    """
    Purpose:
        Test a model with random inputs just to make sure it runs.
    """
    #Generate a random instance of model
    net = model()
    print(f"{net=}")

    #Generate 5 random board states
    inputs = [random_board() for _ in range(5)]
    #print(f"{inputs=}")
    inputs = torch.tensor(np.stack(inputs))

    #Run the model
    outputs = net(inputs)
    #print(f"{outputs=}")
    
    for i,o in zip(inputs, outputs):
        print(f"{i} mapped to {o}")

run_model_on_random_inputs(base_single_layer_model)
