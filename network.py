"""
Purpose:
    Standardizes the model reference
"""

import torch
seed = 49
torch.manual_seed(seed)
print(f"Setting manual seed as {seed}")


from base_single_layer_model import base_single_layer_model as model



