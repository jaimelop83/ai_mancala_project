"""
Purpose:
    A transformer player. Might try tweaking someof the mutable parameters.
    
Internal layers:
    setup, a single transformer later, and postprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mancala_model import mancala_model

class updated_transformer_model(mancala_model):
    model_name = "Transformer"

    #Fixed: num_inputs, num_actions, vocab_size
    #Mutable: d_model, nhead, num_layers
    def __init__(self, num_inputs=14, num_actions=6, vocab_size=49, d_model=32, nhead=4, num_layers=1):
        super(updated_transformer_model, self).__init__()
        self.fitness = 0
        self.special = False
        self.eval()
        self.wins = 0

        # Embeds the board into the transformer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Information is ordinal, I think this tells the transformer that.
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_inputs, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True  # (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Postprocessing to put it back into a usable action
        self.fc_out = nn.Linear(num_inputs * d_model, num_actions)

    def forward(self, x):
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0,:6]]) 
        x = x.int()
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        logits = self.fc_out(x)
        # return torch.argmax(logits, dim=1)
    
        masked_x = logits.clone()
        masked_x[0,legal_mask == 0] = -float('inf')
        move = torch.argmax(masked_x, dim=1)
        return move
    
