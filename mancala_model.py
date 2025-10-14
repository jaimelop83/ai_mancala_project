"""
This is intended to be a superclass of all the models for any common funcctions
"""

import torch.nn as nn
import torch

class mancala_model(nn.Module):

    def make_move(self, board_state):
        """
        Purpose:
            A board state is a native python list. Convert it to torch.
            The return is also in torch, convert it back to a number.
        """
        board = torch.tensor(board_state).unsqueeze(0)
        out = self.forward(board)
        out = out.item()
        return out

#from base_single_layer_model import base_single_layer_model as model