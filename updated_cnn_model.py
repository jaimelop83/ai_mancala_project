import torch
import torch.nn as nn
import torch.nn.functional as F
from mancala_model import mancala_model

class updated_cnn_model(mancala_model):
    model_name = "updated_cnn"

    def __init__(self):
        super(updated_cnn_model, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(16 * 14, 32)
        self.fc2 = nn.Linear(32, 6)

        self.fitness = 0
        self.special = False
        self.wins = 0
        self.eval()

    def forward(self, x):
        legal_mask = x.new_tensor([1 if v > 0 else 0 for v in x[0, :6]])

        x = x.float().unsqueeze(1) 

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        masked_x = x.clone()
        masked_x[0, legal_mask == 0] = -float('inf')

        move = torch.argmax(masked_x, dim=1)
        return move
