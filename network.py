"""
Purpose:
    Currently just a hello-world neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#Neural Netork with 1 hidden layer (2->3->2)
class ExampleNN(nn.Module):
    def __init__(self):
        super(ExampleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3) 
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Generate a random instance
torch.manual_seed(42)
net = ExampleNN()
print(f"{net=}")

#Run it on 10 random input vectors
inputs = torch.randn(10, 2)
outputs = net(inputs)
for i,o in zip(inputs, outputs):
    print(f"{[round(x,2) for x in i.tolist()]} mapped to {[round(x,2) for x in o.tolist()]}")