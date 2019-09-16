import torch
import torch.nn as nn
import torch.nn.functional as F

class Action_Network(nn.Module):

    def __init__(self, inputs, outputs):
        super(Action_Network, self).__init__()

        self.fc1 = nn.Linear(inputs, 2 * inputs)

        self.fc2 = nn.Linear(2 * inputs, 2 * outputs)

        self.fc3 = nn.Linear(2 * outputs, outputs)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x