import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, inputs, outputs):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(inputs, 2 * inputs)
        self.fc2 = nn.Linear(2 * inputs, 2 * outputs)
        self.fc3 = nn.Linear(2 * outputs, outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return self.fc3(x)



if __name__ == '__main__':
    net = Network(3, 4)
    params = list(net.parameters())

    for param in params:
        print(param)

