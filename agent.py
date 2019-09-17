from value_network import Value_Network
from action_network import Action_Network
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from memory import Memory, Transition
from utils import soft_update


class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        assert isinstance(state_space, int) and isinstance(action_space, int)

        self.action_space = action_space

        self.memory = Memory(5000)

        self.value_network = Value_Network(state_space + action_space, 1)
        self.value_target_network = Value_Network(state_space + action_space, 1)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

        self.action_network = Action_Network(state_space, action_space)
        self.action_target_network = Action_Network(state_space, action_space)
        self.action_optimizer = optim.Adam(self.action_network.parameters(), lr=0.001)

        # init network parameters
        for f in self.value_network.parameters():
            f.data.zero_()

        for f in self.action_network.parameters():
            f.data.zero_()

        # hard copy learning network's params to target network's params
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

        self.action_target_network.load_state_dict(self.action_network.state_dict())
        self.action_target_network.eval()

        self.criterion = nn.MSELoss()

        # Hyper-parameters
        self.BATCH_SIZE = 128
        self.DISCOUNT = 0

    def act(self, state):

        with torch.no_grad():

            return self.action_network(state) + 3 * torch.randn(1, 2)

    def update(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        # get training batch

        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)

        action_batch = torch.cat(batch.action)

        reward_batch = torch.cat(batch.reward).unsqueeze(1)

        next_state = torch.cat(batch.next_state)

        # update value network

        state_action = torch.cat((state_batch, action_batch), dim=1)
        state_action_value = self.value_network(state_action)

        next_action = self.action_target_network(next_state).detach()

        next_state_action = torch.cat((next_state, next_action), dim=1)
        next_state_action_value = self.value_target_network(next_state_action).detach()

        expected_state_action_value = (self.DISCOUNT * next_state_action_value) + reward_batch

        value_loss = self.criterion(state_action_value, expected_state_action_value)

        self.value_optimizer.zero_grad()

        value_loss.backward()
        self.value_optimizer.step()

        # update action network

        optim_action = self.action_network(state_batch)

        optim_state_action = torch.cat((state_batch, optim_action), dim=1)

        action_loss = -self.value_network(optim_state_action)
        action_loss = action_loss.mean()

        self.action_optimizer.zero_grad()

        action_loss.backward()
        self.action_optimizer.step()

        # update target network
        soft_update(self.value_target_network, self.value_network, 0.01)
        soft_update(self.action_target_network, self.action_network, 0.01)



