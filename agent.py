from value_network import Value_Network
from action_network import Action_Network
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        assert isinstance(state_space, int) and isinstance(action_space, int)

        self.action_space = action_space

        self.value_network = Value_Network(state_space + action_space, 1)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

        self.action_network = Action_Network(state_space, action_space)
        self.action_optimizer = optim.Adam(self.action_network.parameters(), lr=0.001)

        self.criterion = nn.MSELoss()

    def act(self, state):
        state = state.unsqueeze(0)

        with torch.no_grad():

            return self.action_network(state).squeeze(0)

    def update(self, state, next_state, action, reward):

        # update value network

        state_action = torch.cat([state, action]).unsqueeze(0)
        state_action_value = self.value_network(state_action)

        next_action = self.action_network(next_state).detach()
        next_action = next_action.squeeze(0)

        next_state_action = torch.cat([next_state, next_action]).unsqueeze(0)
        next_state_action_value = self.value_network(next_state_action).detach()

        expected_state_action_value = (0.9 * next_state_action_value) + reward

        value_loss = self.criterion(state_action_value, expected_state_action_value)

        self.value_optimizer.zero_grad()

        value_loss.backward()
        self.value_optimizer.step()

        # update action network

        optim_action = self.action_network(state.unsqueeze(0)).detach()
        optim_action = optim_action.squeeze(0)

        optim_state_action = torch.cat([state, optim_action]).unsqueeze(0)
        action_loss = -self.value_network(optim_state_action)

        self.action_optimizer.zero_grad()

        action_loss.backward()
        self.action_optimizer.step()




