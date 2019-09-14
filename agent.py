from network import Network
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

class Random(object):
    def __init__(self, action_space, state_dim):
        assert isinstance(action_space, list)

        self.action_space = action_space

        self.Q = Network(state_dim, len(action_space))
        self.optimizer = optim.SGD(self.Q.parameters(), lr=0.01)

    def act(self, state):
        sample = random.random()
        if sample > 0.3:
            with torch.no_grad():
                q_values = self.Q(state)

                return q_values.max(1)[1].view(1, 1)

        else:
            index = random.randrange(len(self.action_space))

            return torch.tensor([[index]], dtype=torch.long)

    def update(self, state, next_state, action_index, reward):

        state_action_value = self.Q(state).gather(1, action_index)
        expected_state_action_value =  ( 0.9 * self.Q(next_state).max(1)[0].detach()) + reward

        loss = F.smooth_l1_loss(state_action_value, expected_state_action_value.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()




