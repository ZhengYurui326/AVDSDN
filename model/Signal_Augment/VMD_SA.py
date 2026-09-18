import torch
import torch.nn as nn


class SignalAugmentNetwork(nn.Module):
    def __init__(self, kaiming=False):
        super(SignalAugmentNetwork, self).__init__()

        self.i_filters = nn.Parameter(torch.ones(5, 128))
        self.q_filters = nn.Parameter(torch.ones(5, 128))

        if kaiming:
            nn.init.kaiming_uniform_(self.i_filters, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.q_filters, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        I_signals = x[:, :, 0, :]
        Q_signals = x[:, :, 1, :]

        weighted_I = I_signals * self.i_filters.unsqueeze(0)
        weighted_Q = Q_signals * self.q_filters.unsqueeze(0)

        summed_I = weighted_I.sum(dim=1, keepdim=True)
        summed_Q = weighted_Q.sum(dim=1, keepdim=True)

        return torch.stack([summed_I, summed_Q], dim=2)
