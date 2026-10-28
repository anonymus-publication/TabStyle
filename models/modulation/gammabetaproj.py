import torch
import torch.nn as nn

class GammaBetaGenerator(nn.Module):
    def __init__(self, tab_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(tab_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:num_channels])   # gamma = 1
        nn.init.zeros_(self.proj.bias[num_channels:])  # beta = 0

    def forward(self, t):
        gamma_beta = self.proj(t)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta