import torch
from torch import nn

class Synapse:
    def __init__(self, shape):
        self.weight = nn.Parameter(torch.zeros(shape))

    def forward(self, z):
        z = z * self.weight.data

        return z

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
