import torch

# Our module!
from ._C import d_sigmoid

__all__ = ['d_sigmoid', 'DSigmoid']


class DSigmoid(torch.nn.Module):
    def forward(self, x):
        return d_sigmoid(x)
