import torch.nn as nn
import cuda_ext

__all__ = ['DSigmoid']


class DSigmoid(nn.Module):
    def forward(self, x):
        return cuda_ext.d_sigmoid(x)
