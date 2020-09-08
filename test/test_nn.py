import unittest
import torch
from torch.nn import Module
from cuda_ext import nn


class NNTester(unittest.TestCase):
    def test_d_sigmoid(self):

        mod = nn.DSigmoid()
        # Type verification
        self.assertIsInstance(mod, Module)

        num_batches = 2
        num_classes = 4
        x = torch.rand(num_batches, num_classes, 20, 20)

        # Value check
        self.assertTrue(torch.allclose(mod(x), mod(x.cuda()).cpu()))


if __name__ == '__main__':
    unittest.main()
