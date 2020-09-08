import unittest
import torch
import torch.nn as nn
import cuda_ext


class Tester(unittest.TestCase):
    def test_d_sigmoid(self):

        mod = cuda_ext.DSigmoid()
        # Type verification
        self.assertIsInstance(mod, nn.Module)

        num_batches = 2
        num_classes = 4
        x = torch.rand(num_batches, num_classes, 20, 20)

        # Value check
        self.assertTrue(torch.allclose(mod(x), mod(x.cuda()).cpu()))


if __name__ == '__main__':
    unittest.main()
