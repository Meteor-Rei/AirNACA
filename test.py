


import numpy as np


a = np.zeros((8,6))

import torch


a = torch.tensor((2,3))
print(a)
print(a.shape)
a = a.unsqueeze(dim = 0)
print(a)
print(a.shape)

a = a.expand(5, -1)
print(a.shape)
print(a)