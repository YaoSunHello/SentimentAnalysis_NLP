from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch




m = nn.MaxPool1d(3)
input = torch.randn(20, 232, 150)
output = m(input)
print(output.shape)
