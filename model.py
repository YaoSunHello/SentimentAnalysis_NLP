import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_data import text_process

from matplotlib.pyplot import text
import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

from load_data import text_process


class Model(nn.Module):
    def __init__(self, dict_size,embed_size,num_class):
        super(Model, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings = dict_size ,embedding_dim= embed_size,\
            sparse = True) #len dictionary, embedding vector length, pad words
        self.fc = nn.Linear(embed_size, num_class)
        #self.soft_max = nn.Softmax(dim = 1)
        self.init_weight()

    def init_weight(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange,initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        embed = self.embedding(x) 
        out = self.fc(embed)
        return out


