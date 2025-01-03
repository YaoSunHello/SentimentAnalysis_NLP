from turtle import width
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_data import text_process

from matplotlib.pyplot import text
import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
from torch import dropout, optim

from load_data import text_process
import math 


class Model(nn.Module):
    def __init__(self, dict_size,embed_size,num_class,hidden_size,num_layer,word_length):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = dict_size ,embedding_dim= embed_size,\
            sparse = True,padding_idx= dict_size-1) #len dictionary, embedding vector length, pad words
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layer,\
            batch_first = True, dropout = 0.8)
        self.fc = nn.Linear((embed_size+ hidden_size)* word_length ,num_class)
        self.soft_max = nn.Softmax(dim=1)



    def forward(self, x):
        embed = self.embedding(x) #[batchsize * sqlength * embedsize]
        out , _ = self.lstm(embed)
        out = torch.cat((embed, out) , 2)
        out = F.relu((out)).reshape(out.size()[0],-1)
        out = self.fc(out)
        out = self.soft_max(out)
        return out

'''
dict_size,embed_size,num_class,hidden_size,num_layer,word_length = 150,23,2,250,3,199

model = Model(dict_size,embed_size,num_class,hidden_size,num_layer,word_length)

input = torch.tensor([0]*128*199).reshape(128,199)

output = model.forward(input)

print(output)'''