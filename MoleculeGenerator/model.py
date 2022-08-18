# -*- coding: utf-8 -*-
import torch
from torch import nn


class MoleculeGenerator(nn.Module):
    def __init__(self, oneHotSize, hiddenDim = 1024, layers = 3, dropout = 0.2):
        super(MoleculeGenerator, self).__init__()
        self.hiddenDim = hiddenDim
        self.layers = layers
        self.dropout = dropout
        self.oneHotSize = oneHotSize

        self.model = nn.LSTM(input_size= oneHotSize, hidden_size= self.hiddenDim, num_layers= self.layers,
                             dropout = self.dropout)
        self.linear = nn.Linear(self.hiddenDim, self.oneHotSize)

    def init_hidden(self, batchSize):
        h = torch.randn(self.layers, batchSize,self. hiddenDim)
        c = torch.randn(self.layers, batchSize, self.hiddenDim)
        return (h,c)

    def forward(self, input, h_c):
        output, (hn, cn) = self.model.forward(input, h_c)

        output = self.linear(output)
        return output, (hn, cn)