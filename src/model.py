import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import layer

class GCN(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCN, self).__init__()
        self.gc1 = layer.GraphConvolutionLayer(in_features, out_features, adj_matrix)
        # self.gc2 = layer.GraphConvolutionLayer(1024, 512, adj_matrix)
        # self.gc3 = layer.GraphConvolutionLayer(512, out_features, adj_matrix)
        self.dense = nn.Linear(out_features, 1)
    
    def expectation(self):
        self.gc1.expectation()
        # self.gc2.expectation()
        # self.gc3.expectation()
        for param in self.dense.parameters():
            param.requires_grad_(False)
    
    def maximization(self):
        self.gc1.maximization()
        # self.gc2.maximization()
        # self.gc3.maximization()
        for param in self.dense.parameters():
            param.requires_grad_()
    
    def forward(self, x):
        x = F.relu(self.gc1(x))
        # x = F.relu(self.gc2(x))
        # x = F.relu(self.gc3(x))
        # x = F.dropout(x, p=.5)
        x = self.dense(x)
        return x