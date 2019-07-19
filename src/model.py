import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layer import GraphConvolutionLayer

class GCN(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(in_features, 1024, np.copy(adj_matrix))
        self.gc2 = GraphConvolutionLayer(1024, 512, np.copy(adj_matrix))
        self.gc3 = GraphConvolutionLayer(512, out_features, np.copy(adj_matrix))
        # self.dense = nn.Linear(out_features, 1)
    
    def expectation(self):
        self.gc1.expectation()
        self.gc2.expectation()
        self.gc3.expectation()
        # for param in self.dense.parameters():
        #     param.requires_grad_(False)
    
    def maximization(self):
        self.gc1.maximization()
        self.gc2.maximization()
        self.gc3.maximization()
        # for param in self.dense.parameters():
        #     param.requires_grad_()
    
    def forward(self, x):
        x = torch.tanh(self.gc1(x))
        x = torch.tanh(self.gc2(x))
        x = self.gc3(x)
        # x = F.dropout(x, p=.5)
        # x = self.dense(x)
        x = torch.mm(x, x.t())
        x = x / torch.max(x)
        return x