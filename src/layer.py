import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GraphConvolutionLayer, self).__init__()
        self.adj_matrix = nn.Parameter(torch.from_numpy(adj_matrix).type(torch.sparse.FloatTensor))
        self.register_parameter("adj_matrix", self.adj_matrix)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias)
        self.maximization()
    
    def expectation(self):
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)
        self.adj_matrix.requires_grad_()
    
    def maximization(self):
        self.weight.requires_grad_()
        self.bias.requires_grad_()
        self.adj_matrix.requires_grad_(False)

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = torch.spmm(self.adj_matrix, x)
        x = x + self.bias
        return x