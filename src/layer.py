import torch
import torch.nn as nn
import torch.optim as optim

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GraphConvolutionLayer, self).__init__()
        self.adj_matrix = torch.from_numpy(adj_matrix)
        self.weight = torch.FloatTensor(in_features, out_features)
        self.bias = torch.FloatTensor(out_features)
    
    def forward(self, x):
        pass