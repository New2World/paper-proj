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

class Dense(nn.Module):
    def __init__(self, in_features):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(in_features, 152)
        self.fc2 = nn.Linear(152, 48)
        self.fc3 = nn.Linear(48, 1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class OurModel(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(OurModel, self).__init__()
        self.gcn = GCN(in_features, out_features, adj_matrix)
        self.dense = Dense(out_features)