import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GraphConvolutionLayer, self).__init__()
        self.adj_matrix = nn.Parameter(adj_matrix)
        self.w = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.uniform_(self.w)
        nn.init.uniform_(self.bias)
        
    def expectation(self):
        self.adj_matrix.requires_grad_()
        self.w.requires_grad_(False)
        self.bias.requires_grad_(False)
    
    def maximization(self):
        self.adj_matrix.requires_grad_(False)
        self.w.requires_grad_()
        self.bias.requires_grad_()

    def forward(self, x):
        x = torch.mm(x, self.w)
        x = torch.mm(self.adj_matrix, x)
        x = x + self.bias
        return x

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_classes, adj_matrix):
        super(GCN, self).__init__()
        self.q = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.gc1 = GraphConvolutionLayer(in_features, hidden_features, adj_matrix)
        self.gc2 = GraphConvolutionLayer(hidden_features, out_features, adj_matrix)
        self.fc1 = nn.Linear(out_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        nn.init.normal_(self.q)
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc3.weight)
        
    def expectation(self):
        self.gc1.expectation()
        self.gc2.expectation()
        self.q.requires_grad_()
        self.fc1.weight.requires_grad_(False)
        self.fc2.weight.requires_grad_(False)
        self.fc3.weight.requires_grad_(False)
        self.fc1.bias.requires_grad_(False)
        self.fc2.bias.requires_grad_(False)
        self.fc3.bias.requires_grad_(False)
        
    def maximization(self):
        self.gc1.maximization()
        self.gc2.maximization()
        self.q.requires_grad_(False)
        self.fc1.weight.requires_grad_()
        self.fc2.weight.requires_grad_()
        self.fc3.weight.requires_grad_()
        self.fc1.bias.requires_grad_()
        self.fc2.bias.requires_grad_()
        self.fc3.bias.requires_grad_()
        
    def forward(self, x):
        x = self.gc1(x)
        x = F.relu(x)
        x = self.gc2(x)
        h = F.relu(x)
        x = self.fc1(h)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        m = torch.mm(h, self.q)
        m = torch.mm(m, h.t())
        return x, m