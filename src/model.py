import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class GCN(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(in_features, 1024, np.copy(adj_matrix))
        self.gc2 = GraphConvolutionLayer(1024, 512, np.copy(adj_matrix))
        self.gc3 = GraphConvolutionLayer(512, out_features, np.copy(adj_matrix))
    
    def expectation(self):
        self.gc1.expectation()
        self.gc2.expectation()
        self.gc3.expectation()
    
    def maximization(self):
        self.gc1.maximization()
        self.gc2.maximization()
        self.gc3.maximization()
    
    def forward(self, x):
        x = torch.tanh(self.gc1(x))
        x = torch.tanh(self.gc2(x))
        x = self.gc3(x)
        return x

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 69, kernel_size=7)
        self.conv2 = nn.Conv1d(69, 256, kernel_size=7)
        self.mid_layers = nn.Sequential()
        for i in range(4):
            self.mid_layers.add_module(
                f'mid_conv{i+1}', 
                nn.Conv1d(256, 256, kernel_size=3)
            )
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        for m in self.mid_layers.modules():
            if type(m) is nn.Conv1d:
                nn.init.kaiming_normal_(m.weight)
    
    def expectation(self):
        for p in self.conv1.parameters():
            p.requires_grad_(False)
        for p in self.conv2.parameters():
            p.requires_grad_(False)
        for p in self.mid_layers.parameters():
            p.requires_grad_(False)
    
    def maximization(self):
        for p in self.conv1.parameters():
            p.requires_grad_()
        for p in self.conv2.parameters():
            p.requires_grad_()
        for p in self.mid_layers.parameters():
            p.requires_grad_()

    def forward(self, x):
        x = self.conv1(x)
        x = F._max_pool1d(x, kernel_size=3)
        x = self.conv2(x)
        x = F._max_pool1d(x, kernel_size=3)
        x = self.mid_layers(x)
        x = F._max_pool1d(x, kernel_size=7)
        return x

class Dense(nn.Module):
    def __init__(self, in_features):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(in_features, 152)
        self.fc2 = nn.Linear(152, 48)
        self.fc3 = nn.Linear(48, 1)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
    
    def maximization(self):
        for param in self.fc1.parameters():
            param.requires_grad_()
        for param in self.fc2.parameters():
            param.requires_grad_()
        for param in self.fc3.parameters():
            param.requires_grad_()
    
    def expectation(self):
        for param in self.fc1.parameters():
            param.requires_grad_(False)
        for param in self.fc2.parameters():
            param.requires_grad_(False)
        for param in self.fc3.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class OurModel(nn.Module):
    def __init__(self, post_features, context_features, post_embedding, context_embedding, adj_matrix):
        super(OurModel, self).__init__()
        self.gcn = GCN(post_features, post_embedding, adj_matrix)
        self.txcnn = TextCNN(context_features, context_embedding)
        # self.dense = Dense(post_embedding+context_embedding)
        self.dense = Dense(post_embedding)
    
    def maximization(self):
        self.gcn.maximization()
        self.txcnn.maximization()
        self.dense.maximization()
    
    def expectation(self):
        self.gcn.expectation()
        self.txcnn.expectation()
        self.dense.expectation()

    def forward(self, x):
        g = self.gcn(x)
        d = g
        # t = self.txcnn(x)
        # d = torch.cat((g,t), 0)
        d = F.dropout(d)
        d = self.dense(d)
        return d