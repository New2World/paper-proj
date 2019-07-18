import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model

train_data = torch.rand((10,10))    # 10 nodes, each node has 10 features
train_label = torch.rand((10,1))

adj_matrix = np.eye(10)

gcn = model.GCN(10, 5, adj_matrix)

optimizer = optim.Adam(gcn.parameters(), lr=0.01)
criteria = nn.MSELoss()


# maximization
gcn.maximization()
y = gcn(train_data)
loss = criteria(train_label, y)
optimizer.zero_grad()
loss.backward()
for param in gcn.parameters():
    print(param.grad)
optimizer.step()

# print(list(gcn.parameters()))

# expectation
gcn.expectation()
y = gcn(train_data)
loss = criteria(train_label, y)
optimizer.zero_grad()
loss.backward()
for param in gcn.parameters():
    print(param.grad)
optimizer.step()

# print(list(gcn.parameters()))

# maximization
gcn.maximization()
y = gcn(train_data)
loss = criteria(train_label, y)
optimizer.zero_grad()
loss.backward()
for param in gcn.parameters():
    print(param.grad)
optimizer.step()

# print(list(gcn.parameters()))