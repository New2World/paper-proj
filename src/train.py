import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import GCN

# train_data = torch.rand((10,10))    # 10 nodes, each node has 10 features
# train_label = torch.rand((10,1))
# adj_matrix = np.eye(10)

raw_data = np.load("../data/one-hot-encoding.npz")
data = raw_data["encoding"]
popularity = raw_data["popularity"]
adj_matrix = data.dot(data.T)
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = GCN(data.shape[1], 300, adj_matrix)

optimizer = optim.Adam(gcn.parameters(), lr=0.001)
criteria = nn.MSELoss()

def train_block(loop):
    y = gcn(data)
    loss = criteria(gcn.gc1.adj_matrix, y)
    print(f"    #{loop+1:3d} - loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Start training...")
data = torch.from_numpy(data).type(torch.FloatTensor)
for epoch in range(50):
    print(f"Epoch #{epoch:3d}")
    # maximization
    print("  Maximization")
    gcn.maximization()
    for i in range(100):
        train_block(i)

    # expectation
    print("  Expectation")
    gcn.expectation()
    for i in range(100):
        train_block(i)

output = gcn(data)
np.save("output.npy", output.detach().numpy())
print("output saved")
torch.save(gcn.state_dict(), "../checkpoints/model.pt")
print("model saved")