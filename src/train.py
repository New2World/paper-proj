import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import OurModel

def DataSet(Dataset):
    def __init__(self, validation=.2):
        self.usr_onehot = np.load("../data/onehot-encoding.npy")
        self.hotreply = np.load("../data/first-5-reply-matrix.npy")
        self.popularity = np.load("../data/popularity.npy")
        self.length = 

raw_data = np.load("../data/one-hot-encoding.npz")
data = raw_data["encoding"]
popularity = torch.from_numpy(raw_data["popularity"][:,np.newaxis]).type(torch.FloatTensor)
adj_matrix = data.dot(data.T)
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = OurModel(data.shape[1], 300, 300, adj_matrix)

e_opt = optim.Adam(gcn.parameters(), lr=1e-4)
m_opt = optim.Adam(gcn.parameters(), lr=1e-3)
e_schedule = optim.lr_scheduler.StepLR(e_opt, 3)
m_schedule = optim.lr_scheduler.StepLR(m_opt, 3)
criteria = nn.MSELoss()

def train_block(loop, optimizer):
    y = gcn(data)
    loss = criteria(popularity, y)
    print(f"    #{loop+1:3d} - loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Start training...")
data = torch.from_numpy(data).type(torch.FloatTensor)
for epoch in range(10):
    print(f"Epoch #{epoch+1:3d}")
    # maximization
    print("  Maximization")
    gcn.maximization()
    for i in range(20):
        train_block(i, m_opt)

    # expectation
    print("  Expectation")
    gcn.expectation()
    for i in range(50):
        train_block(i, e_opt)
    
    e_schedule.step()
    m_schedule.step()

    torch.save(gcn.state_dict(), f"../checkpoints/model_ep{epoch+1}.pt")

output = gcn(data)
np.save("output.npy", output.detach().numpy())
print("output saved")
torch.save(gcn.state_dict(), "../checkpoints/model.pt")
print("model saved")