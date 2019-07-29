import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import OurModel

data = np.load("../data/first-5-reply-matrix.npy")
popularity = np.load("../data/popularity.npy")[:,np.newaxis]
adj_matrix = data @ data.T
adj_matrix = adj_matrix / np.max(adj_matrix)
data = torch.from_numpy(data).type(torch.FloatTensor)
adj_matrix = torch.from_numpy(adj_matrix).type(torch.FloatTensor)
popularity = torch.from_numpy(popularity).type(torch.FloatTensor)

gcn = OurModel(data.shape[1], 300, 300, 300, adj_matrix)

e_opt = optim.Adam(gcn.parameters(), lr=2e-4)
m_opt = optim.Adam(gcn.parameters(), lr=1e-3)
e_schedule = optim.lr_scheduler.StepLR(e_opt, 10)
m_schedule = optim.lr_scheduler.StepLR(m_opt, 10)
criteria = nn.MSELoss()

# CUDA
data = data.cuda()
adj_matrix = adj_matrix.cuda()
popularity = popularity.cuda()
gcn.cuda()
criteria.cuda()

def train_block(loop, optimizer):
    y = gcn(data)[100:]
    loss = criteria(popularity[100:], y)
    print(f"    #{loop+1:3d} - loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Start training...")
for epoch in range(50):
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

# output = gcn(data)
# np.save("output.npy", output.detach().numpy())
# print("output saved")
# torch.save(gcn.state_dict(), "../checkpoints/model.pt")
# print("model saved")