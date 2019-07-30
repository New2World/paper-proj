import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import OurModel

class DataSet(Dataset):
    def __init__(self, validation=.2):
        self.allreply = torch.from_numpy(np.load("../data/onehot-encoding.npy")).type(torch.FloatTensor)
        self.hotreply = torch.from_numpy(np.load("../data/first-5-reply-matrix.npy")).type(torch.FloatTensor)
        self.popularity = np.load("../data/popularity.npy")
        self.context = np.load("../data/context-matrix.npy")
        self.length = self.hotreply.size(0)
        self.validation = 1-validation
    
    def __len__(self):
        return self.length * self.validation
    
    def __getitem__(self, id):
        context = self.context[id]
        if context.shape[0] < 200:
            context = np.stack((context, np.zeros((200-context.shape[0],69))), axis=0)
        elif context.shape[0] > 200:
            context = context[:200]
        return {
            'hot-reply': self.hotreply[id].cuda(),
            'all-reply': self.allreply[id].cuda(),
            'context': torch.from_numpy(context).type(torch.FloatTensor).cuda(),
            'popularity': self.popularity[id].cuda()
        }

    def get_data_dimension(self):
        return self.allreply.size(0)

    def get_adjacent_matrix(self):
        return torch.spmm(self.allreply, self.allreply.t())

dataset = DataSet()
loader = DataLoader(dataset, batch_size=512)
adj_matrix = dataset.get_adjacent_matrix()

model = OurModel(dataset.get_data_dimension(), 200, 300, 300, adj_matrix)

e_opt = optim.Adam(model.parameters(), lr=2e-4)
m_opt = optim.Adam([{'params': model.gcn.parameters(), 'lr':1e-3},
                    {'params': model.dense.parameters(), 'lr':1e-3}])
e_schedule = optim.lr_scheduler.StepLR(e_opt, 20)
m_schedule = optim.lr_scheduler.StepLR(m_opt, 20)
criteria = nn.MSELoss()

# CUDA
model.cuda()
criteria.cuda()

def train_block(loop, optimizer):
    avg_loss = 0.
    batch = 0
    for sample in loader:
        onehot = sample['hot-reply']
        context = sample['context']
        popularity = sample['popularity']
        y = model(onehot, context)
        loss = criteria(popularity, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
        batch += 1
    print(f"    #{loop+1:3d} - loss: {avg_loss/batch}")

print("Start training...")
for epoch in range(50):
    print(f"Epoch #{epoch+1:3d}")
    # maximization
    print("  Maximization")
    model.maximization()
    for i in range(20):
        train_block(i, m_opt)

    # expectation
    print("  Expectation")
    model.expectation()
    for i in range(50):
        train_block(i, e_opt)
    
    e_schedule.step()
    m_schedule.step()

    torch.save(model.state_dict(), f"../checkpoints/model_ep{epoch+1}.pt")

# output = model(data)
# np.save("output.npy", output.detach().numpy())
# print("output saved")
# torch.save(model.state_dict(), "../checkpoints/model.pt")
# print("model saved")