import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import OurModel

class DataSet(Dataset):
    def __init__(self, validation=.2):
        self.allreply = torch.from_numpy(np.load("../data/onehot_encoding.npy")).type(torch.FloatTensor)
        self.hotreply = torch.from_numpy(np.load("../data/first_5_reply_matrix.npy")).type(torch.FloatTensor)
        self.popularity = np.load("../data/popularity.npy")
        self.context = np.load("../data/context_matrix.npy", allow_pickle=True)
        self.length = self.hotreply.size(0)
        self.validation = 1-validation
        self.fixlen = 204
    
    def __len__(self):
        return int(self.length * self.validation)
    
    def __getitem__(self, idx):
        context = self.context[idx]
        if context.shape[0] == 0:
            context = np.zeros((self.fixlen,69))
        elif context.shape[0] < self.fixlen:
            context = np.vstack((context, np.zeros((self.fixlen-context.shape[0],69))))
        elif context.shape[0] > self.fixlen:
            context = context[:self.fixlen]
        return {
            'index': idx,
            # 'hotreply': self.hotreply[idx].cuda(),
            # 'allreply': self.allreply[idx].cuda(),
            'context': torch.from_numpy(context.T).type(torch.FloatTensor).cuda(),
            'popularity': self.popularity[idx]
        }
    
    def get_data_dimension(self):
        return self.allreply.size()

    def get_adjacent_matrix(self):
        adj_mat = torch.spmm(self.allreply, self.allreply.t())
        return adj_mat / torch.max(adj_mat)

dataset = DataSet()
loader = DataLoader(dataset, batch_size=512)
adj_matrix = dataset.get_adjacent_matrix().type(torch.FloatTensor)
n_post, n_user = dataset.get_data_dimension()

model = OurModel(n_user, 69, 300, 300, adj_matrix, dataset.hotreply.cuda())

criteria = nn.MSELoss()

# CUDA
model.cuda()
criteria.cuda()

def expectation_block(loop, optimizer):
    avg_loss = 0.
    batch = 0
    for sample in loader:
        y = model.gcn(model.posts)
        loss = criteria(model.gcn.gc1.adj_matrix, torch.mm(y,y.t()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
        batch += 1
    print(f"    #{loop+1:3d} - loss: {avg_loss/batch}")

def maximization_block(loop, optimizer):
    avg_loss = 0.
    batch = 0
    for sample in loader:
        indices = sample['index']
        context = sample['context']
        popularity = sample['popularity']
        y = model(context, indices)
        popularity = popularity.type(torch.FloatTensor).cuda()
        loss = criteria(popularity, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
        batch += 1
    print(f"    #{loop+1:3d} - loss: {avg_loss/batch}")

lr_e = 1e-4
lr_m = 1e-4
print("Start training...")
for epoch in range(5):
    print(f"Epoch #{epoch+1:3d}")
    # maximization
    print("  Maximization")
    model.maximization()
    optimizer = optim.Adam(model.parameters(), lr=lr_m)
    schedule = optim.lr_scheduler.StepLR(optimizer, 100, gamma=.1)
    # lr_m *= .9
    for i in range(200):
        maximization_block(i, optimizer)
        schedule.step()

    # expectation
    print("  Expectation")
    model.expectation()
    optimizer = optim.Adam(model.parameters(), lr=lr_e)
    schedule = optim.lr_scheduler.StepLR(optimizer, 100, gamma=.1)
    lr_e *= .9
    for i in range(200):
        expectation_block(i)
        schedule.step()
    
    torch.save(model.state_dict(), f"../checkpoints/model_ep{epoch+1}.pt")

# output = model(data)
# np.save("output.npy", output.detach().numpy())
# print("output saved")
# torch.save(model.state_dict(), "../checkpoints/model.pt")
# print("model saved")