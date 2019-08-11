import os
import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import seaborn as sb
import matplotlib.pyplot as plt

from model import GCN

def get_knn_graph(data, k):
    n_samples = data.size(0)
    A = torch.mm(data, data.t())
    A_sort = torch.argsort(A, dim=1, descending=True)
    mask = A.clone()
    mask[[[i]*k for i in range(n_samples)], A_sort[:,:k]] = 0
    graph = A-mask
    return graph# / torch.max(graph)

def split_dataset(data, label, valid_size=.2):
    n_samples = data.size(0)
    if isinstance(valid_size, int):
        sz = valid_size
    else:
        sz = int(n_samples*valid_size)
    label_indices = torch.argsort(label)
    span = n_samples // sz
    valid_id = label_indices[::span]
    train_id = list(set(range(n_samples))-set(valid_id))
    return train_id, valid_id

def find_ckpt(path, name):
    epo = 0
    ckpt_list = list(os.walk(path))[0][2]
    for ckpt_file in ckpt_list:
        if ckpt_file.startswith(name) and ckpt_file.endswith('.pt'):
            ep = int(ckpt_file[:-3].split('-')[1])
            if ep > epo:
                epo = ep
    return f'{name}-{epo}.pt'

def save_ckpt(path, name, model, optimizer, scheduler, epoch, step):
    torch.save({
        'epoch': epoch, 
        'step': step, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict(), 
    }, os.path.join(path, f'{name}-{epoch}.pt'))

def load_ckpt(path, name, model, optimizer=None, scheduler=None, epoch=None):
    if epoch is not None:
        ckpt_file = f'{name}-{epoch}.pt'
    else:
        ckpt_file = find_ckpt(path, name)
    state = torch.load(os.path.join(path, ckpt_file))
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state['epoch'], state['step']

def training_loop(model, data, label, train_indices, valid_indices, lr=1e-3, epoch=100, path='../checkpoints', name='model', resume=False):
    model.train().cuda()
    nll = nn.NLLLoss().cuda()
    mse = nn.MSELoss().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.Adam([
        {'params': model.q, 'lr': lr*.1}, 
        {'params': model.gc1.parameters(), 'lr': lr*.1},
        {'params': model.gc2.parameters(), 'lr': lr*.1},
        {'params': model.fc1.parameters(), 'lr': lr},
        {'params': model.fc2.parameters(), 'lr': lr},
        {'params': model.fc3.parameters(), 'lr': lr},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100)
    lb = label[train_indices]
    epo = 0
    last_loss = 0.
    if resume:
        epo = load_ckpt(path, name, model, optimizer, scheduler)
    for ep in range(epo, epoch):
        avg_loss = 0.
        count = 0
        model.maximization()
        for i in range(50):
            y = model(data)[0][train_indices]
            print(y)
            loss = nll(y, lb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss
            count = torch.sum(torch.argmax(y, dim=1) != lb)
        scheduler.step()
        print(f'Ep.{ep+1} - loss: {avg_loss/100.:.6f} - err: {count}')
        if avg_loss == last_loss:
            break
        last_loss = avg_loss
        if (ep+1) % 50 == 0:
            save_ckpt(path, name, model, optimizer, scheduler, ep+1, ep+1)
        if ep+1 == epoch:
            break
        model.expectation()
        for i in range(200):
            y, m = model(data)
            nll_loss = nll(y[train_indices], lb)
            adj = model.gc1.adj_matrix
            # print(m)
            # print(adj)
            mse_loss = mse(adj,m)
            loss = nll_loss + 1e-2 * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(model.q)
        # print(f'Ep.{ep+1} - loss: {avg_loss/100.:.6f} - mse loss: {avg_loss_mse/100.:.6f}')

data, label = datasets.load_digits(return_X_y=True)
data_max = np.max(data)
data = torch.from_numpy(data).type(torch.FloatTensor).cuda() / data_max
label = torch.from_numpy(label).type(torch.LongTensor).cuda()
train_id, valid_id = split_dataset(data, label)

clfr = GCN(64, 128, 128, 10, get_knn_graph(data, 8))
training_loop(clfr, data, label, train_id, valid_id, lr=2e-3, epoch=200, name='digits')

clfr.eval()
yy = torch.argmax(clfr(data)[0], dim=1)
train_acc = torch.sum(yy[train_id]==label[train_id]).cpu().numpy()*1./len(train_id)
valid_acc = torch.sum(yy[valid_id]==label[valid_id]).cpu().numpy()*1./len(valid_id)
print(f'training set accuracy: {train_acc:.6f}')
print(f'validation set accuracy: {valid_acc:.6f}')

init_matrix = get_knn_graph(data,8).cpu().numpy()
init_matrix = init_matrix / np.max(init_matrix)
adj_matrix = clfr.gc1.adj_matrix.detach()
adj_matrix = adj_matrix / torch.max(adj_matrix)
adj_matrix = adj_matrix.cpu().numpy()

plt.subplot(121)
sb.heatmap(init_matrix[:100,:100], cmap='Blues')

plt.subplot(122)
sb.heatmap(adj_matrix[:100,:100]-init_matrix[:100,:100], cmap='Blues')
plt.show()

# nll = nn.NLLLoss()
# mse = nn.MSELoss()
# clfr.maximization()
# optimizer = optim.Adam(clfr.parameters(), lr=1e-3)
# emb, y = clfr(data)
# nll_loss = nll(y[train_id], label[train_id])
# mse_loss = mse(clfr.adj_matrix, emb)
# loss = nll_loss
# optimizer.zero_grad()
# loss.backward()
# print(clfr.fc2.weight.grad)
# optimizer.step()