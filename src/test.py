import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import torch
from torch.utils.data import DataLoader, Dataset

from model import OurModel

# class DataSet(Dataset):
#     def __init__(self, validation=.2):
#         self.allreply = torch.from_numpy(np.load("../data/onehot_encoding.npy")).type(torch.FloatTensor)
#         self.hotreply = torch.from_numpy(np.load("../data/first_5_reply_matrix.npy")).type(torch.FloatTensor)
#         self.popularity = np.load("../data/popularity.npy")
#         self.context = np.load("../data/context_matrix.npy", allow_pickle=True)
#         self.length = self.hotreply.size(0)
#         self.validation = 1-validation
#         self.fixlen = 204
    
#     def __len__(self):
#         return int(self.length * self.validation)
    
#     def __getitem__(self, idx):
#         context = self.context[idx]
#         if context.shape[0] == 0:
#             context = np.zeros((self.fixlen,69))
#         elif context.shape[0] < self.fixlen:
#             context = np.vstack((context, np.zeros((self.fixlen-context.shape[0],69))))
#         elif context.shape[0] > self.fixlen:
#             context = context[:self.fixlen]
#         return {
#             'index': idx,
#             'hotreply': self.hotreply[idx].cuda(),
#             'allreply': self.allreply[idx].cuda(),
#             'context': torch.from_numpy(context.T).type(torch.FloatTensor).cuda(),
#             'popularity': self.popularity[idx]
#         }
    
#     def get_data_dimension(self):
#         return self.allreply.size()

#     def get_adjacent_matrix(self):
#         return torch.spmm(self.allreply, self.allreply.t())

# dataset = DataSet()
# loader = DataLoader(dataset, batch_size=512)
# adj_matrix = dataset.get_adjacent_matrix().type(torch.FloatTensor)
# n_post, n_user = dataset.get_data_dimension()

data = np.load("../data/first_5_reply_matrix.npy")
popularity = np.load("../data/popularity.npy")
n_threads = data.shape[0]
adj_matrix = data @ data.T
adj_matrix = adj_matrix / np.max(adj_matrix)
st = int(data.shape[0]*.8)

model = OurModel(data.shape[1], 69, 300, 300, np.eye(n_threads))
model.load_state_dict(torch.load("../checkpoints/model_ep50.pt"))
model.eval().cuda()
post_inp = torch.from_numpy(data).type(torch.FloatTensor).cuda()[st:]
allcontext = np.load("../data/context_matrix.npy", allow_pickle=True)
context_inp = np.zeros((n_threads-st, 69, 204))
for i, th in enumerate(allcontext[st:]):
    if th.shape[0] == 0:
        context_inp[i] = np.zeros((69, 204))
    elif th.shape[0] < 204:
        context_inp[i] = np.vstack((th, np.zeros((204-th.shape[0], 69)))).T
    else:
        context_inp[i] = th[:204].T
context_inp = torch.from_numpy(context_inp).type(torch.FloatTensor).cuda()
output = np.rint(model(post_inp, context_inp, range(st, data.shape[0])).detach().cpu().numpy().squeeze())
learned_adj = model.gcn.gc1.adj_matrix.detach().cpu().numpy()
partial_learned = learned_adj[:100,:100]
# partial_learned[partial_learned<.0001] = partial_learned[partial_learned<.0001]*10

# output = np.load("../res/prediction.npy")
prediction = output[:100]
popular = popularity[:100]
mae = np.mean(np.abs(popular-prediction))
max_diff = np.max(np.abs(popular-prediction))
min_diff = np.min(np.abs(popular-prediction))
print(f"MAE loss: {mae}")
print(f"min/max loss: {min_diff} / {max_diff}")
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
plt.xlabel('thread ID')
plt.ylabel('popularity')
x = np.arange(100)
plt.scatter(x, popular, c='r', marker='x', label='ground truth')
plt.scatter(x, prediction, c='b', marker='o', label='prediction')
plt.legend()
plt.savefig("../popularity_prediction_100.pdf", format='pdf')
plt.clf()
sb.heatmap(adj_matrix[:100,:100], cmap="Blues", xticklabels=10, yticklabels=10)
plt.title("initial adjacent matrix")
plt.xlabel("thread ID")
plt.ylabel("thread ID")
plt.savefig("../initial_adjacent_matrix.pdf", format='pdf')
plt.clf()
sb.heatmap(partial_learned, cmap="Blues", xticklabels=10, yticklabels=10)
plt.title("learned adjacent matrix")
plt.xlabel("thread ID")
plt.ylabel("thread ID")
plt.savefig("../learned_adjacent_matrix.pdf", format='pdf')