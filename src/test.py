import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import torch

from model import OurModel

data = np.load("../data/first-5-reply-matrix.npy")
popularity = np.load("../data/popularity.npy")
n_threads = data.shape[0]
adj_matrix = data @ data.T
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = OurModel(data.shape[1], 300, 300, 300, np.eye(n_threads))
gcn.load_state_dict(torch.load("../checkpoints/model_ep10.pt"))
gcn.eval().cuda()
output = gcn(torch.from_numpy(data).type(torch.FloatTensor).cuda()).detach().cpu().numpy().squeeze()
learned_adj = gcn.gcn.gc1.adj_matrix.detach().cpu().numpy()
partial_learned = learned_adj[:100,:100]
# partial_learned[partial_learned<.0001] = partial_learned[partial_learned<.0001]*10

# output = np.load("../res/prediction.npy")
mae = np.mean(np.abs(popularity[:100]-output[:100]))
print(f"MAE loss: {mae}")
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
plt.xlabel('thread ID')
plt.ylabel('popularity')
x = np.arange(100)
plt.scatter(x, popularity[:100], c='r', marker='x', label='ground truth')
plt.scatter(x, output[:100], c='b', marker='o', label='prediction')
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