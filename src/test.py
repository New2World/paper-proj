import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import torch

from model import OurModel

raw_data = np.load("../data/one-hot-encoding.npz")
data = raw_data["encoding"]
popularity = raw_data["popularity"]
n_threads = data.shape[0]
adj_matrix = data @ data.T

gcn = OurModel(data.shape[1], 300, 300, np.eye(n_threads))
gcn.load_state_dict(torch.load("../checkpoints/model.pt"))
learned_adj = gcn.gcn.gc1.adj_matrix.detach().numpy()
partial_learned = learned_adj[:100,:100]

output = np.load("../res/prediction.npy")
mae = np.mean(np.abs(popularity-output))
print(f"MAE loss: {mae}")
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
plt.xlabel('thread ID')
plt.ylabel('popularity')
x = np.arange(100)
plt.scatter(x, popularity[:100], c='r', marker='x', label='ground truth')
plt.scatter(x, output[:100], c='b', marker='o', label='prediction')
plt.legend()
plt.savefig("../res/popularity_prediction_100.pdf", format='pdf')
plt.clf()
sb.heatmap(adj_matrix[:100,:100], cmap="Blues", xticklabels=10, yticklabels=10)
plt.title("initial adjacent matrix")
plt.xlabel("thread ID")
plt.ylabel("thread ID")
plt.savefig("../res/initial_adjacent_matrix.pdf", format='pdf')
plt.clf()
sb.heatmap(partial_learned, cmap="Blues", xticklabels=10, yticklabels=10)
plt.title("learned adjacent matrix")
plt.xlabel("thread ID")
plt.ylabel("thread ID")
plt.savefig("../res/learned_adjacent_matrix.pdf", format='pdf')