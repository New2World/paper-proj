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
adj_matrix = data.dot(data.T)
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = OurModel(data.shape[1], 300, 300, np.zeros(adj_matrix.shape))

gcn.load_state_dict(torch.load("../checkpoints/model.pt"))
gcn.eval()
learned_adj = gcn.gcn.gc1.adj_matrix.detach().numpy()
output = gcn(torch.from_numpy(data).type(torch.FloatTensor)).detach().numpy()
partial_learned = learned_adj[:100,:100]
# print(partial_learned)

# np.save("prediction.npy", output)
# np.save("adjacent.npy", learned_adj)

# output = np.load("output.npy")

mae = np.mean(np.abs(popularity-output))
print(f"MAE loss: {mae}")
# plt.xlabel('post')
# plt.ylabel('popularity')
# x = np.arange(100)
# plt.scatter(x, popularity[:100], c='r', marker='x', label='ground truth')
# plt.scatter(x, output[:100], c='b', marker='o', label='prediction')
# plt.legend()
matplotlib.rcParams['font.family'] = 'Times New Roman'
# sb.heatmap(adj_matrix[:100,:100], cmap="Blues", xticklabels=10, yticklabels=10)
# plt.title("initial adjacent matrix")
# plt.xlabel("post ID")
# plt.ylabel("post ID")
# plt.savefig("initial_adjacent_matrix.pdf", format='pdf')
sb.heatmap(partial_learned, cmap="Blues", xticklabels=10, yticklabels=10)
plt.title("learned adjacent matrix")
plt.xlabel("post ID")
plt.ylabel("post ID")
plt.savefig("learned_adjacent_matrix.pdf", format='pdf')
plt.show()