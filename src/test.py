import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import torch

from model import GCN

data = np.load("../data/one-hot-encoding.npz")["encoding"]
adj_matrix = data.dot(data.T)
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = GCN(data.shape[1], 300, adj_matrix)

gcn.load_state_dict(torch.load("../checkpoints/model.pt"))
learned_adj = gcn.gc1.adj_matrix.detach().numpy()

partial_learned = learned_adj[:100,:100]

output_adj = np.load("output.npy")
partial_output = output_adj[:100,:100]

# plt.subplot(211)
# learned_heatmap = sb.heatmap(partial_learned)
# plt.subplot(212)
# output_heatmap = sb.heatmap(partial_output)
diff_heatmap = sb.heatmap(np.abs(partial_learned-partial_output))
plt.show()