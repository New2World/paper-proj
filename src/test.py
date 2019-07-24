import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import torch

from model import OurModel

raw_data = np.load("../data/one-hot-encoding.npz")
data = raw_data["encoding"]
popularity = raw_data["popularity"]
adj_matrix = data.dot(data.T)
adj_matrix = adj_matrix / np.max(adj_matrix)

gcn = OurModel(data.shape[1], 300, 300, adj_matrix)

gcn.load_state_dict(torch.load("../checkpoints/model.pt"))
learned_adj = gcn.gcn.gc1.adj_matrix.detach().numpy()

partial_learned = learned_adj[:100,:100]

output = np.load("output.npy")

l = output.shape[0]
plt.plot(popularity, c='r', marker='x')
plt.plot(output, c='b', marker='o')
plt.show()