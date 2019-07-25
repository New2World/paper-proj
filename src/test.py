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

gcn = OurModel(data.shape[1], 300, 300, np.zeros(adj_matrix.shape))

gcn.load_state_dict(torch.load("../checkpoints/model.pt"))
gcn.eval()
learned_adj = gcn.gcn.gc1.adj_matrix.detach().numpy()
output = gcn(torch.from_numpy(data).type(torch.FloatTensor)).detach().numpy()
# partial_learned = learned_adj[:100,:100]
# print(partial_learned)

# output = np.load("output.npy")

mse = np.mean((popularity-output)**2)
print(f"MSE loss: {mse}")
plt.xlabel('post')
plt.ylabel('popularity')
plt.plot(popularity, 'rx--', label='ground truth')
plt.plot(output, 'bo-', label='prediction')
plt.legend()
plt.show()