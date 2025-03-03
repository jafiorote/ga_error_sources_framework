import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from scipy.stats import binom

fig = plt.figure(figsize=(10, 25))
gs = fig.add_gridspec(12, 2)  

ns = [0, 1, 2, 4, 8, 16]
system = "1BXR_AB"

row_positions = [0, 4, 8]

dist = []
for n_similar in np.arange(1004):
    dist.append(binom.pmf(n_similar, 1004, 0.20))

for idx, n in enumerate(ns):
    col = idx % 2                    
    row = row_positions[idx // 2]     

    data = np.load(f"rencounters/n{n}/histogram_number_of_similar_{system}_n{n}.npy")
    bin_edges = np.load(f"rencounters/n{n}/bin_edges_{system}_n{n}.npy")
    i_mean = np.load(f"rencounters/n{n}/mean_mi_by_bin_{system}_n{n}.npy")
    var = np.load(f"rencounters/n{n}/mean_mi_by_bin_{system}_n{n}.npy")

    #hist
    ax1 = fig.add_subplot(gs[row: row+2, col])
    ax1.hist(data, bins=90, density=True)
    ax1.set_title(f"Histograma n={n}")
    distc = [dist[x] for x in bin_edges]
    ax1.plot(bin_edges, distc)

    #mean

    ax2 = fig.add_subplot(gs[row+2, col])
    ax2.scatter(bin_edges, i_mean, marker=".")


    #vars
    ax3 = fig.add_subplot(gs[row+3, col])
    ax3.scatter(bin_edges, var, marker=".")

plt.show()




