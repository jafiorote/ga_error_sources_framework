import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_percentiles(arr, n_percentiles=10):

    flat_matrix = arr.flatten()
    percentiles = np.percentile(flat_matrix, np.linspace(0, 100, n_percentiles + 1))
    indices = np.digitize(flat_matrix, percentiles, right=True)
    
    value_interval = []

    for i in range(1, len(percentiles)):
        if flat_matrix[indices == i].size > 0:
            value_interval.append(np.max([flat_matrix[indices == i]]))

    return indices.reshape(arr.shape), value_interval


def custom_percentile_cmap(n_percentiles=10):

    num = 10 if n_percentiles > 10 else n_percentiles
    
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i/(num - 1)) for i in range(num)]
    return ListedColormap(colors)


def get_sticks(n_percentiles=10):
    return [((i + (i + 1)) / 2) / (n_percentiles / 10) for i in range(n_percentiles)]