import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models.normPSM import NormPSM

from models.gaModel import GAModel
# plt.style.use('seaborn-poster')
from utils.plot_funcs import get_percentiles, custom_percentile_cmap, get_sticks
from datetime import date
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter


def today():
    return str(date.today()).replace("-", "")


# data model:

M = 1004
i_0 = 4.16
i_nat = 19.14
n_bins = M + 1
sigma2_0 = 0.003

norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
norm_data, norm_bins_center = norm_psm.create_data()
norm_pdfs, _ = norm_psm.get_prob_bins()

# data rencountres:

path_data = "jao_data/random_mi_genomes/"
data_file = "MI_FOR_GENOME_1BXR_AB.npy"
sample = [0, 1, 2, 4, 8, 16]


fig = plt.figure(figsize=(7, 8))
gs = fig.add_gridspec(2, 4)

for idx, n in enumerate(sample):

    row = int(idx / 3)
    col = idx % 3

    data_n = np.load(f"{path_data}/n{n}/{data_file}")
    
    ax1 = fig.add_subplot(gs[row, col])

    ax1.hist(data_n, bins=M, density=True)
    ax1.plot(np.arange(n_bins), norm_pdfs[n])

plt.show()