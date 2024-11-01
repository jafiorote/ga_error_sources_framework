import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models.normPSM import NormPSM
from models.old_pe_PSM import OldPePSM
from models.gaModel import GAModel
#plt.style.use('seaborn-poster')
from utils.plot_funcs import get_percentiles, custom_percentile_cmap, get_sticks


M = 100       
n_max = M + 1 
n_bins = M + 1 
i_nat = 50
i_0s = [10, 20, 30, 40] 
i_0 = 40
sigma2_0s = [0.001, 0.01, 0.1, 1]
sigma2_0 = 1
n_step = 1
minimize=False

fig = plt.figure()
gs = fig.add_gridspec(4, 4)
#cmap = custom_percentile_cmap

for idx1, i_0 in enumerate(i_0s):
    for idx2, sigma2_0 in enumerate(sigma2_0s): 


        norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
        norm_pdfs, norm_probs = norm_psm.get_prob_bins()
        norm_data, norm_bins_center = norm_psm.create_data()
        norm_poison_probs = norm_psm.get_probs()
        reass_norm_poison_probs = norm_psm.reassessment_probs(norm_poison_probs)
        X, Y = np.meshgrid(norm_bins_center / i_nat, np.arange(M  + 1) / M)

        arr = norm_pdfs #* norm_psm.get_poisson_weights()

        Z1 = arr

        ax1 = fig.add_subplot(gs[idx1, idx2], projection='3d')
        c = ax1.plot_surface(X, Y, Z1, cmap="plasma", alpha=0.8)
        cbar = fig.colorbar(c, ax=ax1)

        ax1.set_xlabel(r'$I / I_M$')
        ax1.set_ylabel(r'$n / M$')
        ax1.set_title("".join(['$I_0$', f"={i_0}, ", '$\sigma _0$', f"={sigma2_0}"]))

plt.tight_layout(pad=7.0)
plt.show()

# fig = plt.figure()

# norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
# norm_pdfs, norm_probs = norm_psm.get_prob_bins()
# norm_data, norm_bins_center = norm_psm.create_data()
# norm_poison_probs = norm_psm.get_probs()
# reass_norm_poison_probs = norm_psm.reassessment_probs(norm_poison_probs)
# X, Y = np.meshgrid(norm_bins_center / i_nat, np.arange(M  + 1) / M)
# Z1 = norm_pdfs
# Z2 = norm_pdfs * norm_psm.get_poisson_weights()
# Z3 = norm_psm.reassessment_probs(norm_pdfs) * norm_psm.get_poisson_weights()

# ax1 = fig.add_subplot(131, projection='3d')
# surface1 = ax1.plot_surface(X, Y, Z1, cmap='plasma', alpha=0.8)
# fig.colorbar(surface1, ax=ax1, shrink=0.5, aspect=10)
# ax1.set_xlabel(r'$I / I_M$')
# ax1.set_ylabel(r'$n / M$')
# ax1.set_title("Norm pdf")


# ax2 = fig.add_subplot(132, projection='3d')
# surface2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
# fig.colorbar(surface2, ax=ax2, shrink=0.5, aspect=10)
# ax2.set_xlabel(r'$I / I_M$')
# ax2.set_ylabel(r'$n / M$')
# ax2.set_title("Norm - Poisson pdf")

# ax3 = fig.add_subplot(133, projection='3d')
# surface2 = ax3.plot_surface(X, Y, Z3, cmap='plasma', alpha=0.8)
# fig.colorbar(surface2, ax=ax3, shrink=0.5, aspect=10)
# ax3.set_xlabel(r'$I / I_M$')
# ax3.set_ylabel(r'$n / M$')
# ax3.set_title("Reass Norm - Poisson pdf")

# plt.tight_layout(pad=7.0)
# plt.show()