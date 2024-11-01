import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models.normPSM import NormPSM
from models.old_pe_PSM import OldPePSM
from models.gaModel import GAModel
# plt.style.use('seaborn-poster')
from utils.plot_funcs import get_percentiles, custom_percentile_cmap, get_sticks
from datetime import date
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

def today():
    return str(date.today()).replace("-", "")

M = 100
n_max = M + 1
n_bins = M + 1
i_nat = 50
i_0s = [10, 20, 30, 40]
sigma2_0s = [0.001, 0.01, 0.1, 1]
n_step = 1
minimize = False

fig = plt.figure(figsize=(13, 13))
gs = fig.add_gridspec(4, 4)

# Definir colormap global
cmap = "plasma"

# Inicializar limites globais do colormap
vmin, vmax = np.inf, -np.inf

# Calcular os valores mínimo e máximo para normalizar o cmap
for i_0 in i_0s:
    for sigma2_0 in sigma2_0s:
        norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
        norm_pdfs, _ = norm_psm.get_prob_bins()
        arr = norm_pdfs * norm_psm.get_poisson_weights()
        vmin = min(vmin, np.min(arr))
        vmax = max(vmax, np.max(arr))


# Criar os subplots
for idx1, i_0 in enumerate(i_0s):
    for idx2, sigma2_0 in enumerate(sigma2_0s):
        norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
        norm_pdfs, norm_probs = norm_psm.get_prob_bins()
        norm_data, norm_bins_center = norm_psm.create_data()
        #norm_poison_probs = norm_psm.get_probs()
        X, Y = np.meshgrid(norm_bins_center / i_nat, np.arange(M + 1) / M)

        arr = norm_pdfs * norm_psm.get_poisson_weights()
        
        print(np.max(arr))
        Z1 = arr

        ax1 = fig.add_subplot(gs[idx1, idx2], projection='3d')
        #ax1.view_init(elev=0, azim=0)
        c = ax1.plot_surface(X, Y, Z1, cmap=cmap, vmin=vmin, vmax=(vmax - vmin) * 0.3, alpha=0.8)

        if idx1 == 0  and idx2 == 0:
            ax1.set_xlabel(r'$I / I_M$', fontsize=14)
            ax1.set_ylabel(r'$n / M$', fontsize=14)
        if idx1 == 0:
            ax1.set_title(f"$\sigma_0$={sigma2_0}", fontsize=17, y=1.2)
            
        ax1.xaxis.set_major_locator(MaxNLocator(3))
        ax1.yaxis.set_major_locator(MaxNLocator(3))
        ax1.zaxis.set_major_locator(MaxNLocator(3))
        ax1.tick_params(axis='both', which='major', labelsize=14)  # Ticks principais
        ax1.tick_params(axis='both', which='minor', labelsize=14)
        ax1.set_facecolor('none')


# Adicionar uma única barra de cores global para todos os subplots
cbar = fig.colorbar(c, ax=fig.get_axes(), orientation='vertical', fraction=0.02, pad=0.04)
cbar.ax.tick_params(labelsize=16)
fig.text(0.0, 0.20, r"$I_0$"+f"={i_0s[3]}", ha='center', va='center', rotation='vertical', fontsize=17)
fig.text(0.0, 0.40, r"$I_0$"+f'={i_0s[2]}', ha='center', va='center', rotation='vertical', fontsize=17)
fig.text(0.0, 0.60, r"$I_0$"+f'={i_0s[1]}', ha='center', va='center', rotation='vertical', fontsize=17)
fig.text(0.0, 0.80, r"$I_0$"+f'={i_0s[0]}', ha='center', va='center', rotation='vertical', fontsize=17)



#plt.tight_layout(pad=2.1)
os.makedirs(f"temp_jupyter_plots/{today()}", exist_ok=True)
plt.savefig(f"temp_jupyter_plots/{today()}/3d_M{M}_poisson.svg")
plt.savefig(f"temp_jupyter_plots/{today()}/3d_M{M}_poisson.png")
#plt.show()