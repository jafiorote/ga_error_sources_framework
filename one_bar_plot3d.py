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

def probs_arr_txt(norm_psm, txt):
    norm_pdfs, _ = norm_psm.get_prob_bins()

    if txt == "poisson":
        return norm_pdfs * norm_psm.get_poisson_weights()
    if txt == "r_poisson":
        return norm_psm.reassessment_probs(norm_pdfs * norm_psm.get_poisson_weights())
    else:
        return norm_pdfs

# M = 10
# n_max = M + 1
# n_bins = M + 1
i_nat = 50
i_0s = [10, 20, 30, 40]
sigma2_0s = [0.001, 0.01, 0.1, 1]
n_step = 1
minimize = False



# Definir colormap global
cmap = "plasma"

# Inicializar limites globais do colormap
vmin, vmax = np.inf, -np.inf

for M in [10]:#, 20, 50, 100]:
    n_bins = M + 1
    for txt in ["norm", "poisson", "r_poisson"]:
        
        fig = plt.figure(figsize=(7.2, 4.7))
        gs = fig.add_gridspec(4, 4)
        # Calcular os valores mínimo e máximo para normalizar o cmap
        for i_0 in i_0s:
            for sigma2_0 in sigma2_0s:
                #print(M, i_0, sigma2_0)
                norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
                arr = probs_arr_txt(norm_psm, txt)
                vmin = min(vmin, np.min(arr))
                vmax = max(vmax, np.max(arr))


        # Criar os subplots
        for idx1, i_0 in enumerate(i_0s):
            for idx2, sigma2_0 in enumerate(sigma2_0s):
                norm_psm = NormPSM(M, i_0, i_nat, sigma2_0, n_bins)
                norm_data, norm_bins_center = norm_psm.create_data()
                X, Y = np.meshgrid((norm_bins_center - i_0) / (i_nat - i_0), np.arange(M + 1) / M)

                arr = probs_arr_txt(norm_psm, txt)
                Z1 = arr

                ax1 = fig.add_subplot(gs[idx1, idx2], projection='3d')
                #ax1.view_init(elev=0, azim=0)
                c = ax1.plot_surface(X, Y, Z1, cmap=cmap, vmin=vmin, vmax=(vmax - vmin) * 0.3, alpha=0.8)

                if idx1 == 0  and idx2 == 0:
                    if txt == "r_poisson":
                        ax1.set_ylabel(r"$n' / M$", fontsize=10)
                    else:
                        ax1.set_ylabel(r'$n / M$', fontsize=10)
                    ax1.set_xlabel(r"$I / I'$", fontsize=10)
                    
                if idx1 == 0:
                    ax1.set_title(f"$\sigma_{0}^{2}$={sigma2_0}", fontsize=10, y=1.2)
                    
                ax1.xaxis.set_major_locator(MaxNLocator(3))
                ax1.yaxis.set_major_locator(MaxNLocator(3))
                ax1.zaxis.set_major_locator(MaxNLocator(3))
                ax1.tick_params(axis='both', which='major', labelsize=8, pad=0.5)  # Ticks principais
                ax1.tick_params(axis='both', which='minor', labelsize=8, pad=0.5)
                ax1.set_facecolor('none')
                ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


        # Adicionar uma única barra de cores global para todos os subplots
        cbar = fig.colorbar(c, ax=fig.get_axes(), orientation='vertical', fraction=0.02, pad=0.08)
        cbar.ax.tick_params(labelsize=10)


        fig.text(0.1, 0.20, r"$I_0$"+f"={i_0s[3]}", ha='center', va='center', rotation='vertical', fontsize=10)
        fig.text(0.1, 0.40, r"$I_0$"+f'={i_0s[2]}', ha='center', va='center', rotation='vertical', fontsize=10)
        fig.text(0.1, 0.60, r"$I_0$"+f'={i_0s[1]}', ha='center', va='center', rotation='vertical', fontsize=10)
        fig.text(0.1, 0.80, r"$I_0$"+f'={i_0s[0]}', ha='center', va='center', rotation='vertical', fontsize=10)


        #fig.tight_layout()
        os.makedirs(f"temp_jupyter_plots/{today()}", exist_ok=True)
        plt.savefig(f"temp_jupyter_plots/{today()}/3d_M{M}_{txt}.svg")
        plt.savefig(f"temp_jupyter_plots/{today()}/3d_M{M}_{txt}.png")
        plt.show()