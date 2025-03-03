import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from datetime import date

import pandas as pd

from scipy.stats import linregress


def today():
    return str(date.today()).replace("-", "")

os.makedirs(f"temp_jupyter_plots/{today()}", exist_ok=True)

sys_stats = pd.read_csv("tabela_paper.csv")

plt.style.use('seaborn')
fig = plt.figure(figsize=(7.6, 2.5))
gs = fig.add_gridspec(1, 2) 

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


alpha_norm =[]
alpha_star_norm=[]

for idx, row in sys_stats.iterrows():

    if row["M"] not in [10, 20, 30]:

        color = "red" if row["TP_model_r"] < 0.25 else "#40e0d0"
    
        ax1.scatter(row["alpha_star_norm"], row["alpha_norm"], marker='s', color="#40e0d0", s=20)

        ax2.scatter(row["var"], row["alpha_norm"], marker='s', color=color, s=20)

        alpha_norm.append(row["alpha_norm"])
        alpha_star_norm.append(row["alpha_star_norm"])


slope, intercept, _, _, _ = linregress(alpha_star_norm, alpha_norm)
line = slope * np.array(alpha_star_norm) + intercept

correlation = np.corrcoef(alpha_star_norm, alpha_norm)[0, 1]



ax1.plot(alpha_star_norm, line, c="black")
ax1.grid(True, which='both', alpha=0.1)
ax1.set_xlim([0.08, 0.83])
ax1.set_ylabel(r"$ \alpha / I_M $")
ax1.set_xlabel(r"$ \alpha* / I_M $")
ax1.set_title(f"Correlation Coefficient = {correlation:.2f}")


ax2.grid(True, which='both', alpha=0.1)
#ax2.set_xlim([-0.2, 1.3])
ax2.set_ylabel(r"$ \alpha / I_M $")
ax2.set_xlabel(r"$ \sigma_{0}^{2} $")

# print(len(alpha_star_norm))
#plt.show()
fig.tight_layout()
plt.savefig(f"temp_jupyter_plots/{today()}/sys_corr_{today()}.png", dpi=300)
plt.savefig(f"temp_jupyter_plots/{today()}/sys_corr_{today()}.svg", dpi=300)


