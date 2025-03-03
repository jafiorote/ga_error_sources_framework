import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
from datetime import date

def today():
    return str(date.today()).replace("-", "")

os.makedirs(f"temp_jupyter_plots/{today()}", exist_ok=True)

plt.style.use('seaborn')

systems = ["1BXR AB", "1ZUN AB", "2D1P BC"]

transitions_1BXR = {0: 29786, 1: 114, -1: 93, -2: 0, 2: 1}
transitions_1ZUN = {0: 29694, 1: 172, -1: 125, -2: 2, 2: 1}
transitions_2D1P = {0: 29884, 1: 57, -1: 52, -2: 1, 2:0}

data_convergence = [np.load(f"jao_data/data_convergence_{x.split(' ')[0]}.npy") for x in systems]
transitions = [transitions_1BXR, transitions_1ZUN, transitions_2D1P]

fig = plt.figure(figsize=(7.4, 3.0))
gs = fig.add_gridspec(2, 3)  

for idx, system in enumerate(systems):

    ax1 = fig.add_subplot(gs[0, idx])
    ax1.grid(True, which='both', alpha=0.1)


    gen_vector = np.arange(len(data_convergence[idx][0])) * 10

    for rep in data_convergence[idx]:
        ax1.plot(gen_vector, rep, c="grey", lw=0.5)
    ax1.plot(gen_vector, np.mean(data_convergence[idx], axis=0), c="black")
    ax1.set_xticks([0, 25000, 50000])
    ax1.set_title(systems[idx])
    ax1.set_xlabel("Gerações")
    ax1.set_ylabel(r"$I*$")


    ax2 = fig.add_subplot(gs[1, idx])
    ax2.grid(True, which='both', alpha=0.1)
    values = np.array(list(transitions[idx].values())) / 30000
    ax2.bar(transitions[idx].keys(), values)
    ax2.set_xticks(list(transitions[idx].keys()))
    #ax2.set_title(f"{systems[idx]} elite")
    ax2.set_xlabel("Transições")
    ax2.set_ylabel("Densidade")

 

    # for barra in barras:
    #     altura = barra.get_height()  # Obtém a altura de cada barra
    #     plt.text(barra.get_x() + barra.get_width() / 2,  # Posição X (centro da barra)
    #             altura + 0.1,                          # Posição Y (um pouco acima da barra)
    #             f'{altura:.1e}%',                      # Texto formatado
    #             ha='center', va='bottom', fontsize=10)

plt.tight_layout()


#plt.show()

plt.savefig(f"temp_jupyter_plots/{today()}/transitions_ga{today()}.png")
plt.savefig(f"temp_jupyter_plots/{today()}/transitions_ga{today()}.svg")
