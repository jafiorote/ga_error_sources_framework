import math
from decimal import Decimal, getcontext
from scipy.integrate import quad
import numpy as np
import scipy.special
import pdb


def expec_n(m, i_0, i_nat):

    alpha = math.log(i_nat - i_0 + 1) / m
    e_n = [i_0 + (math.exp(alpha * n) - 1) for n in range(m)]
    return e_n


def sigma_n(m, sigma_0):

    beta = math.log(sigma_0 + 1) / m
    s_n = [sigma_0 - (math.exp(beta * n) - 1) for n in range(m)]
    return s_n


def poison(n):

    # mean = 1
    return float(1 / (Decimal(math.factorial(n)) * Decimal(math.e)))


def lognorm_func(s, sigma, expec):

    base = 1 / (s * math.sqrt(sigma * 2 * math.pi))
    e_pow = - (math.pow(math.log(s) - expec, 2)) / (sigma * 2)
    return base * math.exp(e_pow)


def lognorm_fit(data_s, sigma, expec):

    return np.array([lognorm_func(s, sigma, expec) for s in data_s])


def pdf(data_s, sigma, expec, n):

    w = poison(n)
    return np.array([w * lognorm_func(s, sigma, expec) for s in data_s])


def lognorm_prob(x1, x2, sigma, expec):

    return quad(lognorm_func, x1, x2, args=(sigma, expec))[0]


def reassessment(data, p=0.1, weight=True):

    M = data.shape[0]
    n_bins = data.shape[1]
    reass_pdfs = np.zeros((M, n_bins), dtype='float')

    for n_ in range(M):
        for n in range(M):
            w = poison(n) if weight else 1
            for m in range(M - n):
                b = scipy.special.binom(M - n, m)
                pb = math.pow(p, m) * math.pow(1 - p, M - n - m)
                wnm = b * pb * w

                if n_ == n + m:
                    reass_pdfs[n_] += wnm * data[n]

    return reass_pdfs


def get_n_prob_transitions(m):

    arr = np.zeros((m + 1, m + 1), dtype=float)

    for i in range(m + 1):
        for j in [i - 1, i, i + 1]:
            if 0 <= j < m + 1:
                arr[i, j] = poison(j)
        total = np.sum(arr[i])
        arr[i] /= total

    return arr


def markov_process(prob_arr, trajectory, pathway=[1, 1, 1, 1, 1, 1, 1, 1, 1]):

    prob_steps = []

    for idx, step in enumerate(trajectory):

        n, i = step
        if idx == 0:
            prob_steps.append(prob_arr[n, i]) 
        prob_all_steps = 0
        directions = [
                      [n - 1, i],  # up
                      [n - 1, i + 1],  # front up
                      [n, i + 1],  # front
                      [n + 1, i + 1],  # front down
                      [n + 1, i],  # down
                      [n + 1, i - 1],  # back down
                      [n, i - 1],  # back
                      [n - 1, i - 1],  # back up
                      [n, i],  # stay
                      ]

        if idx < len(trajectory) - 1:

            for path, (y, x) in enumerate(directions):
                prob_step = prob_arr[y, x] * pathway[path] if 0 <= y < prob_arr.shape[0] \
                                                              and 0 <= x < prob_arr.shape[1] else 0
                prob_all_steps += prob_step

            next_step_idx = directions.index(trajectory[idx + 1])
            next_step = directions[next_step_idx]
            prob = (prob_arr[next_step[0], next_step[1]] * pathway[next_step_idx] / prob_all_steps)
            if prob > 0:
                prob_steps.append(prob)

    return prob_steps


def sum_decimal(vec):

    total = Decimal(0)
    for elem in vec:
        total += Decimal(elem)
    return total


def get_transitions_matrix(arr_prob):

    n_max = arr_prob.shape[0]
    n_bins = arr_prob.shape[1]
    
    transitions = np.zeros((n_max * n_bins, n_max * n_bins), dtype="double")

    idxs = [[x, y] for y in range(n_bins) for x in range(n_max)]
   
    for i, idx1 in enumerate(idxs):
        for j, idx2 in enumerate(idxs):
            if (idx1[0] - 1 <= idx2[0] <= idx1[0] + 1) and (idx2[1] == idx1[1] + 1):
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])

        if np.sum(transitions[i]) > 0:
            transitions[i] = transitions[i] / np.sum(transitions[i])
     
    return transitions, idxs


def get_transitions_matrix_allpaths(arr_prob, pathway):

    n_max = arr_prob.shape[0]
    n_bins = arr_prob.shape[1]
    
    transitions = np.zeros((n_max * n_bins, n_max * n_bins), dtype="double")
    idxs = [[x, y] for y in range(n_bins) for x in range(n_max)]

    # steps for tuple [n, i]
    for i, idx1 in enumerate(idxs):
        for j, idx2 in enumerate(idxs):

            if idx2[0] == idx1[0] and idx2[1] == idx1[1] and pathway["stay"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] + 1 and idx2[1] == idx1[1] and pathway["up"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] + 1 and idx2[1] == idx1[1] + 1 and pathway["front_up"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] and idx2[1] == idx1[1] + 1 and pathway["front"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] - 1 and idx2[1] == idx1[1] + 1 and pathway["front_down"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] - 1 and idx2[1] == idx1[1] and pathway["down"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] - 1 and idx2[1] == idx1[1] - 1 and pathway["back_down"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] and idx2[1] == idx1[1] - 1 and pathway["back"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            elif idx2[0] == idx1[0] + 1 and idx2[1] == idx1[1] - 1 and pathway["back_up"]:
                transitions[i, j] = Decimal(arr_prob[idx2[0], idx2[1]])
            else:
                pass

        if sum(transitions[i]) > 0:
            transitions[i] = transitions[i] / np.sum(transitions[i])

    return transitions, idxs

# def get_random_pathway(transitions_matrix, indexes, n_max):

#     rng = np.random.default_rng()
#     probs = [1]
#     pathway = [[0, 0]]
#     n_step = 0
#     next_n_step = 0
#     for i in range(n_max - 1):
#         next_n_step = rng.choice(transitions_matrix.shape[0], 1, 
#                                  p=transitions_matrix[n_step].flatten())[0]
#         prob = transitions_matrix[n_step].flatten()[next_n_step]   
#         probs.append(prob)                           
#         pathway.append(indexes[int(next_n_step)])

#         n_step = next_n_step

#     return [pathway, probs]


def get_random_pathway(transitions_matrix, indexes, idx_max, init_state=[0, 0]):
    
    rng = np.random.default_rng()
    probs = [1]
    pathway = [init_state]
    n_step = indexes.index(init_state)
    next_n_step = 0
    
    for i in range(idx_max - init_state[1] - 1): # doesn't count first state
        next_n_step = rng.choice(transitions_matrix.shape[0], 1, 
                                 p=transitions_matrix[n_step].flatten())[0]
        prob = transitions_matrix[n_step].flatten()[next_n_step]
        probs.append(prob)                           
        pathway.append(indexes[int(next_n_step)])

        n_step = next_n_step

    return [pathway, probs]                        


def get_random_pathway_allpaths(transitions_matrix, indexes, max_steps, init_state=[0, 0]):
    
    rng = np.random.default_rng()
    probs = [1]
    pathway = [init_state]
    n_step = indexes.index(init_state)
    next_n_step = 0
    
    for i in range(0, max_steps):
        
        next_n_step = rng.choice(transitions_matrix.shape[0], 1, 
                                 p=transitions_matrix[n_step].flatten())[0]
        
        prob = transitions_matrix[n_step].flatten()[next_n_step]
        probs.append(prob)                           
        pathway.append(indexes[int(next_n_step)])

        n_step = next_n_step

    return [pathway, probs]     


def get_best_path(transitions_matrix, indexes, idx_max, init_state=[0, 0]):
    probs = [1]
    pathway = [init_state]
    n_step = indexes.index(init_state)
    next_n_step = 0
    
    for i in range(idx_max - init_state[1] - 1): # doesn't count first state
        next_n_step = np.argmax(transitions_matrix[n_step].flatten())
        prob = transitions_matrix[n_step].flatten()[next_n_step]
        if not prob > 0:
            break
        probs.append(prob)                           
        pathway.append(indexes[int(next_n_step)])

        n_step = next_n_step

    return [pathway, probs]    


def get_best_path_allpaths(transitions_matrix, indexes, max_steps, init_state=[0, 0]):
    probs = [1]
    pathway = [init_state]
    n_step = indexes.index(init_state)
    next_n_step = 0
    for i in range(0, max_steps):
        next_n_step = np.argmax(transitions_matrix[n_step].flatten())
        prob = transitions_matrix[n_step].flatten()[next_n_step]
        probs.append(prob)                           
        pathway.append(indexes[int(next_n_step)])

        n_step = next_n_step

    return [pathway, probs]  


def random_pathways(arr_prob, n_paths=1, init_state=[0, 0]):

    transitions, idxs = get_transitions_matrix(arr_prob)
    paths = []

    for i in range(n_paths):
        paths.append(get_random_pathway(transitions, idxs, arr_prob.shape[1] - 1, init_state))

    return(paths)

def random_pathways_allpaths(arr_prob, pathway, n_paths=1, init_state=[0, 0]):

    transitions, idxs = get_transitions_matrix_allpaths(arr_prob, pathway)
    paths = []

    for i in range(n_paths):
        paths.append(get_random_pathway_allpaths(transitions, idxs, len(arr_prob), init_state))

    return(paths)
