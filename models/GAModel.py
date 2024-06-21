import numpy as np
from decimal import Decimal


class GAModel():

    def get_transitions_matrix(self, arr_prob, n_step=1):

        n_max = arr_prob.shape[0]
        n_bins = arr_prob.shape[1]
        
        transitions = np.zeros((n_max * n_bins, n_max * n_bins), dtype="double")

        idxs = [[x, y] for y in range(n_bins) for x in range(n_max)]
    
        for i, idx1 in enumerate(idxs):
            for j, idx2 in enumerate(idxs):
                if (idx1[0] - n_step <= idx2[0] <= idx1[0] + n_step) and (idx2[1] == idx1[1] + 1):
                    transitions[i, j] = arr_prob[idx2[0], idx2[1]]

            if np.sum(transitions[i]) > 0:
                transitions[i] = transitions[i] / np.sum(transitions[i])
        
        return transitions, idxs


    def get_random_pathway(self, transitions_matrix, indexes, idx_max, init_state=[0, 0]):
        
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
    

    def get_best_path(self, state_probs, minimize=False, init_state=False):

        if not init_state:
            init_state = [len(state_probs) - 1, 0] if minimize else [0, 0]

        if minimize:
            state_probs = np.fliplr(state_probs)
        
        transitions, idxs = self.get_transitions_matrix(state_probs)
        initial_prob = state_probs[init_state[0], init_state[1]]
        
        step_probs = [initial_prob]
        pathway = [init_state]
        step = idxs.index(init_state)
        next_step = 0
        step_max = np.max(np.array(idxs)[:, 1]) - init_state[1]

        for i in range(step_max): # doesn't count first state
            
            next_step = np.argmax(transitions[step].flatten())
            prob = transitions[step].flatten()[next_step]
            if not prob > 0:
                break
            step_probs.append(prob)                           
            pathway.append(idxs[int(next_step)])

            step = next_step

        return [pathway, step_probs]