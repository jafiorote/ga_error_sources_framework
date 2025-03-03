import numpy as np


class GAModel():

    def get_transitions_matrix(self, arr_prob, n_step=1):

        """
        Compute the transitions matrix for a given array of probability states.

        Parameters
        ----------
        - arr_prob (numpy.ndarray):
            A 2D array of shape (n_max, n_bins) representing the probability states.
        - n_step (int) - optional:
            The maximum step size for transitions in the n_max dimension. Default is 1.

        Returns
        -------
        - transitions (numpy.ndarray):
            A 2D array of shape (n_max * n_bins, n_max * n_bins) representing the transition matrix.
        - idxs (list of lists):
            A list of index pairs, where each pair represents the (n, I) coordinates.
        """
        
        n_max = arr_prob.shape[0]
        n_bins = arr_prob.shape[1]
        transitions = np.zeros((n_max * n_bins, n_max * n_bins), dtype="double")
        idxs = np.array([[x, y] for y in range(n_bins) for x in range(n_max)])

        for i, idx in enumerate(idxs):
            mask = (idxs[:, 1] == (idx[1] + 1)) & (idxs[:, 0] >= (idx[0] - n_step)) & (idxs[:, 0] <= (idx[0] + n_step))
            potential_steps = np.nonzero(mask)[0]

            transitions[i, potential_steps] = arr_prob[idxs[potential_steps, 0], idxs[potential_steps, 1]]
                
            if np.sum(transitions[i]) > 0:
                transitions[i] = transitions[i] / np.sum(transitions[i])

        return transitions, idxs.tolist()
    

    def get_best_path(self, state_probs, minimize=False, init_state=False):

        """
        Determine the best path through a matrix of state probabilities.

        Parameters
        ----------
        - state_probs(numpy.ndarray):
            A 2D array representing the probabilities of each state.
        - minimize(bool) - optional:
            If True, find the path toward to minimum value of I. Default is False that means maximize I.
        - init_state(Union[list, bool]) - optional:
            The initial state to start the path from, specified as [I, n]. If False, 
            it defaults to [0, 0] for maximization and [len(state_probs) - 1, 0] for minimization.

        Returns
        -------
        - list
            A list containing two elements:
            - pathway(list of lists):
                A list of coordinates (n, I) representing the best path through the state_probs matrix.
            - step_probs(List[float]):
                A list of probabilities corresponding to each step in the pathway.

        Notes
        -----
        The function constructs the best path through the given state probabilities matrix. The path 
        can be configured to either maximize or minimize the probabilities, and it begins from the 
        specified initial state. The transition probabilities between states are normalized and used 
        to determine the next step in the path.

        Examples
        --------
        >>> state_probs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> get_best_path(state_probs)
        ([[0, 0], [1, 1], [2, 2]], [0.1, 0.5, 0.9])
        
        >>> get_best_path(state_probs, minimize=True)
        ([[2, 0], [1, 1], [0, 2]], [0.7, 0.5, 0.3])
        """

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
