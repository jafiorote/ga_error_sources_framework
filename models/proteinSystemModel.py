import numpy as np
from scipy.integrate import quad
from scipy.special import binom
import math
from decimal import Decimal


class ProteinSystemModel():

    """
    Implements a statistical distribution model of mutual information in a system 
    composed by two protein families.

    """

    def __init__(self, M:int, i_0:float, i_nat:float, sigma2_0:float):


        """ 
        Parameters
        ----------
        - M (int): 
            Number of sequences in MSA;
        - i_0 (float): 
            Mean of I for systems arrangments with n = 0;
        - i_nat (float): 
            I for system native arrangment;
        - sigma2_0 (float): 
            Variance of I for systems arrangments with n = 0;
        """
 
        self.__M = M
        self.__i_0 = i_0
        self.__i_nat = i_nat
        self.__sigma2_0 = sigma2_0
        self.__alpha = self.get_alpha()
        self.__beta = self.get_beta()


    def get_alpha(self):

        return math.log(self.__i_nat - self.__i_0 + 1) / self.__M
    

    def get_beta(self):        

        return math.log(self.__sigma2_0 + 1) / self.__M


    def expec_ns(self):

        return [self.__i_0 + (math.exp(self.__alpha * n) - 1) for n in range(self.__M + 1)]


    def sigma2_ns(self):        
 
        sigma2_ns = []
        for n in range(self.__M + 1):
            sigma2 = self.__sigma2_0 - (math.exp(self.__beta * n) - 1) 
            if sigma2 > 0: 
                sigma2_ns.append(sigma2)
            else:
                sigma2_ns.append(sigma2_ns[n - 1]) # get last valid sigma2
                
        return sigma2_ns


    def create_data(self):

        """
        Generate data to be fitted.

        Returns
        -------
        - data (numpy.ndarray): 
            Vector with bins edges.
        - bins_center (numpy.ndarray): 
            Array of shape (n_max, n_bins) containing probabilities computed for each bin.
        """

        n_bins = self.__M + 1

        data =  np.linspace(self.__i_0, self.__i_nat, n_bins + 1) # get edges of bins
        bins_center = np.array([(data[x] + data[x + 1]) / 2 for x in range(n_bins)])
        
        return data, bins_center


    def __poisson(self, n: int):

        """ 
        Implements Poisson function for given n

        Parameters
        ----------
        - n (int): 
            Number of native base pairings formed between the sequences.
        """

        # lambda = 1
        return float(1 / (Decimal(math.factorial(n)) * Decimal(math.e)))


    def get_poisson_weights(self):

        """
        Compute Poisson weights for each n-log-normal.

        Returns
        -------        
        - weights (numpy.ndarray):
           A 2D array with shape (1, M + 1).
        """

        weights = np.vectorize(self.__poisson)
        return weights(np.arange(self.__M + 1)).reshape(-1, 1)


    def statistical_func(self):

        """ To be defined by subclass """

        pass


    def fit(self, data: float, sigma2: float, expec: float):

        return np.array([self.statistical_func(i, sigma2, expec) for i in data])


    def prob_interval(self, x1: float, x2: float, sigma2: float, expec: float):

        return quad(self.statistical_func, x1, x2, args=(sigma2, expec))[0]


    def get_prob_bins(self):

        """
        Computes probability density functions (PDFs) and bins probabilities for each n of the system.

        Returns
        -------
        - pdfs (numpy.ndarray): 
            Array of shape (n_max, n_bins) containing PDFs computed using log-normal fitting.
        - probs (numpy.ndarray): 
            Array of shape (n_max, n_bins) containing probabilities computed for each bin.

        Notes:
        - Uses internal methods `create_data`, `sigma2_ns`, `expec_ns`, `lognorm_prob`, and `lognorm_fit`.
        - `data` and `bins_center` are obtained from `create_data`.
        - `sigma2` and `expec` are computed using `sigma2_ns` and `expec_ns`.
        - PDFs are computed using `lognorm_fit` for each `sigma2` and `expec`.
        - Truncated probabilities are computed for each bin using the log-normal probability and truncation factor.
        - If the integral in `lognorm_prob` is zero or negative, truncation is set to 1 for that bin.

        """

        n_max = self.__M + 1
        n_bins = self.__M + 1

        data, bins_center = self.create_data()
        sigma2 = self.sigma2_ns()
        expec = self.expec_ns()

        pdfs = np.zeros((n_max, n_bins), dtype=float)
        truncs = np.zeros(n_max, dtype=float)
        probs = np.zeros((n_max, n_bins), dtype=float)

        for n in np.arange(n_max):

            integral = self.prob_interval(data[0], data[-1], sigma2[n], expec[n])    ##
            trunc = 1 / integral if integral > 0 else 1                             ## trunc normal
            truncs[n] = trunc                                                       ##

            pdfs[n] = self.fit(bins_center, sigma2[n], expec[n])

            vec = []
            for idx in range(n_bins):
                a = data[idx]
                b = data[idx + 1]
                integ = self.prob_interval(a, b, sigma2[n], expec[n])
                vec.append(integ * trunc)
            probs[n] = vec

        return pdfs, probs


    def get_probs(self):

        """ Return Poisson weighted probabilities."""

        return self.get_prob_bins()[1] * self.get_poisson_weights()
    

    def reassessment_probs(self, p_array, p=0.25):

        """
        Reevaluates the probability distribution based on the similarity between sequences.

        Parameters
        ----------
        probs (numpy.ndarray): 
            A 2D array of containing bins probabilities to be reassessed.
        p (float) - optional:
            Hamming's distance similarity threshold.
        
        Returns
        -------
        reass_probs (numpy.ndarray): 
            A 2D array of containing reassessed bins probabilities.
        """

        n_bins = self.__M + 1
        reass_probs = np.zeros((self.__M + 1, n_bins), dtype='float')

        # Precompute powers of p and (1 - p)
        p_powers = np.power(p, np.arange(self.__M + 1))
        q_powers = np.power(1 - p, np.arange(self.__M + 1))

        for n in range(self.__M + 1):
            for m in range(self.__M + 1 - n):
                n_ = n + m
                b = binom(self.__M - n, m)
                pb = p_powers[m] * q_powers[self.__M - n - m]
                reass_probs[n_] += b * pb * p_array[n]

        return reass_probs