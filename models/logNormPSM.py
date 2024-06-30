import numpy as np
import math
from models.proteinSystemModel import ProteinSystemModel


class LogNormPSM(ProteinSystemModel):

    """
    Implements a log-normal distribution model of protein system model.

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

        super().__init__(M, i_0, i_nat, sigma2_0)
        self.__M = M
        self.__i_0 = i_0
        self.__i_nat = i_nat
        

    def create_data(self):

        """
        Generate data with exp(I) for bins edges and bins centers.

        """

        n_bins = self.__M + 1

        data =  np.linspace(0, self.__i_nat, n_bins + 1) # get edges of bins
        v_func = np.vectorize(lambda x: math.exp(x))
        data_s = v_func(data)
        bins_center = np.array([(data_s[x] + data_s[x + 1]) / 2 for x in range(n_bins)])
        
        return data_s, bins_center


    def statistical_func(self, s: float, sigma2: float, expec: float):

        """
        Implements log-normal pmf for a I distribution in a protein system.

        Parameters
        ----------
        s (float): exponencial I value to be fitted
        sigma_2 (float): variance of I
        expec (float): expected value of I
        
        """

        base = 1 / (s * math.sqrt(sigma2 * 2 * math.pi))
        e_pow = - (math.pow(math.log(s) - expec, 2)) / (sigma2 * 2)
        return base * math.exp(e_pow)