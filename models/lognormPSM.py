import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from models.proteinSystemModel import ProteinSystemModel


class LogNormPSM(ProteinSystemModel):

    """
    Implements a log-normal distribution model of protein system model.

    """
        
    def __init__(self, M:int, i_0:float, i_nat:float, sigma2_0:float, n_bins:int, verbose:bool=False):

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

        super().__init__(M, i_0, i_nat, sigma2_0, n_bins, verbose)


    def statistical_func(self, i: float, sigma2: float, expec: float):

        """
        Implements log-normal pmf for a I distribution in a protein system.

        Parameters
        ----------
        s (float): exponencial I value to be fitted
        sigma_2 (float): variance of I
        expec (float): expected value of I  
        
        """


        sigma = np.sqrt(sigma2)
        scale = np.exp(expec)

        return lognorm.pdf(np.exp(i), s=sigma, scale=scale)

    

    def prob_interval(self, i1, i2, sigma2, expec):
        
        sigma = np.sqrt(sigma2)
        scale = np.exp(expec)
        
        cdf_x1 = lognorm.cdf(np.exp(i1), s=sigma, scale=scale)
        cdf_x2 = lognorm.cdf(np.exp(i2), s=sigma, scale=scale)
        
        return cdf_x2 - cdf_x1
        

        