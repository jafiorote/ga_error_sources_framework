import math
from scipy.stats import norm
from models.proteinSystemModel import ProteinSystemModel


class NormPSM(ProteinSystemModel):

    """
    Implements a normal distribution model of protein system model.

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


    # def statistical_func(self, i: float, sigma2: float, expec: float):

    #     """
    #     Implements normal pmf for a I distribution in a protein system.
        
    #     Parameters
    #     ----------
    #     - i (float): I value to be fitted
    #     - sigma_2 (float): variance of I
    #     - expec (float): expected value of I
        
    #     """

    #     base = 1 / (math.sqrt(sigma2 * 2 * math.pi))
    #     e_pow = -0.5 * math.pow((i - expec) / math.sqrt(sigma2), 2)
    #     return base * math.exp(e_pow)


    def statistical_func(self, i: float, sigma2: float, expec: float):
        """
        Implements normal pmf for a I distribution in a protein system.
        
        Parameters
        ----------
        - i (float): I value to be fitted
        - sigma2 (float): variance of I
        - expec (float): expected value of I
        
        Returns
        -------
        float: Probability density function value at i
        """

        return norm.pdf(i, expec, math.sqrt(sigma2))