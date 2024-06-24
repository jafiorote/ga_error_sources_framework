import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from models.logNormPSM import LogNormPSM
from models.normPSM import NormPSM
import logging

#TODO - Codar logs

class TestPSMModels(unittest.TestCase):

    def setUp(self):

        self.M = 10             
        self.n_max = self.M + 1 
        self.n_bins = self.M + 1 
        self.i_nat = 7
        self.i_0 = 1 
        self.sigma2_0 = 0.02
        self.n_step = 1
        self.minimize = False

        # add models for testing.

        self.ln_psm = LogNormPSM(self.M, self.i_0, self.i_nat, self.sigma2_0)
        self.n_psm = NormPSM(self.M, self.i_0, self.i_nat, self.sigma2_0)

        # load asserted values:
        path = "/home/jfiorote/ga_error_sources_framework/test/asserted_values"
        self.expecs_asserted = np.load(f"{path}/expecs_m10_bins11_i01_inat7_sigma0.02.npy")
        self.sigma2s_asserted = np.load(f"{path}/sigma2s_m10_bins11_i01_inat7_sigma0.02.npy")
        self.pdfs_asserted = np.load(f"{path}/pdfs_m10_bins11_i01_inat7_sigma0.02.npy")
        self.poisson_weight_asserted = np.load(f"{path}/poisson_weight_m10.npy")
        self.probs_asserted = np.load(f"{path}/probs_m10_bins11_i01_inat7_sigma0.02.npy")
        self.probs_poisson_asserted = np.load(f"{path}/probs_poisson_m10_bins11_i01_inat7_sigma0.02.npy")


    #testing for log-normal model

    def test_expec_n_values(self):
        expecs = self.ln_psm.expec_ns()
        assert_array_almost_equal(self.expecs_asserted, expecs)

    def test_sigma_n_values(self):
        sigma2s = self.ln_psm.sigma2_ns()
        assert_array_almost_equal(self.sigma2s_asserted, sigma2s)

    def test_alpha_value(self):
        alpha_asserted = 0.19459101490553132
        self.assertAlmostEqual(alpha_asserted, self.ln_psm.get_alpha())

    def test_beta_values(self):
        beta_asserted = 0.001980262729617973
        self.assertAlmostEqual(beta_asserted, self.ln_psm.get_beta())

    def test_lognorm_pdf_ns(self):
        pdfs = self.ln_psm.get_prob_bins()[0]
        self.assertTrue(np.array_equal(self.pdfs_asserted, pdfs))

    def test_poisson_weight_ns(self):
        poisson_weight = self.ln_psm.get_poisson_weights()
        self.assertTrue(np.array_equal(self.poisson_weight_asserted, poisson_weight))

    def test_prob_lognorm_ns(self):
        probs = self.ln_psm.get_prob_bins()[1]
        self.assertTrue(np.array_equal(self.probs_asserted, probs))

    def test_prob_lognorm_weighted_ns(self):
        probs_poisson = self.ln_psm.get_probs()
        self.assertTrue(np.array_equal(self.probs_poisson_asserted, probs_poisson))

    def test_prob_lognorm_rows_equals_1(self):
        self.assertAlmostEqual(len(self.probs_asserted), np.sum(self.probs_asserted))

    def test_prob_poisson_equals_1(self):
        self.assertAlmostEqual(1, np.sum(self.probs_poisson_asserted))
        

    #testing for normal model

    def test_expec_n_values2(self):
        expecs = self.n_psm.expec_ns()
        assert_array_almost_equal(self.expecs_asserted, expecs)

    def test_sigma_n_values2(self):
        sigma2s = self.n_psm.sigma2_ns()
        assert_array_almost_equal(self.sigma2s_asserted, sigma2s)

    def test_alpha_value2(self):
        alpha_asserted = 0.19459101490553132
        self.assertAlmostEqual(alpha_asserted, self.n_psm.get_alpha())

    def test_beta_values2(self):
        beta_asserted = 0.001980262729617973
        self.assertAlmostEqual(beta_asserted, self.n_psm.get_beta())

    def test_norm_pdf_ns(self):
        pdfs = self.n_psm.get_prob_bins()[0]
        self.assertTrue(np.array_equal(self.pdfs_asserted, pdfs))

    def test_poisson_weight_ns(self):
        poisson_weight = self.n_psm.get_poisson_weights()
        self.assertTrue(np.array_equal(self.poisson_weight_asserted, poisson_weight))

    def test_prob_norm_ns(self):
        probs = self.n_psm.get_prob_bins()[1]
        self.assertTrue(np.array_equal(self.probs_asserted, probs))

    def test_prob_norm_weighted_ns(self):
        probs_poisson = self.n_psm.get_probs()
        self.assertTrue(np.array_equal(self.probs_poisson_asserted, probs_poisson))

    def test_prob_norm_rows_equals_1(self):
        self.assertAlmostEqual(len(self.probs_asserted), np.sum(self.probs_asserted))

    def test_prob_poisson_equals_1_2(self):
        self.assertAlmostEqual(1, np.sum(self.probs_poisson_asserted))
        


if __name__ == '__main__':
    unittest.main()