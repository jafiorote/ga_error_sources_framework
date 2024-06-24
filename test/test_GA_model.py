import unittest
from models.GAModel import GAModel
import numpy as np


class TestGAModel(unittest.TestCase):

    def setUp(self):

        self.ga_model = GAModel()

        path = "/home/jfiorote/ga_error_sources_framework/test/asserted_values"
        self.probs_poisson_asserted = np.load(f"{path}/probs_poisson_m10_bins11_i01_inat7_sigma0.02.npy", allow_pickle=True)
        self.tran_matrix_asserted = np.load(f"{path}/tran_matrix_probs_poisson_m10_bins11_i01_inat7_sigma0.02_nstep1.npy", allow_pickle=True)
        self.indexes_tran_matrix = np.load(f"{path}/indexes_tran_matrix.npy", allow_pickle=True)
        self.best_path_asserted = np.load(f"{path}/best_path_probs_poisson_m10_bins11_i01_inat7_sigma0.02_nstep1.npy", allow_pickle=True)
        self.step_probs_asserted = np.load(f"{path}/step_probs_probs_poisson_m10_bins11_i01_inat7_sigma0.02_nstep1.npy", allow_pickle=True)

    def test_get_transitions_matrix(self):
        tran_matrix = self.ga_model.get_transitions_matrix(self.probs_poisson_asserted)[0]
        self.assertTrue(np.array_equal(self.tran_matrix_asserted, tran_matrix))
        
    def test_indexes_tran_matrix(self):
        indexes_tran_matrix = self.ga_model.get_transitions_matrix(self.probs_poisson_asserted)[1]
        self.assertTrue(np.array_equal(self.indexes_tran_matrix, indexes_tran_matrix))

    def test_best_path_maximize(self):
        best_path = self.ga_model.get_best_path(self.probs_poisson_asserted, minimize=False, init_state=[0, 0])[0]
        self.assertTrue(np.array_equal(self.best_path_asserted, best_path))

    def test_best_path_prob_steps(self):
        step_probs = self.ga_model.get_best_path(self.probs_poisson_asserted, minimize=False, init_state=[0, 0])[1]
        self.assertTrue(np.array_equal(self.step_probs_asserted, step_probs))



