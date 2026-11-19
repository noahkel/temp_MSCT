import unittest
import torch
from spektralmodell import spectral_model



class TestSpectralModel(unittest.TestCase):
    """
    Testet das Spektralmodell auf folgende Arten:

        -   Dimensionen stimmen überein
        -   Normalisierung von S ist korrekt angewandt

    """


    bins = 3
    materialien = 2  

    def test_Dimensions_E_S_M(self):
        E,S,M = spectral_model(n_bin=self.bins)
        self.assertEqual(S.shape[0], self.bins)
        self.assertEqual(len(E), S.shape[1])
        self.assertEqual(M.shape[1], self.materialien)
        self.assertEqual(len(E), M.shape[0])

    def test_S_Normalization(self):
        E,S,M = spectral_model(n_bin=self.bins)
        ones = torch.ones_like(S.sum(dim=1))
        self.assertTrue(torch.equal(S.sum(dim=1), ones))
