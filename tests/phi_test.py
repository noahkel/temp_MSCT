import unittest
import torch
from spektralmodell import spectral_model
from nichtlinearitaet import phi, dphi, log_terms

rtol = 1e-10
atol = 1e-12
class TestPhi(unittest.TestCase):
    """
    Testet Phi auf folgende Arten:

        -   Übereinstimmung der expliziten Ableitung mit der automatisch berechneten von pytorch
        -   Einhaltung der Schranken min_e M_{em} leq [DPhi(z)]_{bm} leq max_e M_{em} an DPhi
        -   Einhaltung DPhi(0) = SM
        -   Negative Semidefinitheit der Hessematrix, konstruiert auf 2 Varianten überprüft
    """


    bins = 3
    materialien = 2  

    def test_Derivative_Phi(self):
        E,S,M = spectral_model(n_bin=self.bins)
        Z = torch.randn(4, M.shape[1], dtype=torch.float64, requires_grad=True)
        V = torch.randn(4, S.shape[0], dtype=torch.float64)

        lhs = (phi(Z,S,M) * V).sum()
        grad_autodiff, = torch.autograd.grad(lhs, Z)

        grad_explicit = torch.einsum("...bm,...b->...m", dphi(Z, S, M), V)
        self.assertTrue(torch.allclose(grad_autodiff, grad_explicit, rtol=rtol, atol=atol))

    def test_Derivative_Phi_Bounds(self):
        E,S,M = spectral_model(n_bin=self.bins)
        Z = torch.randn(4, M.shape[1], dtype=torch.float64)
        J = dphi(Z, S, M)
        for b in range(S.shape[0]):
            with self.subTest(b=b):
                active = S[b] > 0
                lo = M[active].amin(dim=0)
                hi = M[active].amax(dim=0)
                self.assertTrue(torch.all(J[..., b, :] >= lo))
                self.assertTrue(torch.all(J[..., b, :] <= hi))

    def test_DPhi_zero_is_SM(self):
        E,S,M = spectral_model(n_bin=self.bins)
        Z = torch.zeros(4, M.shape[1], dtype=torch.float64)
        J = dphi(Z, S, M)
        self.assertTrue(torch.allclose(J, torch.matmul(S,M), rtol=rtol, atol=atol))


    def assert_negative_semidefinite(self, H):
        self.assertTrue(torch.allclose(H, H.T, rtol=rtol, atol=atol), "Hessian is not symmetric")
        largest_eigenvalue = torch.linalg.eigvalsh(H).max()
        scale = max(1.0, float(torch.linalg.norm(H)))
        self.assertTrue(largest_eigenvalue <= atol + rtol*scale, f"Hessian is not negative semidefinite, largest eigenvalue = {largest_eigenvalue.item()}")

    def test_negative_semidefiniteness_hessian(self):
        _, S, M = spectral_model(n_bin=3)

        z = torch.rand(M.shape[1], dtype=torch.float64)

        for b in range(S.shape[0]):
            def phi_b(x):
                return phi(x,S,M)[b]
            H = torch.autograd.functional.hessian(phi_b, z)
            self.assert_negative_semidefinite(H)

    def test_negative_semidefiniteness_hessian_from_dphi(self):
        _, S, M = spectral_model(n_bin=3)

        z = torch.rand(M.shape[1], dtype=torch.float64)

        a = log_terms(z, S, M)
        w = torch.softmax(a, dim=-1)
        mu = dphi(z, S, M)
        for b in range(S.shape[0]):
            w_b = w[b]
            mu_b = mu[b]

            weighted_second_moment = M.T @ (w_b[:, None] * M)

            H = -(weighted_second_moment - torch.outer(mu_b, mu_b))

            self.assert_negative_semidefinite(H)