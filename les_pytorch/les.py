import numpy as np
import scipy.linalg as spla
import torch
import torch.linalg as tla
from torch import nn
import torch.nn.functional as F


class LES(nn.Module):
    def __init__(self, sigma=2) -> None:
        super().__init__()
        self.sigma = sigma
        self.mu = 2.2e-16
        self.m = 2
        self.nev = 500
        self.gamma = 1e-6
    
    def forward(self, x, y):
        x_eigvals = self._log_eigenvalues(self._heat_kernel(x))
        y_eigvals = self._log_eigenvalues(self._heat_kernel(y))
        return tla.vector_norm(x_eigvals - y_eigvals)

    def _heat_kernel(self, x):
        w = torch.cdist(x, x)**2
        w = torch.exp( - w / (self.sigma * w.quantile(0.5)))
        d = 1 / w.sum(1)
        w *= d
        w *= d[:, None]
        d = 1 / w.sum(1).sqrt()
        w *= d
        w *= d[:, None]
        return w
    
    def _log_eigenvalues(self, h):
        n = h.shape[-2]
        
        with torch.no_grad():
            omega = np.random.randn(n, self.m*self.nev)
            omega = spla.orth(omega)
            omega = torch.from_numpy(omega)
        
        y = h @ omega
        nu = self.mu * tla.matrix_norm(y, ord=2)
        
        y_nu = y + nu * omega
        b = omega.T @ y_nu
        c = tla.cholesky((b + b.T) / 2).T
        eigvals = tla.svdvals(y_nu @ tla.inv(c))
        eigvals = torch.maximum(eigvals**2 - nu, torch.tensor(0))
        eigvals, _ = eigvals.sort()[-self.nev:]
        
        return torch.log(eigvals + self.gamma)
    