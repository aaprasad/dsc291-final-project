import torch
import torch.linalg as tla
from torch import nn
import einops as ein


class LES(nn.Module):
    def __init__(self, sigma=2, mu=2.2e-16, m=2, nev=500, gamma=1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.mu = mu
        self.m = m
        self.nev = nev
        self.gamma = gamma
        self._name = "LESLoss"
    
    def forward(self, x, y):
        no_batch_dim = len(x.shape) < 3
        if no_batch_dim:
            x = ein.rearrange(x, 'n p -> 1 n p')
            y = ein.rearrange(y, 'm o -> 1 m o')
        
        # (... n)
        x_eigvals = self._log_eigenvalues(self._heat_kernel(x))
        # (... m)
        y_eigvals = self._log_eigenvalues(self._heat_kernel(y))
        smallest_n = min(x_eigvals.shape[-1], y_eigvals.shape[-1])
        # (...)
        les_dist = tla.vector_norm(x_eigvals[..., :smallest_n] - y_eigvals[..., :smallest_n], dim=-1)
        
        if no_batch_dim:
            les_dist.squeeze_()
            
        return les_dist

    def _heat_kernel(self, x):
        # x: (... p m)
        w = torch.cdist(x, x)**2 # (... p p)
        # median: (... 1 1)
        median = ein.rearrange(w, '... p r -> ... 1 (p r)').quantile(0.5, -1, keepdim=True)
        w = (- w / (self.sigma * median)).exp().clone()
        d = 1 / w.sum(-1) # (... p)
        w *= d[..., None, :] # (... 1 p)
        w *= d[..., None] # (... p 1)
        d = 1 / w.sum(-1).sqrt() # (... p)
        w *= d[..., None, :]
        w *= d[..., None]
        return w
    
    def _log_eigenvalues(self, h):
        # h: (... n n)
        b_dims = h.shape[:-2]
        n = h.shape[-2]
        
        with torch.no_grad():
            # (b n m*nev)
            omega = torch.randn(*b_dims, n, self.m*self.nev, dtype=h.dtype)
            omega, *_ = tla.qr(omega)
        
        # (b n n) = (b n n) @ (b n n)
        y = h @ omega
        # (b n 1)
        nu = self.mu * tla.matrix_norm(y, ord=2, keepdim=True)
        
        # (b n n) = (b n n) + (b 1 1) * (b n n)
        y_nu = y + nu * omega
        # (b n n) = (b n n) @ (b n n)
        b = omega.transpose(-2, -1) @ y_nu
        # (b n n)
        c = tla.cholesky((b + b.transpose(-2, -1)) / 2).transpose(-2, -1)
        # (b n)
        eigvals = tla.svdvals(y_nu @ tla.inv(c))
        eigvals = torch.clamp(eigvals**2 - nu.squeeze(-1), min=0)
        eigvals, _ = eigvals.sort()
        eigvals = eigvals[..., -self.nev:]
        
        return torch.log(eigvals + self.gamma)
