import torch
import torch.linalg as tla
from torch import nn
import einops as ein


def _heat_kernel(x, sigma=2):
    x = torch.as_tensor(x)
    # x: (... p m)
    w = torch.cdist(x, x) ** 2  # (... p p)
    # median: (... 1 1)
    median = ein.rearrange(w, "... p r -> ... 1 (p r)").quantile(0.5, -1, keepdim=True)
    w = (-w / (sigma * median)).exp().clone()
    d = 1 / w.sum(-1)  # (... p)
    w *= d[..., None, :]  # (... 1 p)
    w *= d[..., None]  # (... p 1)
    d = 1 / w.sum(-1).sqrt()  # (... p)
    w *= d[..., None, :]
    w *= d[..., None]
    return w


def les_descriptor(w, mu=2.2e-16, m=2, nev=500, gamma=1e-6) -> torch.Tensor:
    w = torch.as_tensor(w)
    # h: (... n n)
    b_dims = w.shape[:-2]
    n = w.shape[-2]

    with torch.no_grad():
        # (... n m*nev)
        omega: torch.Tensor
        omega = torch.randn(*b_dims, n, m * nev, dtype=w.dtype)
        omega, *_ = tla.qr(omega)

    # (... n n) = (... n n) @ (... n n)
    y = w @ omega
    # (... n 1)
    nu: torch.Tensor
    nu = mu * tla.matrix_norm(y, ord=2, keepdim=True)

    # (... n n) = (... n n) + (... 1 1) * (... n n)
    y_nu = y + nu * omega
    # (... n n) = (... n n) @ (... n n)
    b = omega.transpose(-2, -1) @ y_nu
    # (... n n)
    c: torch.Tensor
    c = tla.cholesky((b + b.transpose(-2, -1)) / 2).transpose(-2, -1)
    # (... n)
    eigvals: torch.Tensor
    eigvals = tla.svdvals(y_nu @ tla.inv(c))
    eigvals = torch.clamp(eigvals**2 - nu.squeeze(-1), min=0)
    eigvals, _ = eigvals.sort()
    eigvals = eigvals[..., -nev:]

    return torch.log(eigvals + gamma)


def les_distance(desc1, desc2) -> torch.Tensor:
    desc1 = torch.as_tensor(desc1)
    desc2 = torch.as_tensor(desc2)

    smallest_n = min(desc1.shape[-1], desc2.shape[-1])
    # (...)
    les_dist = tla.vector_norm(
        desc1[..., :smallest_n] - desc2[..., :smallest_n], dim=-1
    )

    return les_dist


class LES(nn.Module):
    def __init__(self, sigma=2, mu=2.2e-16, m=2, nev=500, gamma=1e-6) -> None:
        super().__init__()
        self.sigma = sigma
        self.mu = mu
        self.m = m
        self.nev = nev
        self.gamma = gamma
        self._name = "LESLoss"

    def forward(self, x1, x2):
        no_batch_dim = len(x1.shape) < 3
        if no_batch_dim:
            x1 = ein.rearrange(x1, "n p -> 1 n p")
            x2 = ein.rearrange(x2, "m o -> 1 m o")

        # (... n)
        desc1 = les_descriptor(
            _heat_kernel(x1, self.sigma), self.mu, self.m, self.nev, self.gamma
        )
        # (... m)
        desc2 = les_descriptor(
            _heat_kernel(x2, self.sigma), self.mu, self.m, self.nev, self.gamma
        )
        # (...)
        les_dist = les_distance(desc1, desc2)

        if no_batch_dim:
            les_dist.squeeze_()

        return les_dist
