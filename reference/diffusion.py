"""Reference solution for the 2D diffusion assignment. Used to evaluate the checks."""

import math

import torch
import torch.nn.functional as F
from torch import nn

T = 1000


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def compute_alpha_bar(betas):
    return torch.cumprod(1 - betas, dim=0)


def _expand(values, like):
    return values.view(-1, *([1] * (like.dim() - 1)))


def q_sample(x0, t, noise, alpha_bar):
    ab = _expand(alpha_bar[t], x0)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


class Denoiser(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, t):
        freqs = torch.exp(-math.log(1000) * torch.arange(8) / 8)
        angles = t.float()[:, None] * freqs[None]
        return self.net(torch.cat([x, angles.sin(), angles.cos()], dim=1))


def ddpm_loss(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(q_sample(x0, t, noise, alpha_bar), t), noise)


def p_sample(model, x_t, t, betas, alpha_bar):
    tt = torch.full((len(x_t),), t, dtype=torch.long)
    eps = model(x_t, tt)
    mean = (x_t - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / (1 - betas[t]).sqrt()
    if t == 0:
        return mean
    return mean + betas[t].sqrt() * torch.randn_like(x_t)


@torch.no_grad()
def sample(model, n, betas, alpha_bar):
    x = torch.randn(n, 2)
    for t in reversed(range(len(betas))):
        x = p_sample(model, x, t, betas, alpha_bar)
    return x
