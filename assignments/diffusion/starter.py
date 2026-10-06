import torch
import torch.nn.functional as F
from torch import nn

T = 1000


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    raise NotImplementedError


def compute_alpha_bar(betas):
    raise NotImplementedError


def q_sample(x0, t, noise, alpha_bar):
    raise NotImplementedError


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x_t, t):
        raise NotImplementedError


def ddpm_loss(model, x0, alpha_bar):
    raise NotImplementedError


def p_sample(model, x_t, t, betas, alpha_bar):
    raise NotImplementedError


@torch.no_grad()
def sample(model, n, betas, alpha_bar):
    raise NotImplementedError
