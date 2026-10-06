"""Reference solution for the conditional VAE assignment. Used to evaluate the checks."""

import torch
import torch.nn.functional as F
from torch import nn

Z_DIM = 2
N_CLASSES = 10


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784 + N_CLASSES, 256), nn.ReLU(), nn.Linear(256, 2 * Z_DIM))

    def forward(self, x, y):
        h = torch.cat([x.flatten(1), F.one_hot(y, N_CLASSES).float()], dim=1)
        mu, logvar = self.net(h).chunk(2, dim=1)
        return mu, logvar.clamp(-8, 8)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Z_DIM + N_CLASSES, 256), nn.ReLU(), nn.Linear(256, 784))

    def forward(self, z, y):
        h = torch.cat([z, F.one_hot(y, N_CLASSES).float()], dim=1)
        return self.net(h).view(-1, 1, 28, 28)


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(dim=1)


def reconstruction_loss(logits, x):
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))


def elbo_loss(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    z = reparameterize(mu, logvar)
    return (reconstruction_loss(decoder(z, y), x) + beta * kl_divergence(mu, logvar)).mean()


def sample(decoder, y):
    z = torch.randn(len(y), Z_DIM)
    return torch.sigmoid(decoder(z, y))
