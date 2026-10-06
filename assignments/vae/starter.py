import torch
import torch.nn.functional as F
from torch import nn

Z_DIM = 2
N_CLASSES = 10


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, y):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, z, y):
        raise NotImplementedError


def reparameterize(mu, logvar):
    raise NotImplementedError


def kl_divergence(mu, logvar):
    raise NotImplementedError


def reconstruction_loss(logits, x):
    raise NotImplementedError


def elbo_loss(x, y, encoder, decoder, beta=1.0):
    raise NotImplementedError


def sample(decoder, y):
    raise NotImplementedError
