import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None):
    raise NotImplementedError


def causal_mask(T):
    raise NotImplementedError


def padding_mask(lengths, T):
    raise NotImplementedError


def split_heads(x, n_heads):
    raise NotImplementedError


def merge_heads(x):
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError


def positional_encoding(T, d_model):
    raise NotImplementedError
