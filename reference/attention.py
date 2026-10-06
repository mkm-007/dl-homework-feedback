"""Reference solution for the multi-head attention assignment. Used to evaluate the checks."""

import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def causal_mask(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def padding_mask(lengths, T):
    return (torch.arange(T)[None, :] < lengths[:, None])[:, None, None, :]


def split_heads(x, n_heads):
    B, T, D = x.shape
    return x.view(B, T, n_heads, D // n_heads).transpose(1, 2)


def merge_heads(x):
    B, H, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * dh)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        q, k, v = (split_heads(p(x), self.n_heads) for p in (self.q_proj, self.k_proj, self.v_proj))
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        return self.out_proj(merge_heads(out))


def positional_encoding(T, d_model):
    pos = torch.arange(T).float()[:, None]
    freq = torch.exp(-math.log(10000) * torch.arange(0, d_model, 2).float() / d_model)
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe
