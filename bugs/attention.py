"""Known attention mistakes as overrides of the reference solution, plus correct alternatives."""

from __future__ import annotations

import math

import torch
from torch import nn

from reference import attention as ref


def _sdpa(q, k, v, mask=None, scale=None, dim=-1, fill=float("-inf"), after=False, return_scores=False):
    scores = q @ k.transpose(-2, -1) / (math.sqrt(q.shape[-1]) if scale is None else scale)
    if mask is not None and not after:
        scores = scores.masked_fill(~mask, fill)
    weights = scores.softmax(dim=dim)
    if mask is not None and after:
        weights = weights * mask
    return weights @ v, scores if return_scores else weights


def attn_no_scale(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, scale=1.0)


def attn_scale_d(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, scale=q.shape[-1])


def attn_scale_dv(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, scale=math.sqrt(v.shape[-1]))


def attn_softmax_dim(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, dim=-2)


def attn_no_transpose(q, k, v, mask=None):
    w = (q @ k / math.sqrt(q.shape[-1])).softmax(-1)
    return w @ v, w


def mask_inverted(q, k, v, mask=None):
    return _sdpa(q, k, v, None if mask is None else ~mask)


def mask_fill_zero(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, fill=0.0)


def mask_after_softmax(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, after=True)


def causal_inverted(T):
    return torch.triu(torch.ones(T, T, dtype=torch.bool), 1)


def causal_transposed(T):
    return torch.triu(torch.ones(T, T, dtype=torch.bool))


def causal_excludes_self(T):
    return torch.tril(torch.ones(T, T, dtype=torch.bool), -1)


def causal_float(T):
    return torch.tril(torch.ones(T, T))


def padding_flat(lengths, T):
    return torch.arange(T)[None, :] < lengths[:, None]


def padding_inverted(lengths, T):
    return ~ref.padding_mask(lengths, T)


def padding_off_by_one(lengths, T):
    return (torch.arange(T)[None, :] <= lengths[:, None])[:, None, None, :]


def split_without_transpose(x, n_heads):
    B, T, D = x.shape
    return x.reshape(B, n_heads, T, D // n_heads)


def split_interleaved(x, n_heads):
    B, T, D = x.shape
    return x.view(B, T, D // n_heads, n_heads).permute(0, 3, 1, 2)


def merge_without_transpose(x):
    B, H, T, dh = x.shape
    return x.reshape(B, T, H * dh)


class MHANoOutProj(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        q, k, v = (ref.split_heads(p(x), self.n_heads) for p in (self.q_proj, self.k_proj, self.v_proj))
        return ref.merge_heads(ref.scaled_dot_product_attention(q, k, v, mask)[0])


class MHASingleHead(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        q, k, v = (p(x)[:, None] for p in (self.q_proj, self.k_proj, self.v_proj))
        return self.out_proj(ref.scaled_dot_product_attention(q, k, v, mask)[0][:, 0])


class MHAIgnoresMask(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        return super().forward(x, None)


def pe_swapped(T, d_model):
    pe = ref.positional_encoding(T, d_model)
    return torch.stack([pe[:, 1::2], pe[:, 0::2]], 2).reshape(T, d_model)


def pe_concatenated(T, d_model):
    pe = ref.positional_encoding(T, d_model)
    return torch.cat([pe[:, 0::2], pe[:, 1::2]], 1)


def pe_exponent(T, d_model):
    pos = torch.arange(T).float()[:, None]
    freq = torch.exp(-math.log(10000) * torch.arange(d_model // 2).float() / d_model)
    pe = torch.zeros(T, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


MUTATIONS = {
    "attn_no_scale": ({"scaled_dot_product_attention": attn_no_scale}, "attn_no_scale"),
    "attn_scale_d": ({"scaled_dot_product_attention": attn_scale_d}, "attn_scale_d"),
    "attn_scale_dv": ({"scaled_dot_product_attention": attn_scale_dv}, "attn_scale_dv"),
    "attn_softmax_dim": ({"scaled_dot_product_attention": attn_softmax_dim}, "attn_softmax_dim"),
    "attn_no_transpose": ({"scaled_dot_product_attention": attn_no_transpose}, "attn_matmul_shape"),
    "mask_inverted": ({"scaled_dot_product_attention": mask_inverted}, "mask_inverted"),
    "mask_fill_zero": ({"scaled_dot_product_attention": mask_fill_zero}, "mask_fill_zero"),
    "mask_after_softmax": ({"scaled_dot_product_attention": mask_after_softmax}, "mask_after_softmax"),
    "causal_inverted": ({"causal_mask": causal_inverted}, "causal_inverted"),
    "causal_transposed": ({"causal_mask": causal_transposed}, "causal_transposed"),
    "causal_excludes_self": ({"causal_mask": causal_excludes_self}, "causal_excludes_self"),
    "causal_float": ({"causal_mask": causal_float}, "causal_dtype"),
    "padding_flat": ({"padding_mask": padding_flat}, "padding_shape"),
    "padding_inverted": ({"padding_mask": padding_inverted}, "padding_inverted"),
    "padding_off_by_one": ({"padding_mask": padding_off_by_one}, "padding_off_by_one"),
    "split_without_transpose": ({"split_heads": split_without_transpose}, "split_without_transpose"),
    "split_interleaved": ({"split_heads": split_interleaved}, "split_interleaved"),
    "merge_without_transpose": ({"merge_heads": merge_without_transpose}, "merge_without_transpose"),
    "mha_no_out_proj": ({"MultiHeadAttention": MHANoOutProj}, "mha_no_out_proj"),
    "mha_single_head": ({"MultiHeadAttention": MHASingleHead}, "mha_single_head"),
    "mha_ignores_mask": ({"MultiHeadAttention": MHAIgnoresMask}, "mha_leaks_future"),
    "pe_swapped": ({"positional_encoding": pe_swapped}, "pe_swapped"),
    "pe_concatenated": ({"positional_encoding": pe_concatenated}, "pe_concatenated"),
    "pe_exponent": ({"positional_encoding": pe_exponent}, "pe_exponent"),
}


def attn_scale_seq_len(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, scale=math.sqrt(k.shape[-2]))


def attn_returns_scores(q, k, v, mask=None):
    return _sdpa(q, k, v, mask, return_scores=True)


class MHAValueFromKey(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        q, k, v = (ref.split_heads(p(x), self.n_heads) for p in (self.q_proj, self.k_proj, self.k_proj))
        return self.out_proj(ref.merge_heads(ref.scaled_dot_product_attention(q, k, v, mask)[0]))


class MHAAddsResidual(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        return x + super().forward(x, mask)


class MHAScaleByModelDim(ref.MultiHeadAttention):
    def forward(self, x, mask=None):
        q, k, v = (ref.split_heads(p(x), self.n_heads) for p in (self.q_proj, self.k_proj, self.v_proj))
        out, _ = _sdpa(q, k, v, mask, scale=math.sqrt(x.shape[-1]))
        return self.out_proj(ref.merge_heads(out))


def merge_view(x):
    B, H, T, dh = x.shape
    return x.transpose(1, 2).view(B, T, H * dh)


def pe_base_1000(T, d_model):
    pos = torch.arange(T).float()[:, None]
    freq = torch.exp(-math.log(1000) * torch.arange(0, d_model, 2).float() / d_model)
    pe = torch.zeros(T, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


def padding_short(lengths, T):
    return (torch.arange(T)[None, :] < (lengths - 1)[:, None])[:, None, None, :]


HELD_OUT = {
    "attn_scale_seq_len": {"scaled_dot_product_attention": attn_scale_seq_len},
    "attn_returns_scores": {"scaled_dot_product_attention": attn_returns_scores},
    "mha_value_from_key": {"MultiHeadAttention": MHAValueFromKey},
    "mha_adds_residual": {"MultiHeadAttention": MHAAddsResidual},
    "mha_scale_by_model_dim": {"MultiHeadAttention": MHAScaleByModelDim},
    "merge_view": {"merge_heads": merge_view},
    "pe_base_1000": {"positional_encoding": pe_base_1000},
    "padding_short": {"padding_mask": padding_short},
}


def attn_einsum(q, k, v, mask=None):
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * q.shape[-1] ** -0.5
    if mask is not None:
        scores = torch.where(mask, scores, torch.tensor(-1e9))
    w = torch.softmax(scores, dim=3)
    return torch.einsum("bhqk,bhkd->bhqd", w, v), w


def causal_compare(T):
    i = torch.arange(T)
    return i[None, :] <= i[:, None]


def padding_expand(lengths, T):
    return torch.arange(T).expand(len(lengths), T).lt(lengths.unsqueeze(1)).view(len(lengths), 1, 1, T)


def split_permute(x, n_heads):
    B, T, D = x.shape
    return x.reshape(B, T, n_heads, -1).permute(0, 2, 1, 3)


def merge_permute(x):
    B, H, T, dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)


class MHANoBias(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False)
                                                                for _ in range(4))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = (p(x).reshape(B, T, self.h, D // self.h).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        out, _ = attn_einsum(q, k, v, mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


def pe_loop(T, d_model):
    pe = torch.zeros(T, d_model)
    for pos in range(T):
        for i in range(0, d_model, 2):
            angle = pos / math.pow(10000, i / d_model)
            pe[pos, i], pe[pos, i + 1] = math.sin(angle), math.cos(angle)
    return pe


CORRECT = {
    "attn_einsum": {"scaled_dot_product_attention": attn_einsum},
    "causal_compare": {"causal_mask": causal_compare},
    "padding_expand": {"padding_mask": padding_expand},
    "heads_permute": {"split_heads": split_permute, "merge_heads": merge_permute},
    "mha_no_bias": {"MultiHeadAttention": MHANoBias},
    "pe_loop": {"positional_encoding": pe_loop},
}


def namespace(overrides: dict | None = None) -> dict:
    ns = {k: v for k, v in vars(ref).items() if not k.startswith("__")}
    ns.update(overrides or {})
    return ns
