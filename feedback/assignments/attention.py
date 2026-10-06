"""Checks for the multi-head attention assignment (assignments/attention/README.md)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from feedback.core import Check, Fail

LEARN = "Vaswani et al., Attention Is All You Need (arXiv:1706.03762), Sections 3.2 and 3.5"


def _close(a, b, rtol=1e-4, atol=1e-5) -> bool:
    return isinstance(a, torch.Tensor) and a.shape == b.shape and torch.allclose(a.float(), b, rtol=rtol, atol=atol)


def _attend(q, k, v, mask=None, scale=None, dim=-1, fill=float("-inf"), after=False):
    scores = q @ k.transpose(-2, -1) / (math.sqrt(q.shape[-1]) if scale is None else scale)
    if mask is not None and not after:
        scores = scores.masked_fill(~mask, fill)
    weights = scores.softmax(dim=dim)
    if mask is not None and after:
        weights = weights * mask
    return weights @ v, weights


def _pair(result, name="scaled_dot_product_attention"):
    if not (isinstance(result, tuple) and len(result) == 2):
        raise Fail(f"{name} should return a tuple (out, weights).", "attn_return")
    return result


def check_attention_values(ns):
    torch.manual_seed(20)
    q, k, v = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 7, 16), torch.randn(2, 3, 7, 4)
    try:
        out, w = _pair(ns["scaled_dot_product_attention"](q, k, v))
    except RuntimeError as e:
        if any(s in str(e) for s in ("must match", "cannot be multiplied", "mat1 and mat2", "batch2 tensor")):
            raise Fail("A matrix product has mismatched shapes. Scores are q @ k with k's last two dimensions "
                       "swapped, giving (Tq, Tk); the output is weights @ v.", "attn_matmul_shape") from e
        raise
    ref_out, ref_w = _attend(q, k, v)
    if _close(out, ref_out) and _close(w, ref_w):
        return
    if isinstance(w, torch.Tensor) and w.shape == ref_w.shape and not torch.allclose(w.sum(-1), torch.ones(1)):
        if torch.allclose(w.sum(-2), torch.ones(1), atol=1e-5):
            raise Fail("Your softmax normalizes over the queries (dim -2). Each query distributes its attention "
                       "over the keys, so normalize over the last dimension.", "attn_softmax_dim")
        raise Fail("Each row of the attention weights should sum to 1.", "attn_not_normalized")
    wrong = {
        "attn_no_scale": (1.0, "The scores are not scaled. Without dividing by sqrt(dk), dot products grow with "
                               "dimension and the softmax saturates, which shrinks its gradients."),
        "attn_scale_d": (16.0, "You divide the scores by dk instead of sqrt(dk). Dot products of random vectors "
                               "have variance dk, so sqrt(dk) brings them back to unit variance."),
        "attn_scale_dv": (2.0, "You scale by the square root of the value dimension. The scale comes from the "
                               "query and key dimension dk, which is q.shape[-1]."),
    }
    for mistake, (scale, message) in wrong.items():
        if _close(w, _attend(q, k, v, scale=scale)[1]):
            raise Fail(message, mistake)
    raise Fail("Your attention does not match softmax(q k^T / sqrt(dk)) v.", "attn_wrong")


def check_attention_mask(ns):
    torch.manual_seed(21)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    mask = torch.rand(2, 1, 5, 6) < 0.5
    mask[..., 0], mask[..., -1] = True, False
    plain = _pair(ns["scaled_dot_product_attention"](q, k, v))
    if not (_close(plain[0], _attend(q, k, v)[0]) and _close(plain[1], _attend(q, k, v)[1])):
        raise Fail("Fix scaled dot-product attention without a mask first; this check builds on it.", "")
    out, w = _pair(ns["scaled_dot_product_attention"](q, k, v, mask))
    ref_out, ref_w = _attend(q, k, v, mask)
    if _close(out, ref_out) and _close(w, ref_w, atol=1e-6):
        return
    if _close(w, _attend(q, k, v, ~mask)[1], atol=1e-6):
        raise Fail("Your mask is inverted: positions marked True (allowed) are blocked, and padding is "
                   "attended to.", "mask_inverted")
    if _close(w, _attend(q, k, v, mask, fill=0.0)[1]):
        raise Fail("Masked scores are set to 0, but exp(0) = 1, so masked keys still get weight. Fill them with "
                   "-inf (or a very large negative number) before the softmax.", "mask_fill_zero")
    if _close(w, _attend(q, k, v, mask, after=True)[1]):
        raise Fail("The mask is applied after the softmax, so the remaining weights no longer sum to 1. Mask the "
                   "scores before the softmax.", "mask_after_softmax")
    if isinstance(w, torch.Tensor) and w.shape == ref_w.shape and (w * ~mask).abs().max() > 1e-6:
        raise Fail("Masked positions still receive attention weight.", "mask_leaks")
    raise Fail("With a mask, your attention does not match the reference behavior.", "mask_wrong")


def check_causal_mask(ns):
    out = ns["causal_mask"](5)
    ref = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    if not isinstance(out, torch.Tensor) or out.shape != (5, 5):
        raise Fail(f"causal_mask(5) should have shape (5, 5); got {getattr(out, 'shape', type(out))}.",
                   "causal_shape")
    if out.dtype != torch.bool:
        raise Fail(f"causal_mask should be a boolean tensor (True = may attend); yours has dtype {out.dtype}. "
                   "Float masks invert silently when combined with ~ or masked_fill.", "causal_dtype")
    if torch.equal(out, ref):
        return
    if torch.equal(out, ~ref) or torch.equal(out, torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)):
        raise Fail("Your causal mask is inverted: it allows the future and blocks the past.", "causal_inverted")
    if torch.equal(out, torch.triu(torch.ones(5, 5, dtype=torch.bool))):
        raise Fail("Your mask is the transpose of a causal mask: row i should allow columns 0 to i. Rows are "
                   "queries and columns are keys.", "causal_transposed")
    if torch.equal(out, torch.tril(torch.ones(5, 5, dtype=torch.bool), -1)):
        raise Fail("Your mask stops each token from attending to itself. A causal mask allows the current "
                   "position; only later positions are hidden. The first token would have nothing to attend "
                   "to and its weights become NaN.", "causal_excludes_self")
    raise Fail("causal_mask should be True on and below the diagonal.", "causal_wrong")


def check_padding_mask(ns):
    lengths = torch.tensor([3, 5, 1])
    out = ns["padding_mask"](lengths, 5)
    ref = (torch.arange(5)[None, :] < lengths[:, None])[:, None, None, :]
    if isinstance(out, torch.Tensor) and torch.equal(out.bool(), ref) and out.shape == ref.shape:
        return
    if isinstance(out, torch.Tensor) and out.shape == (3, 5):
        raise Fail("padding_mask returns shape (B, T). Attention scores have shape (B, H, Tq, Tk), so a (B, T) "
                   "mask broadcasts against (Tq, Tk) instead of the batch. Return (B, 1, 1, T).", "padding_shape")
    if isinstance(out, torch.Tensor) and out.numel() == ref.numel():
        flat = out.reshape(ref.shape).bool()
        if torch.equal(flat, ~ref):
            raise Fail("Your padding mask is inverted: True should mark real tokens.", "padding_inverted")
        if torch.equal(flat, (torch.arange(5)[None, :] <= lengths[:, None])[:, None, None, :]):
            raise Fail("Your padding mask includes one padding position per sequence: a sequence of length 3 "
                       "has real tokens at positions 0, 1 and 2.", "padding_off_by_one")
    raise Fail("padding_mask should be True for positions before each sequence's length, with shape (B, 1, 1, T).",
               "padding_wrong")


def check_heads(ns):
    x = torch.arange(2 * 3 * 8, dtype=torch.float).view(2, 3, 8)
    out = ns["split_heads"](x, 4)
    ref = x.view(2, 3, 4, 2).transpose(1, 2)
    if not _close(out, ref):
        if isinstance(out, torch.Tensor) and out.shape == ref.shape and torch.equal(out, x.view(2, 4, 3, 2)):
            raise Fail("split_heads reshapes straight to (B, H, T, dh). That mixes positions across heads: "
                       "reshape to (B, T, H, dh) first, then swap the T and H dimensions.",
                       "split_without_transpose")
        if isinstance(out, torch.Tensor) and out.shape == ref.shape and torch.equal(
                out, x.view(2, 3, 2, 4).permute(0, 3, 1, 2)):
            raise Fail("Each head should hold a contiguous block of features. Your split interleaves them "
                       "(feature j goes to head j mod H).", "split_interleaved")
        raise Fail(f"split_heads(x, 4) for x of shape (2, 3, 8) should return shape (2, 4, 3, 2) with head h "
                   f"holding features 2h and 2h + 1.", "split_wrong")
    merged = ns["merge_heads"](ref.contiguous())
    if not _close(merged, x):
        if isinstance(merged, torch.Tensor) and torch.equal(merged, ref.contiguous().reshape(2, 3, 8)):
            raise Fail("merge_heads reshapes (B, H, T, dh) straight to (B, T, D), which mixes positions. Swap H "
                       "and T back first.", "merge_without_transpose")
        raise Fail("merge_heads should undo split_heads exactly.", "merge_wrong")


def _reference_mha(m, x, mask, n_heads):
    def split(t):
        B, T, D = t.shape
        return t.view(B, T, n_heads, D // n_heads).transpose(1, 2)
    q, k, v = split(m.q_proj(x)), split(m.k_proj(x)), split(m.v_proj(x))
    out, _ = _attend(q, k, v, mask)
    B, H, T, dh = out.shape
    return out, m.out_proj(out.transpose(1, 2).reshape(B, T, H * dh))


def check_mha(ns):
    torch.manual_seed(22)
    m = ns["MultiHeadAttention"](16, 4)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        if not isinstance(getattr(m, name, None), nn.Linear):
            raise Fail(f"MultiHeadAttention should have an nn.Linear named {name}.", "mha_layers")
    x = torch.randn(2, 6, 16)
    out = m(x)
    heads, ref = _reference_mha(m, x, None, 4)
    if _close(out, ref, atol=1e-5):
        m(x).pow(2).mean().backward()
        missing = [n for n in ("q_proj", "k_proj", "v_proj", "out_proj")
                   if getattr(m, n).weight.grad is None or getattr(m, n).weight.grad.abs().sum() == 0]
        if missing:
            raise Fail(f"No gradient reaches {', '.join(missing)}.", "mha_no_gradient")
        return
    B, H, T, dh = heads.shape
    if _close(out, heads.transpose(1, 2).reshape(B, T, H * dh), atol=1e-5):
        raise Fail("The output projection is never applied. After merging heads, out_proj mixes information "
                   "across heads.", "mha_no_out_proj")
    if _close(out, _reference_mha(m, x, None, 1)[1], atol=1e-5):
        raise Fail("Your module behaves like a single head: it ignores n_heads. Each head should attend with its "
                   "own slice of the projected features.", "mha_single_head")
    raise Fail("MultiHeadAttention's output does not match projecting q, k and v, splitting heads, attending, "
               "merging and applying out_proj.", "mha_wrong")


def check_mha_causal(ns):
    torch.manual_seed(23)
    m = ns["MultiHeadAttention"](16, 4)
    x = torch.randn(1, 6, 16)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))[None, None]
    a = m(x, mask)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 3, 16)
    b = m(x2, mask)
    if not torch.allclose(a[:, :3], b[:, :3], atol=1e-5):
        raise Fail("With a causal mask, changing tokens 3 to 5 changes the outputs at positions 0 to 2: "
                   "information leaks from the future. Check that forward passes the mask to the attention.",
                   "mha_leaks_future")


def check_positional_encoding(ns):
    out = ns["positional_encoding"](50, 16)
    pos = torch.arange(50).float()[:, None]
    freq = torch.exp(-math.log(10000) * torch.arange(0, 16, 2).float() / 16)
    interleaved = torch.zeros(50, 16)
    interleaved[:, 0::2], interleaved[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    if _close(out, interleaved, atol=1e-4):
        return
    swapped = torch.zeros(50, 16)
    swapped[:, 0::2], swapped[:, 1::2] = torch.cos(pos * freq), torch.sin(pos * freq)
    if _close(out, swapped, atol=1e-4):
        raise Fail("Sine and cosine are swapped: even features use sine, odd features use cosine.", "pe_swapped")
    if _close(out, torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], 1), atol=1e-4):
        raise Fail("Your encoding puts all sines first and all cosines after. The assignment asks for the "
                   "interleaved layout from the paper; models trained with one layout cannot load weights "
                   "trained with the other.", "pe_concatenated")
    freq_i = torch.exp(-math.log(10000) * torch.arange(8).float() / 16)
    wrong = torch.zeros(50, 16)
    wrong[:, 0::2], wrong[:, 1::2] = torch.sin(pos * freq_i), torch.cos(pos * freq_i)
    if _close(out, wrong, atol=1e-4):
        raise Fail("The frequency exponent uses i / d instead of 2i / d, so the wavelengths only reach about "
                   "100 positions instead of 10000 and long sequences repeat.", "pe_exponent")
    raise Fail("positional_encoding does not match sin(pos / 10000^(2i/d)) and cos(pos / 10000^(2i/d)).",
               "pe_wrong")


def check_training(ns):
    torch.manual_seed(24)
    vocab, T, d = 8, 8, 32
    embed, head = nn.Embedding(vocab, d), nn.Linear(d, vocab)
    attn = ns["MultiHeadAttention"](d, 4)
    pe = ns["positional_encoding"](T, d)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]
    params = list(embed.parameters()) + list(head.parameters()) + list(attn.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    losses = []
    for step in range(300):
        tokens = torch.randint(0, vocab, (64, T))
        h = embed(tokens) + pe
        logits = head(h + attn(h, mask))
        loss = F.cross_entropy(logits.reshape(-1, vocab), tokens[:, :1].expand(-1, T).reshape(-1))
        if not torch.isfinite(loss):
            raise Fail(f"The loss became {loss.item()} at step {step}.", "training_nan")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    if last > 0.5 * first:
        raise Fail(f"A one-layer model with your attention should learn to copy the first token to every "
                   f"position, but the loss only went from {first:.2f} to {last:.2f}.", "training_stalls")


CHECKS = [
    Check("attention", "Scaled dot-product attention", ("scaled_dot_product_attention",), check_attention_values,
          LEARN),
    Check("attention_mask", "Attention with a mask", ("scaled_dot_product_attention",), check_attention_mask, LEARN),
    Check("causal_mask", "Causal mask", ("causal_mask",), check_causal_mask, LEARN),
    Check("padding_mask", "Padding mask", ("padding_mask",), check_padding_mask),
    Check("heads", "Splitting and merging heads", ("split_heads", "merge_heads"), check_heads, LEARN),
    Check("mha", "Multi-head attention", ("MultiHeadAttention",), check_mha, LEARN),
    Check("mha_causal", "No information from the future", ("MultiHeadAttention",), check_mha_causal),
    Check("positional_encoding", "Sinusoidal positional encoding", ("positional_encoding",),
          check_positional_encoding, LEARN),
    Check("training", "A short training run learns a simple attention task",
          ("MultiHeadAttention", "positional_encoding"), check_training),
]
