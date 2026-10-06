# Assignment: multi-head attention

Implement the attention block of a transformer from scratch: scaled dot-product attention, masks, head splitting, multi-head self-attention and sinusoidal positional encoding. Everything runs on a CPU in seconds.

## What to define

Masks are boolean tensors where `True` means "may attend". Every query is allowed at least one key.

| Name | Signature | Returns |
|---|---|---|
| `scaled_dot_product_attention` | `(q, k, v, mask=None)` with `q` `(B, H, Tq, dk)`, `k` `(B, H, Tk, dk)`, `v` `(B, H, Tk, dv)` | `(out, weights)`: `out` `(B, H, Tq, dv)` and attention weights `(B, H, Tq, Tk)` |
| `causal_mask` | `(T)` | `(T, T)` boolean mask: query `i` may attend to keys `0 … i` |
| `padding_mask` | `(lengths, T)` with `lengths` a `(B,)` integer tensor | `(B, 1, 1, T)` boolean mask: `True` for real tokens, `False` for padding |
| `split_heads` | `(x, n_heads)` with `x` `(B, T, D)` | `(B, n_heads, T, D // n_heads)`; head `h` holds features `h*dh … (h+1)*dh - 1` |
| `merge_heads` | `(x)` with `x` `(B, H, T, dh)` | `(B, T, H * dh)`, the inverse of `split_heads` |
| `MultiHeadAttention` | `MultiHeadAttention(d_model, n_heads)` then `forward(x, mask=None)` | `(B, T, d_model)`. Must have `nn.Linear(d_model, d_model)` layers named `q_proj`, `k_proj`, `v_proj` and `out_proj` |
| `positional_encoding` | `(T, d_model)` | `(T, d_model)` sinusoidal encoding from *Attention Is All You Need*: `sin(pos / 10000^(2i/d))` in feature `2i`, `cos(...)` in feature `2i + 1` |

## Check your work

```bash
python -m feedback attention path/to/your_notebook.ipynb
```
