# Assignment: denoising diffusion on 2D data

Implement a denoising diffusion probabilistic model (DDPM) that learns a 2D point distribution, such as a ring or eight Gaussians. Everything runs on a CPU in seconds, so you can watch the forward process destroy the data and the reverse process rebuild it.

## What to define

Define these names at the top level of your notebook or `.py` file. Functions must use only their arguments; the checks pass in their own schedules and models to test each piece on its own.

Timesteps are integers from `0` to `T - 1`, and `alpha_bar[t]` is the product of `(1 - betas[s])` for `s = 0 … t`.

| Name | Signature | Returns |
|---|---|---|
| `T` | constant | `1000` |
| `linear_beta_schedule` | `(T, beta_start=1e-4, beta_end=0.02)` | `betas`, shape `(T,)`, from `beta_start` to `beta_end` |
| `compute_alpha_bar` | `(betas)` | `alpha_bar`, shape `(T,)` |
| `q_sample` | `(x0, t, noise, alpha_bar)` | `x_t`, the noisy version of `x0` at timesteps `t` (a `(B,)` integer tensor) |
| `Denoiser` | `Denoiser()` then `forward(x_t, t)` | predicted noise, same shape as `x_t` of shape `(B, 2)` |
| `ddpm_loss` | `(model, x0, alpha_bar)` | scalar training loss for a batch, using a random timestep per example |
| `p_sample` | `(model, x_t, t, betas, alpha_bar)` | one reverse step `x_{t-1}` for an integer `t`, using variance `betas[t]` |
| `sample` | `(model, n, betas, alpha_bar)` | `n` generated points, shape `(n, 2)` |

`model(x_t, t)` always receives `t` as a `(B,)` integer tensor and predicts the noise.

## Check your work

```bash
python -m feedback diffusion path/to/your_notebook.ipynb
```
