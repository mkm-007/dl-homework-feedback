"""Known diffusion mistakes as overrides of the reference solution, plus correct alternatives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from reference import diffusion as ref


def schedule_reversed(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_end, beta_start, T)


def schedule_length(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T + 1)


def schedule_hardcoded(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(1e-4, 0.02, 1000)


def alpha_bar_cumprod_beta(betas):
    return torch.cumprod(betas, 0)


def alpha_bar_no_cumprod(betas):
    return 1 - betas


def alpha_bar_off_by_one(betas):
    return torch.cat([torch.ones(1), torch.cumprod(1 - betas, 0)[:-1]])


def alpha_bar_sqrt(betas):
    return torch.cumprod(1 - betas, 0).sqrt()


def q_no_sqrt(x0, t, noise, alpha_bar):
    a = alpha_bar[t][:, None]
    return a * x0 + (1 - a) * noise


def q_swapped(x0, t, noise, alpha_bar):
    a = alpha_bar[t][:, None]
    return (1 - a).sqrt() * x0 + a.sqrt() * noise


def q_index_shift(x0, t, noise, alpha_bar):
    a = alpha_bar[t - 1][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def q_single_t(x0, t, noise, alpha_bar):
    a = alpha_bar[t[0]]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def q_broadcast(x0, t, noise, alpha_bar):
    a = alpha_bar[t]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


class DenoiserIgnoresT(ref.Denoiser):
    def forward(self, x, t):
        return super().forward(x, torch.zeros_like(t))


def loss_target_x0(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(ref.q_sample(x0, t, noise, alpha_bar), t), x0)


def loss_uniform_noise(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    noise = torch.rand_like(x0)
    return F.mse_loss(model(ref.q_sample(x0, t, noise, alpha_bar), t), noise)


def loss_sum_over_batch(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(ref.q_sample(x0, t, noise, alpha_bar), t), noise, reduction="sum")


def loss_detached(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    noise = torch.randn_like(x0)
    with torch.no_grad():
        pred = model(ref.q_sample(x0, t, noise, alpha_bar), t)
    return F.mse_loss(pred, noise)


def loss_t_out_of_range(model, x0, alpha_bar):
    t = torch.randint(1, len(alpha_bar) + 1, (len(x0),))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(ref.q_sample(x0, t, noise, alpha_bar), t), noise)


def loss_single_t(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (1,)).repeat(len(x0))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(ref.q_sample(x0, t, noise, alpha_bar), t), noise)


def _p_step(model, x_t, t, betas, alpha_bar, scale, coef, noise_std, noise_at_zero=False):
    eps = model(x_t, torch.full((len(x_t),), t, dtype=torch.long))
    mean = (x_t - coef * eps) * scale
    if t == 0 and not noise_at_zero:
        return mean
    return mean + noise_std * torch.randn_like(x_t)


def p_noise_at_t0(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a).sqrt(), b.sqrt(), True)


def p_no_rescale(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1.0, b / (1 - a).sqrt(), b.sqrt())


def p_alpha_bar_rescale(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / a.sqrt(), b / (1 - a).sqrt(), b.sqrt())


def p_coef_no_sqrt(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a), b.sqrt())


def p_sign(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), -b / (1 - a).sqrt(), b.sqrt())


def p_var_as_std(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a).sqrt(), b)


def p_no_noise(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a).sqrt(), 0.0)


@torch.no_grad()
def sample_wrong_order(model, n, betas, alpha_bar):
    x = torch.randn(n, 2)
    for t in range(len(betas)):
        x = ref.p_sample(model, x, t, betas, alpha_bar)
    return x


@torch.no_grad()
def sample_skips_last(model, n, betas, alpha_bar):
    x = torch.randn(n, 2)
    for t in range(len(betas) - 1, 0, -1):
        x = ref.p_sample(model, x, t, betas, alpha_bar)
    return x


@torch.no_grad()
def sample_zero_start(model, n, betas, alpha_bar):
    x = torch.zeros(n, 2)
    for t in reversed(range(len(betas))):
        x = ref.p_sample(model, x, t, betas, alpha_bar)
    return x


MUTATIONS = {
    "schedule_reversed": ({"linear_beta_schedule": schedule_reversed}, "schedule_reversed"),
    "schedule_length": ({"linear_beta_schedule": schedule_length}, "schedule_length"),
    "schedule_hardcoded": ({"linear_beta_schedule": schedule_hardcoded}, "schedule_ignores_args"),
    "alpha_bar_cumprod_beta": ({"compute_alpha_bar": alpha_bar_cumprod_beta}, "alpha_bar_cumprod_beta"),
    "alpha_bar_no_cumprod": ({"compute_alpha_bar": alpha_bar_no_cumprod}, "alpha_bar_no_cumprod"),
    "alpha_bar_off_by_one": ({"compute_alpha_bar": alpha_bar_off_by_one}, "alpha_bar_off_by_one"),
    "alpha_bar_sqrt": ({"compute_alpha_bar": alpha_bar_sqrt}, "alpha_bar_sqrt"),
    "q_no_sqrt": ({"q_sample": q_no_sqrt}, "q_no_sqrt"),
    "q_swapped": ({"q_sample": q_swapped}, "q_swapped"),
    "q_index_shift": ({"q_sample": q_index_shift}, "q_index_shift"),
    "q_single_t": ({"q_sample": q_single_t}, "q_single_t"),
    "q_broadcast": ({"q_sample": q_broadcast}, "q_broadcast"),
    "denoiser_ignores_t": ({"Denoiser": DenoiserIgnoresT}, "denoiser_ignores_t"),
    "loss_target_x0": ({"ddpm_loss": loss_target_x0}, "loss_target_x0"),
    "loss_uniform_noise": ({"ddpm_loss": loss_uniform_noise}, "loss_uniform_noise"),
    "loss_sum_over_batch": ({"ddpm_loss": loss_sum_over_batch}, "loss_sum_over_batch"),
    "loss_detached": ({"ddpm_loss": loss_detached}, "loss_detached"),
    "loss_t_out_of_range": ({"ddpm_loss": loss_t_out_of_range}, "loss_t_out_of_range"),
    "loss_single_t": ({"ddpm_loss": loss_single_t}, "loss_single_t"),
    "p_noise_at_t0": ({"p_sample": p_noise_at_t0}, "p_noise_at_t0"),
    "p_no_rescale": ({"p_sample": p_no_rescale}, "p_no_rescale"),
    "p_alpha_bar_rescale": ({"p_sample": p_alpha_bar_rescale}, "p_alpha_bar_rescale"),
    "p_coef_no_sqrt": ({"p_sample": p_coef_no_sqrt}, "p_coef_no_sqrt"),
    "p_sign": ({"p_sample": p_sign}, "p_sign"),
    "p_var_as_std": ({"p_sample": p_var_as_std}, "p_var_as_std"),
    "p_no_noise": ({"p_sample": p_no_noise}, "p_no_noise"),
    "sample_wrong_order": ({"sample": sample_wrong_order}, "sample_wrong_order"),
    "sample_skips_last": ({"sample": sample_skips_last}, "sample_skips_last"),
    "sample_zero_start": ({"sample": sample_zero_start}, "sample_bad_start"),
}


def schedule_arange(T, beta_start=1e-4, beta_end=0.02):
    return beta_start + (beta_end - beta_start) * torch.arange(T) / (T - 1)


def alpha_bar_logspace(betas):
    return torch.exp(torch.cumsum(torch.log1p(-betas), 0))


def q_gather(x0, t, noise, alpha_bar):
    a = alpha_bar.gather(0, t).unsqueeze(-1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


class TimeScalarDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t.float()[:, None] / 1000], 1))


def loss_manual_sum_dims(model, x0, alpha_bar):
    t = torch.randint(len(alpha_bar), size=(x0.shape[0],))
    noise = torch.randn(x0.shape)
    pred = model(ref.q_sample(x0, t, noise, alpha_bar), t)
    return ((pred - noise) ** 2).sum(1).mean()


def p_posterior_variance(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    eps = model(x_t, torch.full((x_t.shape[0],), t, dtype=torch.long))
    mean = (x_t - b / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b)
    if t > 0:
        var = (1 - alpha_bar[t - 1]) / (1 - a) * b
        mean = mean + torch.sqrt(var) * torch.randn_like(x_t)
    return mean


@torch.no_grad()
def sample_while(model, n, betas, alpha_bar):
    x = torch.randn(n, 2)
    t = len(betas) - 1
    while t >= 0:
        x = ref.p_sample(model, x, t, betas, alpha_bar)
        t -= 1
    return x


CORRECT = {
    "schedule_arange": {"linear_beta_schedule": schedule_arange},
    "alpha_bar_logspace": {"compute_alpha_bar": alpha_bar_logspace},
    "q_gather": {"q_sample": q_gather},
    "time_scalar_denoiser": {"Denoiser": TimeScalarDenoiser},
    "loss_sum_over_dims": {"ddpm_loss": loss_manual_sum_dims},
    "p_posterior_variance": {"p_sample": p_posterior_variance},
    "sample_while_loop": {"sample": sample_while},
}

def q_partial_sqrt(x0, t, noise, alpha_bar):
    a = alpha_bar[t][:, None]
    return a.sqrt() * x0 + (1 - a) * noise


def q_ignores_noise(x0, t, noise, alpha_bar):
    return ref.q_sample(x0, t, torch.randn_like(x0), alpha_bar)


def loss_fresh_noise_target(model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (len(x0),))
    x_t = ref.q_sample(x0, t, torch.randn_like(x0), alpha_bar)
    return F.mse_loss(model(x_t, t), torch.randn_like(x0))


def p_index_shift(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t - 1] if t > 0 else torch.tensor(1.0 - 1e-8)
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a).sqrt(), b.sqrt())


def p_shared_noise(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    eps = model(x_t, torch.full((len(x_t),), t, dtype=torch.long))
    mean = (x_t - b / (1 - a).sqrt() * eps) / (1 - b).sqrt()
    return mean if t == 0 else mean + b.sqrt() * torch.randn(1, x_t.shape[1])


def p_wrong_sigma(model, x_t, t, betas, alpha_bar):
    b, a = betas[t], alpha_bar[t]
    return _p_step(model, x_t, t, betas, alpha_bar, 1 / (1 - b).sqrt(), b / (1 - a).sqrt(), (1 - a).sqrt())


@torch.no_grad()
def sample_clamps(model, n, betas, alpha_bar):
    return ref.sample(model, n, betas, alpha_bar).clamp(-1, 1)


class DenoiserTanh(ref.Denoiser):
    def forward(self, x, t):
        return torch.tanh(super().forward(x, t))


HELD_OUT = {
    "q_partial_sqrt": {"q_sample": q_partial_sqrt},
    "q_ignores_noise": {"q_sample": q_ignores_noise},
    "loss_fresh_noise_target": {"ddpm_loss": loss_fresh_noise_target},
    "p_index_shift": {"p_sample": p_index_shift},
    "p_shared_noise": {"p_sample": p_shared_noise},
    "p_wrong_sigma": {"p_sample": p_wrong_sigma},
    "sample_clamps": {"sample": sample_clamps},
    "denoiser_tanh": {"Denoiser": DenoiserTanh},
}


def namespace(overrides: dict | None = None) -> dict:
    ns = {k: v for k, v in vars(ref).items() if not k.startswith("__")}
    ns.update(overrides or {})
    return ns
