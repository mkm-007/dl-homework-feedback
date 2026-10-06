"""Checks for the 2D diffusion assignment (assignments/diffusion/README.md)."""

from __future__ import annotations

import torch
from torch import nn

from feedback.core import Check, Fail

LEARN = ("Ho, Jain and Abbeel, Denoising Diffusion Probabilistic Models (arXiv:2006.11239), "
         "Section 3 and Algorithms 1 and 2")


def _schedule(T=1000, start=1e-4, end=0.02):
    betas = torch.linspace(start, end, T)
    return betas, torch.cumprod(1 - betas, 0)


def _close(a, b, rtol=1e-3, atol=1e-5) -> bool:
    return isinstance(a, torch.Tensor) and a.shape == b.shape and torch.allclose(a.float(), b, rtol=rtol, atol=atol)


def check_schedule(ns):
    f = ns["linear_beta_schedule"]
    out = f(1000)
    ref = torch.linspace(1e-4, 0.02, 1000)
    if _close(out, ref):
        other = f(50, 1e-3, 0.05)
        if not _close(other, torch.linspace(1e-3, 0.05, 50)):
            raise Fail("Your schedule ignores its arguments: linear_beta_schedule(50, 1e-3, 0.05) should go from "
                       "1e-3 to 0.05 in 50 steps.", "schedule_ignores_args")
        return
    if isinstance(out, torch.Tensor) and len(out) != 1000:
        raise Fail(f"The schedule should have one beta per timestep, {1000} values for T = 1000; yours has "
                   f"{len(out)}.", "schedule_length")
    if _close(out, ref.flip(0)):
        raise Fail("Your betas decrease over time. Noise is added slowly at first and faster later, so beta "
                   "grows from beta_start to beta_end.", "schedule_reversed")
    raise Fail("linear_beta_schedule should return T evenly spaced values from beta_start to beta_end.",
               "schedule_wrong")


def check_alpha_bar(ns):
    betas = torch.linspace(0.01, 0.2, 20)
    ref = torch.cumprod(1 - betas, 0)
    out = ns["compute_alpha_bar"](betas)
    if _close(out, ref):
        return
    wrong = {
        "alpha_bar_cumprod_beta": (torch.cumprod(betas, 0), "You multiply the betas together. alpha_bar is the "
                                   "product of alphas, where alpha = 1 - beta is the fraction of signal kept "
                                   "at each step."),
        "alpha_bar_no_cumprod": (1 - betas, "You return alpha = 1 - beta for each step. alpha_bar[t] is the "
                                 "cumulative product up to t: how much signal survives all steps so far."),
        "alpha_bar_off_by_one": (torch.cat([torch.ones(1), ref[:-1]]), "Your alpha_bar is shifted by one step: "
                                 "alpha_bar[0] should already include the first beta, so alpha_bar[0] = 1 - "
                                 "betas[0], not 1."),
        "alpha_bar_sqrt": (ref.sqrt(), "You return the square root of alpha_bar. Keep alpha_bar itself and take "
                           "square roots where the formulas need them; mixing the two is a common source of "
                           "silent errors."),
    }
    for mistake, (value, message) in wrong.items():
        if _close(out, value):
            raise Fail(message, mistake)
    raise Fail("compute_alpha_bar should return the cumulative product of (1 - beta).", "alpha_bar_wrong")


def check_q_sample(ns):
    torch.manual_seed(10)
    _, ab = _schedule(20, 0.01, 0.2)
    x0, noise = torch.randn(64, 2), torch.randn(64, 2)
    t = torch.randint(0, 20, (64,))
    t[0] = 0
    a = ab[t][:, None]
    ref = a.sqrt() * x0 + (1 - a).sqrt() * noise
    try:
        out = ns["q_sample"](x0, t, noise, ab)
    except RuntimeError as e:
        if "must match the size" in str(e):
            raise Fail("alpha_bar[t] has shape (B,) but x0 has shape (B, 2), so they cannot be multiplied. "
                       "Reshape the coefficient to (B, 1) so each example gets its own noise level.",
                       "q_broadcast") from e
        raise
    if _close(out, ref):
        return
    shifted = ab[t - 1][:, None]
    wrong = {
        "q_no_sqrt": (a * x0 + (1 - a) * noise, "You use alpha_bar and 1 - alpha_bar as the coefficients. They "
                      "are variances; the coefficients are their square roots, so that x_t keeps unit variance."),
        "q_swapped": ((1 - a).sqrt() * x0 + a.sqrt() * noise, "The coefficients are swapped: at small t, x_t "
                      "should be almost x0, so x0 gets sqrt(alpha_bar), which is close to 1 early on."),
        "q_index_shift": (shifted.sqrt() * x0 + (1 - shifted).sqrt() * noise, "You use alpha_bar[t - 1]. With "
                          "timesteps 0 to T - 1, the noise level of x_t is alpha_bar[t]. At t = 0, index -1 "
                          "silently wraps to the last step."),
        "q_single_t": (ab[t[0]].sqrt() * x0 + (1 - ab[t[0]]).sqrt() * noise, "Every example gets the noise "
                       "level of the first timestep. Each example has its own t."),
    }
    for mistake, (value, message) in wrong.items():
        if _close(out, value):
            raise Fail(message, mistake)
    raise Fail("q_sample does not produce x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise.",
               "q_wrong")


def check_denoiser(ns):
    torch.manual_seed(11)
    model = ns["Denoiser"]()
    x = torch.randn(8, 2)
    out = model(x, torch.full((8,), 10))
    if out.shape != (8, 2):
        raise Fail(f"Denoiser should return the predicted noise with the shape of x_t, (8, 2); got "
                   f"{tuple(out.shape)}.", "denoiser_shape")
    if torch.allclose(out, model(x, torch.full((8,), 900)), atol=1e-6):
        raise Fail("The denoiser ignores t. The same point needs a different correction at t = 10 (almost "
                   "clean) than at t = 900 (almost pure noise).", "denoiser_ignores_t")


class _Oracle(nn.Module):
    """Predicts the true noise of a known x0 batch, plus an offset; records what it receives."""

    def __init__(self, x0, alpha_bar, offset=0.0):
        super().__init__()
        self.x0, self.ab, self.offset = x0, alpha_bar, offset
        self.p = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x_t, t):
        self.calls.append((x_t.detach(), t))
        if not isinstance(t, torch.Tensor) or t.dtype.is_floating_point or t.shape != (len(x_t),):
            raise Fail("The model should receive t as a (B,) tensor of integer timesteps.", "loss_t_format")
        a = self.ab[t][:, None]
        noise = (x_t - a.sqrt() * self.x0) / (1 - a).sqrt()
        return noise + self.offset + self.p


def check_loss(ns):
    torch.manual_seed(12)
    _, ab = _schedule()
    x0 = 0.5 * torch.randn(4096, 2) + 1.0
    oracle = _Oracle(x0, ab)
    try:
        loss = ns["ddpm_loss"](oracle, x0, ab)
    except IndexError as e:
        raise Fail("A timestep is out of range. Timesteps run from 0 to T - 1; torch.randint(low, high) "
                   "excludes high.", "loss_t_out_of_range") from e
    if not oracle.calls:
        raise Fail("ddpm_loss does not call the model you pass in.", "loss_ignores_model")
    x_t, t = oracle.calls[-1]
    if t.unique().numel() == 1:
        raise Fail("The whole batch shares one timestep. Draw a timestep per example so each batch covers "
                   "many noise levels and the gradient is less noisy.", "loss_single_t")
    if t.min() > 10 or t.max() < len(ab) - 11:
        raise Fail(f"Timesteps should cover 0 to T - 1; yours range from {t.min().item()} to {t.max().item()}.",
                   "loss_t_range")
    if not torch.is_tensor(loss) or loss.dim() != 0:
        raise Fail("ddpm_loss should return a single scalar.", "loss_not_scalar")
    a = ab[t][:, None]
    noise = (x_t - a.sqrt() * x0) / (1 - a).sqrt()
    if abs(noise.mean().item() - 0.5) < 0.05 and noise.std().item() < 0.4:
        raise Fail("Your noise is uniform on [0, 1]. The forward process adds Gaussian noise: use "
                   "torch.randn_like, not torch.rand_like.", "loss_uniform_noise")
    if abs(noise.mean().item()) > 0.05 or abs(noise.std().item() - 1) > 0.05:
        raise Fail(f"The noise in x_t should be standard Gaussian; yours has mean {noise.mean().item():.2f} and "
                   f"standard deviation {noise.std().item():.2f}.", "loss_noise_not_standard")
    if loss.item() > 1e-3:
        if abs(loss.item() - ((noise - x0) ** 2).mean().item()) < 1e-3 * max(1.0, loss.item()):
            raise Fail("Your loss compares the prediction with x0. This model predicts the noise that was "
                       "added, so the target is the noise.", "loss_target_x0")
        raise Fail("With a model that predicts the noise perfectly, your loss should be 0; it is "
                   f"{loss.item():.3f}. Check that the model's target is the same noise used to make x_t.",
                   "loss_wrong_target")
    oracle = _Oracle(x0, ab, offset=1.0)
    loss = ns["ddpm_loss"](oracle, x0, ab)
    if abs(loss.item() - len(x0)) < 0.05 * len(x0) or abs(loss.item() - 2 * len(x0)) < 0.1 * len(x0):
        raise Fail("Your loss sums over the batch, so its size and the effective learning rate change with "
                   "batch size. Average over examples.", "loss_sum_over_batch")
    if not (abs(loss.item() - 1) < 0.01 or abs(loss.item() - 2) < 0.02):
        raise Fail(f"With every prediction off by exactly 1, the mean squared error should be 1; yours is "
                   f"{loss.item():.3f}.", "loss_scale")
    if not loss.requires_grad:
        raise Fail("The loss is cut off from the model's parameters, so training cannot change them. Check for "
                   "torch.no_grad() or .detach() around the model call.", "loss_detached")
    loss.backward()
    if oracle.p.grad is None or oracle.p.grad.abs() < 1e-6:
        raise Fail("No gradient reaches the model's parameters.", "loss_detached")


class _Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts = []

    def forward(self, x_t, t):
        self.ts.append(t)
        return 0.3 * x_t + 0.1


def check_p_sample(ns):
    betas, ab = _schedule()
    f = ns["p_sample"]
    x = torch.tensor([1.0, -0.5]).repeat(1000, 1)

    def mean_for(a_scale, coef):
        eps = 0.3 * x[:1] + 0.1
        return ((x[:1] - coef * eps) * a_scale)[0]

    model = _Affine()
    torch.manual_seed(13)
    first = f(model, x, 0, betas, ab)
    if not model.ts or not torch.is_tensor(model.ts[0]) or model.ts[0].shape != (len(x),):
        raise Fail("p_sample should call the model with t as a (B,) integer tensor, for example "
                   "torch.full((B,), t).", "p_t_format")
    torch.manual_seed(14)
    second = f(model, x, 0, betas, ab)
    if not torch.allclose(first, second):
        raise Fail("The last step (t = 0) adds noise. The final step returns the mean; adding noise there "
                   "only blurs the samples.", "p_noise_at_t0")

    n = 40000
    for t in (900, 20):
        xs = torch.tensor([1.0, -0.5]).repeat(n, 1)
        torch.manual_seed(15 + t)
        out = f(_Affine(), xs, t, betas, ab)
        b, a = betas[t], ab[t]
        candidates = {
            "": mean_for(1 / (1 - b).sqrt(), b / (1 - a).sqrt()),
            "p_no_rescale": mean_for(1.0, b / (1 - a).sqrt()),
            "p_alpha_bar_rescale": mean_for(1 / a.sqrt(), b / (1 - a).sqrt()),
            "p_coef_no_sqrt": mean_for(1 / (1 - b).sqrt(), b / (1 - a)),
            "p_sign": mean_for(1 / (1 - b).sqrt(), -b / (1 - a).sqrt()),
        }
        m = out.mean(0)
        resid_std = (out - candidates[""]).std().item()
        tol = 6 * max(resid_std, 1e-4) / n ** 0.5 + 1e-5
        matches = [k for k, v in candidates.items() if (m - v).abs().max() < tol]
        messages = {
            "p_no_rescale": "Your reverse step is missing the 1 / sqrt(alpha_t) factor that undoes the shrinking "
                            "of the forward step.",
            "p_alpha_bar_rescale": "You rescale by 1 / sqrt(alpha_bar[t]). One reverse step undoes one forward "
                                   "step, so the factor is 1 / sqrt(alpha_t) = 1 / sqrt(1 - beta_t).",
            "p_coef_no_sqrt": "The noise coefficient should be beta_t / sqrt(1 - alpha_bar[t]); the square root "
                              "is missing. The error is largest at small t, where the final details form.",
            "p_sign": "You add the predicted noise instead of subtracting it.",
        }
        if "" not in matches:
            if matches:
                raise Fail(messages[matches[0]], matches[0])
            raise Fail(f"At t = {t} the average of your reverse step does not match the DDPM mean "
                       "(x_t - beta_t / sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_t).", "p_mean_wrong")
        prev = ab[t - 1]
        sigmas = (b.sqrt().item(), ((1 - prev) / (1 - a) * b).sqrt().item())
        if resid_std < 0.1 * sigmas[1]:
            raise Fail("The reverse step adds no noise for t > 0, so sampling is deterministic and collapses "
                       "toward the mean.", "p_no_noise")
        if abs(resid_std - b.item()) < 0.1 * b.item():
            raise Fail("The added noise has standard deviation beta_t, but beta_t is a variance. Scale the noise "
                       "by sqrt(beta_t).", "p_var_as_std")
        if not (0.95 * sigmas[1] <= resid_std <= 1.05 * sigmas[0]):
            raise Fail(f"At t = {t} the added noise has standard deviation {resid_std:.4f}; expected "
                       f"sqrt(beta_t) = {sigmas[0]:.4f}.", "p_noise_scale")


class _Spy(nn.Module):
    """Exact noise for data concentrated at one point; records each step."""

    def __init__(self, point, alpha_bar):
        super().__init__()
        self.point, self.ab, self.steps, self.starts = point, alpha_bar, [], []

    def forward(self, x_t, t):
        if not self.steps:
            self.starts.append(x_t.clone())
        self.steps.append(int(t[0]))
        a = self.ab[t][:, None]
        return (x_t - a.sqrt() * self.point) / (1 - a).sqrt()


def check_sample(ns):
    torch.manual_seed(16)
    T = 50
    betas, ab = _schedule(T, 1e-4, 0.2)
    point = torch.tensor([2.0, -1.0])
    spy = _Spy(point, ab)
    out = ns["sample"](spy, 2000, betas, ab)
    if not torch.is_tensor(out) or out.shape != (2000, 2):
        raise Fail(f"sample should return n points of shape (n, 2); got {getattr(out, 'shape', type(out))}.",
                   "sample_shape")
    if not spy.steps:
        raise Fail("sample does not call the model you pass in.", "sample_ignores_model")
    expected = list(range(T - 1, -1, -1))
    if spy.steps == expected[::-1]:
        raise Fail("You run the timesteps from 0 up to T - 1. Generation reverses the forward process: start "
                   "at T - 1 (pure noise) and end at 0.", "sample_wrong_order")
    if spy.steps == expected[:-1]:
        raise Fail("Sampling stops before t = 0, so the final denoising step never runs.", "sample_skips_last")
    if spy.steps != expected:
        raise Fail(f"sample should visit each timestep from T - 1 down to 0 once; it visited {len(spy.steps)} "
                   f"steps starting with {spy.steps[:3]}.", "sample_wrong_steps")
    start = spy.starts[0]
    if abs(start.mean().item()) > 0.1 or abs(start.std().item() - 1) > 0.1:
        raise Fail(f"Generation should start from pure noise, N(0, I). Your starting points have mean "
                   f"{start.mean().item():.2f} and standard deviation {start.std().item():.2f}.", "sample_bad_start")
    err = (out - point).norm(dim=1).mean().item()
    if err > 0.05:
        raise Fail(f"With a perfect noise predictor for data at a single point, sampling should end at that "
                   f"point; your samples end {err:.2f} away on average.", "sample_misses_data")


def check_training(ns):
    torch.manual_seed(17)
    _, ab = _schedule()
    angles = torch.rand(512) * 2 * torch.pi
    data = torch.stack([angles.cos(), angles.sin()], 1) * 2
    model = ns["Denoiser"]()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(300):
        loss = ns["ddpm_loss"](model, data[torch.randint(0, 512, (256,))], ab)
        if not torch.isfinite(loss):
            raise Fail(f"The loss became {loss.item()} at step {step}.", "training_nan")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:20]) / 20, sum(losses[-50:]) / 50
    if last > 0.9 * first:
        raise Fail(f"The loss barely decreases in a short training run ({first:.3f} to {last:.3f}).",
                   "training_stalls")


CHECKS = [
    Check("schedule", "Noise schedule", ("linear_beta_schedule",), check_schedule, LEARN),
    Check("alpha_bar", "Cumulative signal alpha_bar", ("compute_alpha_bar",), check_alpha_bar, LEARN),
    Check("q_sample", "Forward process q(x_t | x_0)", ("q_sample",), check_q_sample, LEARN),
    Check("denoiser", "Denoiser shape and use of t", ("Denoiser",), check_denoiser),
    Check("loss", "Training loss: target, timesteps, reduction", ("ddpm_loss",), check_loss, LEARN),
    Check("p_sample", "Reverse step p(x_{t-1} | x_t)", ("p_sample",), check_p_sample, LEARN),
    Check("sample", "Sampling loop", ("sample",), check_sample, LEARN),
    Check("training", "A short training run improves the loss", ("Denoiser", "ddpm_loss"), check_training),
]
