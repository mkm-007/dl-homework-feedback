"""Known VAE mistakes as small overrides of the reference solution, plus correct alternatives.

Each mutation replaces one or two names in the reference namespace. `expect` is the mistake label the
checks should report. Correct variants are different but valid implementations that must pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence as dist_kl

from reference import vae as ref


def kl_missing_half(mu, logvar):
    return (mu ** 2 + logvar.exp() - logvar - 1).sum(1)


def kl_sign(mu, logvar):
    return -0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)


def kl_mean_over_latent(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).mean(1)


def kl_logvar_as_logstd(mu, logvar):
    return 0.5 * (mu ** 2 + (2 * logvar).exp() - 2 * logvar - 1).sum(1)


def kl_batch_mean(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1).mean()


def reparam_var_as_std(mu, logvar):
    return mu + logvar.exp() * torch.randn_like(mu)


def reparam_no_noise(mu, logvar):
    return mu


def reparam_torch_normal(mu, logvar):
    return torch.normal(mu, torch.exp(0.5 * logvar))


def reparam_detached(mu, logvar):
    return (mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)).detach()


def recon_mean_over_pixels(logits, x):
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").mean((1, 2, 3))


def recon_double_sigmoid(logits, x):
    return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), x, reduction="none").sum((1, 2, 3))


def recon_batch_mean(logits, x):
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum((1, 2, 3)).mean()


def recon_expects_probs(logits, x):
    return F.binary_cross_entropy(logits, x, reduction="none").sum((1, 2, 3))


class SigmoidDecoder(ref.Decoder):
    def forward(self, z, y):
        return torch.sigmoid(super().forward(z, y))


class DecoderIgnoresLabel(ref.Decoder):
    def forward(self, z, y):
        return super().forward(z, torch.zeros_like(y))


class EncoderIgnoresLabel(ref.Encoder):
    def forward(self, x, y):
        return super().forward(x, torch.zeros_like(y))


def elbo_beta_on_recon(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    z = ref.reparameterize(mu, logvar)
    return (beta * ref.reconstruction_loss(decoder(z, y), x) + ref.kl_divergence(mu, logvar)).mean()


def elbo_sum_over_batch(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    z = ref.reparameterize(mu, logvar)
    return (ref.reconstruction_loss(decoder(z, y), x) + beta * ref.kl_divergence(mu, logvar)).sum()


def elbo_ignores_beta(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    z = ref.reparameterize(mu, logvar)
    return (ref.reconstruction_loss(decoder(z, y), x) + ref.kl_divergence(mu, logvar)).mean()


def sample_returns_logits(decoder, y):
    return decoder(torch.randn(len(y), ref.Z_DIM), y)


def sample_uniform_prior(decoder, y):
    return torch.sigmoid(decoder(torch.rand(len(y), ref.Z_DIM), y))


MUTATIONS = {
    "kl_missing_half": ({"kl_divergence": kl_missing_half}, "kl_missing_half"),
    "kl_sign": ({"kl_divergence": kl_sign}, "kl_sign"),
    "kl_mean_over_latent": ({"kl_divergence": kl_mean_over_latent}, "kl_mean_over_latent"),
    "kl_logvar_as_logstd": ({"kl_divergence": kl_logvar_as_logstd}, "kl_logvar_as_logstd"),
    "kl_batch_mean": ({"kl_divergence": kl_batch_mean}, "kl_reduced_over_batch"),
    "reparam_var_as_std": ({"reparameterize": reparam_var_as_std}, "reparam_var_as_std"),
    "reparam_no_noise": ({"reparameterize": reparam_no_noise}, "reparam_no_noise"),
    "reparam_torch_normal": ({"reparameterize": reparam_torch_normal}, "reparam_no_gradient"),
    "reparam_detached": ({"reparameterize": reparam_detached}, "reparam_no_gradient"),
    "recon_mean_over_pixels": ({"reconstruction_loss": recon_mean_over_pixels}, "recon_mean_over_pixels"),
    "recon_double_sigmoid": ({"reconstruction_loss": recon_double_sigmoid}, "double_sigmoid"),
    "recon_batch_mean": ({"reconstruction_loss": recon_batch_mean}, "recon_reduced_over_batch"),
    "recon_expects_probs": ({"reconstruction_loss": recon_expects_probs}, "recon_expects_probs"),
    "decoder_sigmoid": ({"Decoder": SigmoidDecoder}, "decoder_outputs_probs"),
    "decoder_ignores_label": ({"Decoder": DecoderIgnoresLabel}, "decoder_ignores_label"),
    "encoder_ignores_label": ({"Encoder": EncoderIgnoresLabel}, "encoder_ignores_label"),
    "elbo_beta_on_recon": ({"elbo_loss": elbo_beta_on_recon}, "beta_on_reconstruction"),
    "elbo_sum_over_batch": ({"elbo_loss": elbo_sum_over_batch}, "elbo_sum_over_batch"),
    "elbo_ignores_beta": ({"elbo_loss": elbo_ignores_beta}, "beta_ignored"),
    "sample_returns_logits": ({"sample": sample_returns_logits}, "sample_returns_logits"),
    "sample_uniform_prior": ({"sample": sample_uniform_prior}, "sample_wrong_prior"),
}


def kl_std_for_var(mu, logvar):
    return 0.5 * (mu ** 2 + torch.exp(0.5 * logvar) - 0.5 * logvar - 1).sum(1)


def recon_mse(logits, x):
    return ((torch.sigmoid(logits) - x) ** 2).sum((1, 2, 3))


def recon_swapped_args(logits, x):
    return F.binary_cross_entropy_with_logits(x, torch.sigmoid(logits), reduction="none").sum((1, 2, 3))


def elbo_uses_mu(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    return (ref.reconstruction_loss(decoder(mu, y), x) + beta * ref.kl_divergence(mu, logvar)).mean()


def elbo_negated(x, y, encoder, decoder, beta=1.0):
    return -ref.elbo_loss(x, y, encoder, decoder, beta)


def elbo_kl_on_z(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    z = ref.reparameterize(mu, logvar)
    return (ref.reconstruction_loss(decoder(z, y), x) + beta * 0.5 * (z ** 2).sum(1)).mean()


def sample_fixed_z(decoder, y):
    return torch.sigmoid(decoder(torch.zeros(len(y), ref.Z_DIM), y))


class EncoderSharedHead(ref.Encoder):
    def forward(self, x, y):
        mu, _ = super().forward(x, y)
        return mu, mu


HELD_OUT = {
    "kl_std_for_var": {"kl_divergence": kl_std_for_var},
    "recon_mse": {"reconstruction_loss": recon_mse},
    "recon_swapped_args": {"reconstruction_loss": recon_swapped_args},
    "elbo_uses_mu": {"elbo_loss": elbo_uses_mu},
    "elbo_negated": {"elbo_loss": elbo_negated},
    "elbo_kl_on_z": {"elbo_loss": elbo_kl_on_z},
    "sample_fixed_z": {"sample": sample_fixed_z},
    "encoder_mu_as_logvar": {"Encoder": EncoderSharedHead},
}


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU())
        self.label = nn.Embedding(ref.N_CLASSES, 32)
        self.head = nn.Linear(32 * 7 * 7 + 32, 2 * ref.Z_DIM)

    def forward(self, x, y):
        mu, logvar = self.head(torch.cat([self.conv(x).flatten(1), self.label(y)], 1)).chunk(2, 1)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = nn.Embedding(ref.N_CLASSES, 32)
        self.fc = nn.Linear(ref.Z_DIM + 32, 32 * 7 * 7)
        self.deconv = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, z, y):
        return self.deconv(self.fc(torch.cat([z, self.label(y)], 1)).view(-1, 32, 7, 7))


def kl_distributions(mu, logvar):
    q = Normal(mu, torch.exp(0.5 * logvar))
    return dist_kl(q, Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(1)


def reparam_rsample(mu, logvar):
    return Normal(mu, torch.exp(0.5 * logvar)).rsample()


def recon_manual(logits, x):
    p = torch.sigmoid(logits)
    eps = 1e-7
    return -(x * torch.log(p + eps) + (1 - x) * torch.log(1 - p + eps)).flatten(1).sum(1)


def elbo_inline(x, y, encoder, decoder, beta=1.0):
    mu, logvar = encoder(x, y)
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)
    recon = F.binary_cross_entropy_with_logits(decoder(z, y), x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)
    return recon + beta * kl


CORRECT = {
    "conv_networks": {"Encoder": ConvEncoder, "Decoder": ConvDecoder},
    "kl_torch_distributions": {"kl_divergence": kl_distributions},
    "reparam_rsample": {"reparameterize": reparam_rsample},
    "recon_manual_bce": {"reconstruction_loss": recon_manual},
    "elbo_inline_sums": {"elbo_loss": elbo_inline},
}


def namespace(overrides: dict | None = None) -> dict:
    ns = {k: v for k, v in vars(ref).items() if not k.startswith("__")}
    ns.update(overrides or {})
    return ns
