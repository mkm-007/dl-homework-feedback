"""Checks for the conditional VAE assignment (assignments/vae/README.md)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from feedback.core import Check, Fail

LEARN = ("Kingma and Welling, Auto-Encoding Variational Bayes (arXiv:1312.6114), "
         "Section 2.4 and Appendix B")


def _close(a, b, rtol=1e-3, atol=1e-4) -> bool:
    return a.shape == b.shape and torch.allclose(a, b, rtol=rtol, atol=atol)


def _data(n=64, seed=0):
    """Binary 28x28 images with a class-dependent bar pattern, plus labels."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.zeros(n, 1, 28, 28)
    for i, label in enumerate(y.tolist()):
        x[i, 0, 2 + 2 * label: 4 + 2 * label, 4:24] = 1.0
        x[i, 0, 4:24, 2 + 2 * label: 4 + 2 * label] = 1.0
    flip = torch.rand(x.shape, generator=g) < 0.02
    return torch.where(flip, 1 - x, x), y


def check_shapes(ns):
    torch.manual_seed(0)
    x, y = _data(8)
    z_dim = ns["Z_DIM"]
    mu, logvar = ns["Encoder"]()(x, y)
    if mu.shape != (8, z_dim) or logvar.shape != (8, z_dim):
        raise Fail(f"Encoder should return mu and logvar of shape (B, Z_DIM) = (8, {z_dim}); "
                   f"got {tuple(mu.shape)} and {tuple(logvar.shape)}.", "encoder_shape")
    out = ns["Decoder"]()(torch.randn(8, z_dim), y)
    if out.shape != (8, 1, 28, 28):
        raise Fail(f"Decoder should return shape (B, 1, 28, 28) = (8, 1, 28, 28); got {tuple(out.shape)}.",
                   "decoder_shape")


def check_kl(ns):
    torch.manual_seed(1)
    mu, logvar = torch.randn(64, 4), torch.randn(64, 4)
    ref = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)
    out = ns["kl_divergence"](mu, logvar)
    if _close(out, ref):
        return
    if out.dim() == 0:
        if torch.isclose(out, ref.mean(), rtol=1e-3) or torch.isclose(out, ref.sum(), rtol=1e-3):
            raise Fail("kl_divergence returns a single number for the whole batch. Return one KL value per "
                       "example (shape (B,)) and average in elbo_loss, so the reduction is in one place.",
                       "kl_reduced_over_batch")
    wrong = {
        "kl_missing_half": (2 * ref, "Your KL is exactly twice the correct value: the factor 1/2 is missing."),
        "kl_sign": (-ref, "Your KL has the wrong sign. A KL divergence is never negative; check the order "
                          "of the terms."),
        "kl_mean_over_latent": (ref / mu.shape[1], "Your KL averages over latent dimensions. The KL of a "
                                                   "diagonal Gaussian is a sum over dimensions."),
        "kl_logvar_as_logstd": (0.5 * (mu ** 2 + (2 * logvar).exp() - 2 * logvar - 1).sum(1),
                                "Your KL treats logvar as log(sigma) instead of log(sigma^2). The variance is "
                                "exp(logvar), not exp(2 * logvar)."),
    }
    for mistake, (value, message) in wrong.items():
        if _close(out, value):
            raise Fail(message, mistake)
    raise Fail("kl_divergence does not match the closed-form KL between N(mu, exp(logvar)) and N(0, I), "
               f"expected shape (B,) = ({mu.shape[0]},), got {tuple(out.shape)}. Re-derive it per dimension and "
               "sum over the latent dimensions.", "kl_wrong")


def check_reparameterize(ns):
    torch.manual_seed(2)
    mu = torch.full((20000, 2), 1.5, requires_grad=True)
    logvar = torch.full((20000, 2), math.log(0.25), requires_grad=True)
    z = ns["reparameterize"](mu, logvar)
    if z.shape != mu.shape:
        raise Fail(f"reparameterize should return the shape of mu, {tuple(mu.shape)}; got {tuple(z.shape)}.",
                   "reparam_shape")
    if not z.requires_grad:
        raise Fail("The sample is cut off from the computation graph, so no gradient reaches the encoder. "
                   "Sampling with torch.normal(mu, std) or detaching breaks backpropagation; write the "
                   "sample as mu + std * noise.", "reparam_no_gradient")
    std = z.detach().std().item()
    mean = z.detach().mean().item()
    if std < 0.05:
        raise Fail("Your samples have almost no spread: no noise is being added, so the model is a plain "
                   "autoencoder during training.", "reparam_no_noise")
    if abs(std - 0.25) < 0.03:
        raise Fail("The spread of your samples equals the variance exp(logvar), not the standard deviation. "
                   "The standard deviation is exp(0.5 * logvar).", "reparam_var_as_std")
    if abs(mean - 1.5) > 0.03 or abs(std - 0.5) > 0.03:
        raise Fail(f"With mu = 1.5 and variance 0.25, samples should have mean 1.5 and standard deviation 0.5; "
                   f"yours have mean {mean:.2f} and standard deviation {std:.2f}.", "reparam_wrong")
    z.sum().backward()
    if mu.grad is None or logvar.grad is None or logvar.grad.abs().sum() == 0:
        raise Fail("Gradients do not reach both mu and logvar through your sample.", "reparam_no_gradient")


def check_reconstruction(ns):
    torch.manual_seed(3)
    logits = 3 * torch.randn(16, 1, 28, 28)
    x = (torch.rand(16, 1, 28, 28) < 0.3).float()
    ref = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum((1, 2, 3))
    try:
        out = ns["reconstruction_loss"](logits, x)
    except RuntimeError as e:
        if "between 0 and 1" in str(e) or "0 <= " in str(e):
            raise Fail("reconstruction_loss expects probabilities, but the decoder outputs logits. Use the "
                       "logits version of binary cross-entropy.", "recon_expects_probs") from e
        raise
    if _close(out, ref):
        return
    if out.dim() == 0 and (torch.isclose(out, ref.mean(), rtol=1e-3) or torch.isclose(out, ref.sum(), rtol=1e-3)):
        raise Fail("reconstruction_loss returns one number for the batch. Return one value per example.",
                   "recon_reduced_over_batch")
    if _close(out, ref / 784):
        raise Fail("Your reconstruction loss averages over pixels instead of summing. That makes it 784 times "
                   "smaller relative to the KL term, which acts like a very large beta and can collapse the "
                   "latent code.", "recon_mean_over_pixels")
    double = F.binary_cross_entropy_with_logits(torch.sigmoid(logits), x, reduction="none").sum((1, 2, 3))
    if _close(out, double):
        raise Fail("A sigmoid is applied before a loss that already applies one (binary cross-entropy with "
                   "logits). The predictions get squashed twice and reconstructions look washed out.",
                   "double_sigmoid")
    raise Fail("reconstruction_loss does not match binary cross-entropy on logits summed over pixels.",
               "recon_wrong")


def check_decoder_logits(ns):
    torch.manual_seed(4)
    out = ns["Decoder"]()(torch.randn(256, ns["Z_DIM"]), torch.randint(0, 10, (256,)))
    if out.min() >= 0 and out.max() <= 1:
        raise Fail("The decoder's outputs are all between 0 and 1, which suggests a sigmoid at the end. The "
                   "assignment asks for logits; apply the sigmoid only when you need probabilities, as in "
                   "sample().", "decoder_outputs_probs")


def check_conditioning(ns):
    torch.manual_seed(5)
    x, _ = _data(32)
    a, b = torch.zeros(32, dtype=torch.long), torch.full((32,), 7, dtype=torch.long)
    enc, dec = ns["Encoder"](), ns["Decoder"]()
    if torch.allclose(enc(x, a)[0], enc(x, b)[0]):
        raise Fail("The encoder ignores the label: changing y does not change mu.", "encoder_ignores_label")
    z = torch.randn(32, ns["Z_DIM"])
    if torch.allclose(dec(z, a), dec(z, b)):
        raise Fail("The decoder ignores the label: changing y does not change its output, so you cannot "
                   "choose which class to generate.", "decoder_ignores_label")


def check_elbo(ns):
    x, y = _data(32, seed=6)
    torch.manual_seed(6)
    enc, dec = ns["Encoder"](), ns["Decoder"]()

    def loss(beta):
        torch.manual_seed(60)
        return ns["elbo_loss"](x, y, enc, dec, beta=beta).item()

    l0, l1, l4 = loss(0.0), loss(1.0), loss(4.0)
    with torch.no_grad():
        mu, logvar = enc(x, y)
        kl = (0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1)).mean().item()
        torch.manual_seed(60)
        mu2, logvar2 = enc(x, y)
        z = ns["reparameterize"](mu2, logvar2)
        recon = F.binary_cross_entropy_with_logits(dec(z, y), x, reduction="none").sum((1, 2, 3)).mean().item()

    def near(a, b, tol=0.02):
        return abs(a - b) <= tol * max(abs(b), 1e-6)

    if near(l0, kl, 0.05) and not near(l0, recon, 0.05):
        raise Fail("With beta = 0 your loss equals the KL term, so beta multiplies the reconstruction instead "
                   "of the KL.", "beta_on_reconstruction")
    if abs(l4 - l0) < kl:
        raise Fail("Your loss is the same for beta = 0 and beta = 4: beta is never used. It should multiply "
                   "the KL term.", "beta_ignored")
    if not near(l4 - l0, 4 * (l1 - l0), 0.05):
        raise Fail("Your loss does not change linearly with beta; beta should multiply only the KL term.",
                   "beta_wrong")
    if near(l1 - l0, kl * len(x), 0.05):
        raise Fail("Your loss sums over the batch instead of averaging, so its scale and the effective "
                   "learning rate change with batch size.", "elbo_sum_over_batch")
    if not near(l1 - l0, kl, 0.05):
        raise Fail(f"The KL part of your loss ({l1 - l0:.3f}) does not match the batch-mean KL of your encoder's "
                   f"outputs ({kl:.3f}).", "elbo_kl_term_wrong")
    if not near(l0, recon, 0.05):
        raise Fail(f"With beta = 0 your loss ({l0:.2f}) should equal the batch-mean reconstruction loss "
                   f"({recon:.2f}).", "elbo_recon_term_wrong")


def check_sample(ns):
    torch.manual_seed(7)
    seen = []
    dec = ns["Decoder"]()
    original = dec.forward

    def spy(z, y):
        seen.append(z.detach())
        return original(z, y)

    dec.forward = spy
    y = torch.randint(0, 10, (2000,))
    with torch.no_grad():
        imgs = ns["sample"](dec, y)
    if imgs.shape != (2000, 1, 28, 28):
        raise Fail(f"sample should return shape (len(y), 1, 28, 28); got {tuple(imgs.shape)}.", "sample_shape")
    if imgs.min() < 0 or imgs.max() > 1:
        raise Fail("sample returns values outside [0, 1]. The decoder outputs logits; convert them to "
                   "probabilities with a sigmoid before returning images.", "sample_returns_logits")
    if not seen:
        raise Fail("sample does not call the decoder you pass in.", "sample_ignores_decoder")
    z = torch.cat(seen)
    if abs(z.mean().item()) > 0.1 or abs(z.std().item() - 1) > 0.1:
        raise Fail(f"Generation should draw z from the prior N(0, I). The codes you decode have mean "
                   f"{z.mean().item():.2f} and standard deviation {z.std().item():.2f}.", "sample_wrong_prior")


def check_training(ns):
    torch.manual_seed(8)
    x, y = _data(256, seed=8)
    enc, dec = ns["Encoder"](), ns["Decoder"]()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    losses = []
    for step in range(80):
        idx = torch.randint(0, len(x), (64,))
        loss = ns["elbo_loss"](x[idx], y[idx], enc, dec, beta=1.0)
        if not torch.isfinite(loss):
            raise Fail(f"The loss became {loss.item()} at step {step}. Check for exp of large values or log of "
                       "zero.", "training_nan")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    if last > 0.9 * first:
        raise Fail(f"The loss barely decreases in a short training run ({first:.1f} to {last:.1f}). Check that "
                   "gradients reach both networks.", "training_stalls")


CHECKS = [
    Check("shapes", "Encoder and decoder shapes", ("Z_DIM", "Encoder", "Decoder"), check_shapes),
    Check("kl", "KL divergence", ("kl_divergence",), check_kl, LEARN),
    Check("reparameterize", "Reparameterization trick", ("reparameterize",), check_reparameterize, LEARN),
    Check("reconstruction", "Reconstruction loss", ("reconstruction_loss",), check_reconstruction, LEARN),
    Check("decoder_logits", "Decoder outputs logits", ("Z_DIM", "Decoder"), check_decoder_logits, LEARN),
    Check("conditioning", "Both networks use the label", ("Z_DIM", "Encoder", "Decoder"), check_conditioning),
    Check("elbo", "ELBO loss: terms, beta and reduction",
          ("Encoder", "Decoder", "elbo_loss", "reparameterize"), check_elbo, LEARN),
    Check("sample", "Sampling from the prior", ("Z_DIM", "Decoder", "sample"), check_sample, LEARN),
    Check("training", "A short training run improves the loss", ("Encoder", "Decoder", "elbo_loss"),
          check_training),
]
