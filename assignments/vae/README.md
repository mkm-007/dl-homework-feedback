# Assignment: conditional variational autoencoder

Build a conditional VAE for 28×28 grayscale images with values in [0, 1] and 10 classes. Training on MNIST is a good target, but the checks run on small synthetic data so they finish in seconds on a CPU.

## What to define

Define these names at the top level of your notebook or `.py` file. Functions must not depend on global variables other than the two constants.

| Name | Signature | Returns |
|---|---|---|
| `Z_DIM` | constant | latent size, for example `2` |
| `N_CLASSES` | constant | `10` |
| `Encoder` | `Encoder()` then `forward(x, y)` | `(mu, logvar)`, each of shape `(B, Z_DIM)` |
| `Decoder` | `Decoder()` then `forward(z, y)` | **logits** of shape `(B, 1, 28, 28)` |
| `reparameterize` | `(mu, logvar)` | a differentiable sample `z` of shape `(B, Z_DIM)` |
| `kl_divergence` | `(mu, logvar)` | KL to N(0, I) **per example**, shape `(B,)` |
| `reconstruction_loss` | `(logits, x)` | binary cross-entropy **summed over pixels**, per example, shape `(B,)` |
| `elbo_loss` | `(x, y, encoder, decoder, beta=1.0)` | scalar: batch mean of `reconstruction + beta * KL` |
| `sample` | `(decoder, y)` | images of shape `(len(y), 1, 28, 28)` with values in [0, 1] |

Here `x` has shape `(B, 1, 28, 28)` and `y` is a `(B,)` tensor of integer labels.

## Check your work

```bash
python -m feedback vae path/to/your_notebook.ipynb
```

Only imports, constants, functions and classes are loaded, so your training cells do not run during checking. The report names each problem it finds and links to an explanation; it never shows a solution.
