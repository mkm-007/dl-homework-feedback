# Behavioral Code Diagnosis for Deep Learning

Checks a student's deep learning code by running it, then explains which concept is wrong. It does not compare against an answer key or show a solution.

Most grading scripts compare outputs and report "wrong". A VAE whose KL term is off by a factor of two still trains and still produces blurry digits, and a diffusion sampler that skips its last step still produces plausible points, so the student never finds out. These checks call the student's functions on controlled inputs, compare the behavior with the mathematics, and recognize common mistakes by their signature:

```
[FIX ] Reparameterization trick
       The spread of your samples equals the variance exp(logvar), not the standard deviation.
       The standard deviation is exp(0.5 * logvar).
[FIX ] Reconstruction loss
       Your reconstruction loss averages over pixels instead of summing. That makes it 784 times
       smaller relative to the KL term, which acts like a very large beta and can collapse the
       latent code.
```

## Try it

```bash
pip install -r requirements.txt
python -m feedback vae assignments/vae/starter.py           # every check reports what is missing
python -m feedback vae reference/vae.py                     # 9/9 checks passed
python -m feedback diffusion reference/diffusion.py         # 8/8 checks passed
```

Students' notebooks work too (`python -m feedback vae my_homework.ipynb`). Only imports, constants, functions and classes are loaded, so training cells and plots don't run.

## Assignments

- [`assignments/vae/`](assignments/vae/README.md): a conditional VAE. Nine checks cover shapes, the KL and reconstruction terms, the reparameterization trick and its gradient, conditioning on the label, the beta-weighted ELBO and its reduction, sampling from the prior, and a short training run.
- [`assignments/diffusion/`](assignments/diffusion/README.md): a DDPM on 2D points. Eight checks cover the noise schedule, alpha_bar, the forward process, the denoiser's use of t, the training loss, the reverse step, the sampling loop, and a short training run.

The diffusion checks test each piece in isolation by passing in their own models:

- **Training loss.** An oracle model knows the exact noise that was added. With it, a correct loss is exactly 0, and an offset of 1 gives exactly 1. Any other value points to a specific mistake: the wrong target, the wrong noise distribution, or summing over the batch.
- **Reverse step.** Averaging 40,000 reverse steps from one point recovers the mean of the step. That mean is compared with the DDPM formula and with each known wrong variant, at both an early and a late timestep. The spread is checked separately; both standard choices of variance are accepted.
- **Sampling loop.** The loop runs on data concentrated at a single point, with a perfect noise predictor. A correct loop ends exactly on that point. A spy model also records the order of the timesteps and the starting distribution.

## How well it works

`python -m evaluation.run_eval <assignment> --seeds 20` checks every case under 20 random seeds:

| | VAE | Diffusion |
|---|---|---|
| Known mistakes injected into a correct solution | 21: 100% detected, 100% named | 29: 100% detected, 100% named |
| Correct implementations written differently | 6: 0% flagged | 8: 0% flagged |
| Held-out mistakes, written before the evaluation ran and never used to tune the checks | 8: 75% detected | 8: 75% detected |

The first row is optimistic because the known mistakes and the checks were written together. The held-out row is the more honest measure, though the same author wrote both. Missed held-out mistakes:

- **VAE:** an ELBO that decodes `mu` instead of a sample, and an encoder that returns `mu` as `logvar`. Every term still has the right value and scale.
- **Diffusion:** a reverse step that indexes `alpha_bar[t - 1]`, and a denoiser with a `tanh` output. The first changes the mean by less than the sampling noise of the check. The second still trains, just worse.

Some held-out mistakes are detected but described by a generic or misleading message. A negated VAE loss is reported as a wrong KL term. A reverse step that shares one noise vector across the batch is reported as a wrong mean.

## Layout

```
feedback/core.py                    loading (with notebook filtering), running checks, the report
feedback/assignments/<name>.py      the checks for each assignment
assignments/<name>/                 the assignment and a starter file
reference/<name>.py                 a reference solution, used only for evaluation
bugs/<name>.py                      known mistakes, held-out mistakes, correct alternatives
evaluation/run_eval.py              detection, diagnosis and false-positive measurement
tests/                              72 tests
```

## Adding a mistake

Write the mistake as a small function or class in `bugs/<name>.py`, add it to `MUTATIONS` with the label you expect, and run the tests. If no check names it yet, add a branch to the relevant check that recognizes its signature and explains the concept behind it. Add mistakes you have not designed for to `HELD_OUT` before changing any check, so the held-out rate stays honest.
