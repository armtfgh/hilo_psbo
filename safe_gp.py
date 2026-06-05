"""
Robust GP fitting with cascading fallbacks.

The default BoTorch ``fit_gpytorch_mll`` occasionally fails with
``ModelFittingError: All attempts to fit the model have failed.`` This happens
when the covariance matrix becomes ill-conditioned, typically when:

  - several observations are extremely close in input space,
  - the target variance is very small relative to the noise,
  - the initial hyperparameters happen to put the optimizer in a bad basin.

For long sweeps over many seeds a single bad seed is enough to kill an
entire figure. ``safe_fit_gp`` cascades through three strategies and returns
a fitted GP. It only re-raises if all three fail (very rare).

Strategy 1: plain SingleTaskGP fit with mild Cholesky jitter (1e-4) and
            ``max_attempts = 5``.
Strategy 2: same data + tiny output jitter, wrap with Normalize(input) and
            Standardize(output) transforms, fit with jitter 1e-3 and
            ``max_attempts = 10``.
Strategy 3: heavier output jitter (1e-4), no transforms, jitter 1e-2,
            ``max_attempts = 20``.
"""
from __future__ import annotations

from typing import Tuple

import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.settings import cholesky_jitter


def safe_fit_gp(
    X_obs: torch.Tensor,
    Y_obs: torch.Tensor,
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
    # --- Strategy 1 ------------------------------------------------------
    try:
        gp = SingleTaskGP(X_obs, Y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with cholesky_jitter(1e-4):
            fit_gpytorch_mll(mll, max_attempts=5)
        return gp, mll
    except ModelFittingError:
        pass

    # --- Strategy 2 ------------------------------------------------------
    try:
        Y_j = Y_obs + 1e-6 * torch.randn_like(Y_obs)
        gp = SingleTaskGP(
            X_obs,
            Y_j,
            input_transform=Normalize(d=X_obs.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with cholesky_jitter(1e-3):
            fit_gpytorch_mll(mll, max_attempts=10)
        return gp, mll
    except ModelFittingError:
        pass

    # --- Strategy 3 (last-ditch) ---------------------------------------
    Y_j = Y_obs + 1e-4 * torch.randn_like(Y_obs)
    gp = SingleTaskGP(X_obs, Y_j)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    with cholesky_jitter(1e-2):
        fit_gpytorch_mll(mll, max_attempts=20)
    return gp, mll


__all__ = ["safe_fit_gp"]
