from __future__ import annotations
from dataclasses import dataclass, field
import warnings
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
import gpytorch
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

@dataclass
class Prior:
    effects: Dict[str, Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    bumps: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    feature_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self._feature_lookup = {}
        self._feature_lookup_lower = {}
        if self.feature_names:
            for idx, name in enumerate(self.feature_names):
                key = str(name)
                self._feature_lookup[key] = idx
                self._feature_lookup_lower[key.lower()] = idx

    def m0_torch(self, X: Tensor) -> Tensor:
        """Evaluate the prior mean m0(X) for normalized inputs X in [0,1]^d."""
        if X.ndim == 1:
            X = X.unsqueeze(0)
        d = X.shape[-1]
        pos_signal = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        def _parse_dim(name: Any) -> Optional[int]:
            if isinstance(name, int):
                idx = name
            elif isinstance(name, str):
                label = name.strip()
                if label.lower().startswith("x"):
                    try:
                        idx = int(label[1:]) - 1
                    except ValueError:
                        idx = None
                else:
                    idx = self._feature_lookup.get(label)
                    if idx is None:
                        idx = self._feature_lookup_lower.get(label.lower())
            else:
                return None
            if idx is None:
                return None
            return idx if 0 <= idx < d else None

        def _sigmoid(z: Tensor, center: float = 0.5, k: float = 6.0) -> Tensor:
            return torch.sigmoid(k * (z - center))

        def _gauss1d(z: Tensor, mu: float, s: float) -> Tensor:
            s = max(s, 1e-6)
            return torch.exp(-0.5 * ((z - mu) / s) ** 2)

        # ----- main effects -----
        for name, spec in (self.effects or {}).items():
            idx = _parse_dim(name)
            if idx is None:
                continue
            z = X[..., idx]
            eff = str(spec.get("effect", "flat")).lower()
            scale = float(spec.get("scale", 0.0))
            conf = float(spec.get("confidence", 0.0))
            amp = 0.6 * scale * conf
            if amp == 0.0:
                continue

            range_hint = spec.get("range_hint")
            center = 0.5
            width = 0.18
            if isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
                lo, hi = float(range_hint[0]), float(range_hint[1])
                center = 0.5 * (lo + hi)
                width = max(abs(hi - lo) / 3.0, 0.05)

            if eff in {"increase", "increasing"}:
                pos_signal = pos_signal + amp * _sigmoid(z, center=center)
            elif eff in {"decrease", "decreasing"}:
                pos_signal = pos_signal - amp * _sigmoid(z, center=center)
            elif eff in {"nonmonotone-peak", "peak"}:
                pos_signal = pos_signal + amp * _gauss1d(z, mu=center, s=width)
            elif eff in {"nonmonotone-valley", "valley"}:
                pos_signal = pos_signal - amp * _gauss1d(z, mu=center, s=width)
            # "flat" or unknown => no contribution

        # ----- pairwise interactions -----
        for inter in (self.interactions or []):
            pair = inter.get("vars") or inter.get("pair") or inter.get("indices")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            idx_a = _parse_dim(pair[0])
            idx_b = _parse_dim(pair[1])
            if idx_a is None or idx_b is None:
                continue

            itype = str(inter.get("type", "synergy")).lower()
            sign = 1.0
            if itype in {"antagonism", "tradeoff", "negative"}:
                sign = -1.0
            strength = max(float(inter.get("scale", inter.get("strength", 0.0))), 0.0)
            conf = max(float(inter.get("confidence", 0.0)), 0.0)
            if strength == 0.0 and conf == 0.0:
                conf = 0.5
            amp = 0.2 * (strength if strength > 0 else 1.0) * conf
            term = (X[..., idx_a] * X[..., idx_b])
            pos_signal = pos_signal + sign * amp * term

        # ----- bumps -----
        for bump in (self.bumps or []):
            if not bump:
                continue
            mu_vals = bump.get("mu")
            if mu_vals is None:
                continue
            mu = torch.tensor(list(mu_vals)[:d], device=X.device, dtype=X.dtype)
            if mu.numel() == 0:
                mu = torch.full((d,), 0.5, device=X.device, dtype=X.dtype)
            elif mu.numel() < d:
                mu = torch.cat([mu, torch.full((d - mu.numel(),), 0.5, device=X.device, dtype=X.dtype)], dim=0)

            sigma_vals = bump.get("sigma", 0.15)
            if isinstance(sigma_vals, (list, tuple)):
                sigma = torch.tensor(list(sigma_vals)[:d], device=X.device, dtype=X.dtype)
                if sigma.numel() == 0:
                    sigma = torch.full((d,), 0.15, device=X.device, dtype=X.dtype)
                elif sigma.numel() < d:
                    sigma = torch.cat([sigma, torch.full((d - sigma.numel(),), sigma[-1], device=X.device, dtype=X.dtype)], dim=0)
            else:
                sigma = torch.full((d,), float(sigma_vals), device=X.device, dtype=X.dtype)
            sigma = torch.clamp(sigma, min=1e-6)

            amp = float(bump.get("amp", 0.1))
            diff = (X - mu) / sigma
            gauss = torch.exp(-0.5 * torch.sum(diff ** 2, dim=-1))
            pos_signal = pos_signal + amp * gauss

        # ----- constraints (negative knowledge; forbidden regions) -----
        neg_penalty = torch.zeros_like(pos_signal)
        for c in (self.constraints or []):
            if not isinstance(c, dict):
                continue
            var = c.get("var", None)
            r = c.get("range", None)
            if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            idx = _parse_dim(var)
            if idx is None:
                continue

            lo = float(r[0])
            hi = float(r[1])
            if hi < lo:
                lo, hi = hi, lo
            lo = float(min(max(lo, 0.0), 1.0))
            hi = float(min(max(hi, 0.0), 1.0))

            strength = float(c.get("penalty", c.get("weight", 5.0)))
            k = float(c.get("sharpness", c.get("k", 60.0)))

            z = X[..., idx]
            # Soft "inside interval" gate: ~1 inside [lo,hi], ~0 outside.
            gate = torch.sigmoid(k * (z - lo)) - torch.sigmoid(k * (z - hi))
            gate = gate.clamp(0.0, 1.0)
            neg_penalty = neg_penalty + strength * gate

        # m0(x) = positive signal - penalty
        return pos_signal - neg_penalty

from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from typing import Optional, List

class GPWithPriorMean(Model):
    def __init__(self, base_gp: SingleTaskGP, prior: Prior, m0_scale: float = 1.0):
        super().__init__() # <-- important!
        self.base_gp = base_gp # safe to assign after super().__init__
        self.prior = prior
        self.m0_scale = float(m0_scale)

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        base_post = self.base_gp.posterior(X, observation_noise=observation_noise, **kwargs)
        mvn = base_post.mvn
        m0 = self.prior.m0_torch(X).reshape(mvn.mean.shape)
        mean = mvn.mean + self.m0_scale * m0
        cov = mvn.covariance_matrix
        try:
            new_mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
        except RuntimeError:
            # Minimal jitter fallback for rare non-PD covariance issues.
            diag = cov.diagonal(dim1=-2, dim2=-1)
            diag_mean = float(diag.mean().clamp_min(1e-12).item())
            jitter = max(1e-8, 1e-6 * diag_mean)
            eye = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
            while eye.ndim < cov.ndim:
                eye = eye.unsqueeze(0)
            cov = cov + jitter * eye
            new_mvn = MultivariateNormal(mean=mean, covariance_matrix=cov)
        return GPyTorchPosterior(new_mvn)

    def condition_on_observations(self, X: Tensor, Y: Tensor, noise: Optional[Tensor] = None, **kwargs):
        cm = self.base_gp.condition_on_observations(X=X, Y=Y, noise=noise, **kwargs)
        return GPWithPriorMean(base_gp=cm, prior=self.prior, m0_scale=self.m0_scale)

    def fantasize(self, X: Tensor, sampler, observation_noise: bool = True, **kwargs):
        fm = self.base_gp.fantasize(X=X, sampler=sampler, observation_noise=observation_noise, **kwargs)
        return GPWithPriorMean(base_gp=fm, prior=self.prior, m0_scale=self.m0_scale)

    def subset_output(self, idcs: List[int]):
        return self

def fit_residual_gp(X: Tensor, Y: Tensor, prior: Prior) -> Tuple[SingleTaskGP, float]:
    m0 = prior.m0_torch(X).reshape(-1)
    yv = Y.reshape(-1)
    m0c = m0 - m0.mean(); yc = yv - yv.mean()
    denom = torch.dot(m0c, m0c).item()
    alpha = (torch.dot(m0c, yc).item() / (denom + 1e-12)) if denom > 0 else 0.0
    Y_resid = Y - alpha * m0.unsqueeze(-1)
    fit_device = torch.device("cpu") if USE_CUDA else DEVICE
    X_fit = X.detach().to(fit_device)
    Y_fit = Y_resid.detach().to(fit_device)

    if X_fit.shape[0] < 2:
        jitter = 1e-3 * torch.randn_like(X_fit)
        X_fit = torch.cat([X_fit, (X_fit + jitter).clamp(0.0, 1.0)], dim=0)
        Y_fit = torch.cat([Y_fit, Y_fit + 1e-4 * torch.randn_like(Y_fit)], dim=0)

    y_std = float(torch.std(Y_fit).item()) if Y_fit.numel() > 1 else 0.0
    if y_std < 1e-6:
        Y_fit = Y_fit + 1e-4 * torch.randn_like(Y_fit)

    X_fit = X_fit + 1e-6 * torch.randn_like(X_fit)
    gp = SingleTaskGP(X_fit, Y_fit, outcome_transform=Standardize(m=1)).to(fit_device)
    gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-6))
    gp.likelihood.noise = torch.tensor(1e-4, device=fit_device, dtype=DTYPE)

    def _set_heuristics() -> None:
        with torch.no_grad():
            if X_fit.shape[0] > 1:
                dists = torch.cdist(X_fit, X_fit)
                mask = dists > 0
                med = torch.median(dists[mask]) if mask.any() else torch.tensor(0.2, device=fit_device)
                if hasattr(gp.covar_module, "base_kernel"):
                    gp.covar_module.base_kernel.lengthscale = med.clamp_min(1e-3)
            y_std_local = torch.std(Y_fit)
            if not torch.isfinite(y_std_local):
                y_std_local = torch.tensor(1.0, device=fit_device, dtype=DTYPE)
            y_std_local = y_std_local.clamp_min(1e-3)
            if hasattr(gp.covar_module, "outputscale"):
                gp.covar_module.outputscale = (y_std_local ** 2).clamp_min(1e-5)
            noise_var = ((0.05 * y_std_local) ** 2).clamp_min(1e-5)
            gp.likelihood.noise_covar.initialize(noise=noise_var)

    _set_heuristics()
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(fit_device)
    try:
        fit_gpytorch_mll(mll)
    except Exception as exc:
        warnings.warn(f"GP fit failed; using heuristic hyperparameters. ({exc})")
        gp.eval()
    gp = gp.to(DEVICE)
    return gp, alpha

def alignment_on_obs(X: Tensor, Y: Tensor, prior: Prior) -> float:
    m0 = prior.m0_torch(X).reshape(-1); yv = Y.reshape(-1)
    m0c = m0 - m0.mean(); yc = yv - yv.mean()
    num = torch.dot(m0c, yc).item()
    den = torch.sqrt(torch.dot(m0c, m0c) * torch.dot(yc, yc) + 1e-12).item()
    return num / den if den > 0 else 0.0

__all__ = ["Prior", "GPWithPriorMean", "fit_residual_gp", "alignment_on_obs"]
