#%%

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from prior_gp import GPWithPriorMean, Prior
from readout_schema import flat_readout, readout_to_prior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


def ground_truth(x: torch.Tensor) -> torch.Tensor:
    """Synthetic 1D ground truth with low early yields and an increasing trend."""
    bump = 0.7 * torch.exp(-0.5 * ((x - 0.75) / 0.08) ** 2)
    trend = 0.3 * torch.sigmoid(12.0 * (x - 0.5))
    return bump + trend


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sample_training_points(
    n_points: int = 5,
    seed: int = 7,
    *,
    method: str = "random",
) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty((0, 1), device=DEVICE, dtype=DTYPE)
    rng = np.random.default_rng(int(seed))
    method = method.lower().strip()
    if method == "jittered":
        base = np.linspace(0.05, 0.95, n_points)
        jitter = (rng.random(n_points) - 0.5) * 0.08
        points = np.clip(base + jitter, 0.0, 1.0)
    else:
        points = rng.random(n_points)
    points = np.sort(points)
    return torch.tensor(points, device=DEVICE, dtype=DTYPE).unsqueeze(-1)


def fit_plain_gp(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def fit_prior_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    prior: Prior,
    *,
    prior_weight: float = 1.0,
    hyperparams_source: Optional[SingleTaskGP] = None,
    freeze_hyperparams: bool = True,
) -> Tuple[GPWithPriorMean, float]:
    if train_x.shape[0] < 2:
        m0_scale = float(prior_weight)
    else:
        m0 = prior.m0_torch(train_x).reshape(-1)
        yv = train_y.reshape(-1)
        m0c = m0 - m0.mean()
        yc = yv - yv.mean()
        denom = torch.dot(m0c, m0c).item()
        alpha = (torch.dot(m0c, yc).item() / (denom + 1e-12)) if denom > 0 else 0.0
        m0_scale = max(float(alpha), 0.0) * float(prior_weight)

    m0 = prior.m0_torch(train_x).unsqueeze(-1)
    resid = train_y - m0_scale * m0
    gp_resid = SingleTaskGP(train_x, resid)

    if hyperparams_source is not None:
        gp_resid.covar_module.load_state_dict(hyperparams_source.covar_module.state_dict())
        gp_resid.likelihood.load_state_dict(hyperparams_source.likelihood.state_dict())
        gp_resid.mean_module.load_state_dict(hyperparams_source.mean_module.state_dict())
        if freeze_hyperparams:
            for param in gp_resid.covar_module.parameters():
                param.requires_grad_(False)
            for param in gp_resid.likelihood.parameters():
                param.requires_grad_(False)
            for param in gp_resid.mean_module.parameters():
                param.requires_grad_(False)

    if not freeze_hyperparams or hyperparams_source is None:
        mll = ExactMarginalLogLikelihood(gp_resid.likelihood, gp_resid)
        fit_gpytorch_mll(mll)

    gp_resid.eval()
    model = GPWithPriorMean(base_gp=gp_resid, prior=prior, m0_scale=m0_scale)
    model.eval()
    return model, m0_scale


def posterior_stats(model: torch.nn.Module, grid: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3):
        if isinstance(model, GPWithPriorMean):
            base_post = model.base_gp.posterior(grid)
            mean = base_post.mean.squeeze(-1)
            m0 = model.prior.m0_torch(grid).reshape(base_post.mean.shape).squeeze(-1)
            mean = mean + model.m0_scale * m0
            std = base_post.variance.clamp_min(0.0).sqrt().squeeze(-1)
        else:
            posterior = model.posterior(grid)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1)
    return mean.cpu().numpy(), std.cpu().numpy()


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_f: float, *, xi: float = 0.0) -> np.ndarray:
    mu_t = torch.as_tensor(mu, dtype=DTYPE, device=DEVICE)
    sigma_t = torch.as_tensor(sigma, dtype=DTYPE, device=DEVICE)
    sigma_safe = torch.clamp(sigma_t, min=1e-12)
    imp = mu_t - float(best_f) - float(xi)
    z = imp / sigma_safe
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * z**2)
    Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    ei = imp * Phi + sigma_safe * phi
    ei = torch.where(sigma_t <= 1e-12, torch.clamp(imp, min=0.0), ei)
    return torch.clamp(ei, min=0.0).detach().cpu().numpy()


def plot_posterior(
    ax: plt.Axes,
    x_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    true_curve: np.ndarray,
    *,
    prior_curve: Optional[np.ndarray] = None,
) -> None:
    ax.plot(x_grid, true_curve, color="#111111", lw=1.5, label="Ground truth")
    ax.plot(x_grid, mean, color="#1f77b4", lw=2.2, label="Posterior mean")
    ax.fill_between(x_grid, mean - 2 * std, mean + 2 * std, color="#1f77b4", alpha=0.18)
    ax.scatter(train_x, train_y, color="#d62728", edgecolors="k", zorder=4, label="Samples")
    if prior_curve is not None:
        ax.plot(x_grid, prior_curve, color="#7b2cbf", ls="--", lw=1.8, label="Prior mean (m0_x)")
    ax.set_xlabel("Temperature (Normalised)")
    ax.set_ylabel("Yield")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")


def build_default_readouts(feature_key: str = "x1") -> Dict[str, Dict[str, Any]]:
    flat = flat_readout([feature_key])
    trend = {
        "effects": {
            feature_key: {
                "effect": "increasing",
                "scale": 0.9,
                "confidence": 0.8,
                "range_hint": [0.45, 0.7],
            }
        },
        "interactions": [],
        "bumps": [],
        "constraints": [],
    }
    bump = {
        "effects": {feature_key: {"effect": "flat", "scale": 0.0, "confidence": 0.0}},
        "interactions": [],
        "bumps": [{"mu": [0.75], "sigma": [0.08], "amp": 0.7}],
        "constraints": [],
    }
    constraint = {
        "effects": {feature_key: {"effect": "flat", "scale": 0.0, "confidence": 0.0}},
        "interactions": [],
        "bumps": [],
        "constraints": [{"var": feature_key, "range": [0.0, 0.2], "penalty": 1.8, "k": 60.0}],
    }
    combined = {
        "effects": trend["effects"],
        "interactions": [],
        "bumps": bump["bumps"],
        "constraints": constraint["constraints"],
    }
    return {
        "Flat": flat,
        "Trend": trend,
        "Bump": bump,
        "Constraint": constraint,
        "Combined": combined,
    }


def plot_prior_means(
    *,
    readouts: Dict[str, Dict[str, Any]],
    grid: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    set_manuscript_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x_np = grid.squeeze(-1).cpu().numpy()

    colors = {
        "Trend": "#1f77b4",
        "Bump": "#2ca02c",
        "Constraint": "#d62728",
        "Combined": "#7b2cbf",
        "Flat": "#7f7f7f",
    }
    for label, ro in readouts.items():
        feature_names = list((ro.get("effects") or {}).keys()) or ["x1"]
        prior = readout_to_prior(ro, feature_names=feature_names)
        m0 = prior.m0_torch(grid).squeeze(-1).cpu().numpy()
        ax.plot(
            x_np,
            m0,
            lw=2.0,
            ls="--",
            color=colors.get(label, "#444444"),
            label=label,
        )

    ax.set_xlabel("Temperature (Normalised)")
    ax.set_ylabel("Prior mean (m0_x)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", ncol=2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_prior_mean_single(
    *,
    readout: Dict[str, Any],
    grid: torch.Tensor,
    label: str = "Combined",
    color: str = "#7b2cbf",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    set_manuscript_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x_np = grid.squeeze(-1).cpu().numpy()
    feature_names = list((readout.get("effects") or {}).keys()) or ["x1"]
    prior = readout_to_prior(readout, feature_names=feature_names)
    m0 = prior.m0_torch(grid).squeeze(-1).cpu().numpy()
    ax.plot(x_np, m0, lw=2.6, color=color, ls="--", label=label)
    ax.set_xlabel("Temperature (Normalised)")
    ax.set_ylabel("Prior mean (m0_x)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def compute_gp_data(
    *,
    readout: Dict[str, Any],
    n_train: int,
    seed: int,
    sample_method: str,
    noise_sd: float,
    prior_weight: float,
) -> Dict[str, np.ndarray]:
    train_x = sample_training_points(n_points=n_train, seed=seed, method=sample_method)
    train_y = ground_truth(train_x)
    if noise_sd > 0:
        train_y = train_y + noise_sd * torch.randn_like(train_y)

    grid = torch.linspace(0.0, 1.0, 400, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
    true_curve = ground_truth(grid).squeeze(-1).cpu().numpy()

    gp_flat = fit_plain_gp(train_x, train_y)
    flat_mean, flat_std = posterior_stats(gp_flat, grid)

    prior = readout_to_prior(readout, feature_names=["x1"])
    gp_prior, m0_scale = fit_prior_gp(
        train_x,
        train_y,
        prior,
        prior_weight=prior_weight,
        hyperparams_source=gp_flat,
        freeze_hyperparams=True,
    )
    prior_mean, prior_std = posterior_stats(gp_prior, grid)
    prior_curve = (m0_scale * prior.m0_torch(grid).squeeze(-1)).cpu().numpy()

    best_f = float(np.max(train_y.cpu().numpy())) if train_y.numel() else 0.0
    ei_flat = expected_improvement(flat_mean, flat_std, best_f)
    ei_prior = expected_improvement(prior_mean, prior_std, best_f)

    return {
        "grid": grid.squeeze(-1).cpu().numpy(),
        "true_curve": true_curve,
        "train_x": train_x.squeeze(-1).cpu().numpy(),
        "train_y": train_y.squeeze(-1).cpu().numpy(),
        "flat_mean": flat_mean,
        "flat_std": flat_std,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "prior_curve": prior_curve,
        "ei_flat": ei_flat,
        "ei_prior": ei_prior,
    }


def plot_prior_means_separate(
    *,
    readouts: Dict[str, Dict[str, Any]],
    grid: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    set_manuscript_style()
    labels = [k for k in readouts.keys() if k.lower() not in {"flat", "combined"}]
    n_rows = len(labels)
    if n_rows == 0:
        raise ValueError("No non-flat readouts available for plotting.")

    fig, axes = plt.subplots(n_rows, 1, figsize=(6.5, 2.3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    x_np = grid.squeeze(-1).cpu().numpy()
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#7b2cbf", "#8c564b"]

    for idx, label in enumerate(labels):
        ro = readouts[label]
        feature_names = list((ro.get("effects") or {}).keys()) or ["x1"]
        prior = readout_to_prior(ro, feature_names=feature_names)
        m0 = prior.m0_torch(grid).squeeze(-1).cpu().numpy()
        ax = axes[idx]
        ax.plot(x_np, m0, lw=2.4, color=colors[idx % len(colors)], ls="--", label=label)
        ax.set_ylabel("Prior mean (m0_x)")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Temperature (Normalised)")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def compute_prior_means_data(
    *,
    readouts: Dict[str, Dict[str, Any]],
    grid: torch.Tensor,
) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for label, ro in readouts.items():
        feature_names = list((ro.get("effects") or {}).keys()) or ["x1"]
        prior = readout_to_prior(ro, feature_names=feature_names)
        data[label] = prior.m0_torch(grid).squeeze(-1).cpu().numpy()
    return data


def plot_gp_comparison(
    *,
    readout: Dict[str, Any],
    n_train: int = 5,
    seed: int = 7,
    sample_method: str = "random",
    noise_sd: float = 0.0,
    prior_weight: float = 1.0,
    save_path: Optional[str] = None,
    show: bool = False,
    gp_data: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    set_manuscript_style()

    if gp_data is None:
        gp_data = compute_gp_data(
            readout=readout,
            n_train=n_train,
            seed=seed,
            sample_method=sample_method,
            noise_sd=noise_sd,
            prior_weight=prior_weight,
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    plot_posterior(
        axes[0],
        gp_data["grid"],
        gp_data["flat_mean"],
        gp_data["flat_std"],
        gp_data["train_x"],
        gp_data["train_y"],
        gp_data["true_curve"],
    )
    plot_posterior(
        axes[1],
        gp_data["grid"],
        gp_data["prior_mean"],
        gp_data["prior_std"],
        gp_data["train_x"],
        gp_data["train_y"],
        gp_data["true_curve"],
        prior_curve=gp_data["prior_curve"],
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_ei_comparison(
    *,
    gp_data: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    set_manuscript_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)

    ei_flat = gp_data.get("ei_flat")
    ei_prior = gp_data.get("ei_prior")
    if ei_flat is None or ei_prior is None:
        best_f = float(np.max(gp_data["train_y"])) if gp_data["train_y"].size else 0.0
        ei_flat = expected_improvement(gp_data["flat_mean"], gp_data["flat_std"], best_f)
        ei_prior = expected_improvement(gp_data["prior_mean"], gp_data["prior_std"], best_f)

    axes[0].plot(gp_data["grid"], ei_flat, color="#6c757d", lw=2.4)
    axes[1].plot(gp_data["grid"], ei_prior, color="#6c757d", lw=2.4)

    axes[0].set_xlabel("Temperature (Normalised)")
    axes[1].set_xlabel("Temperature (Normalised)")
    axes[0].set_ylabel("Expected Improvement")
    axes[0].set_xlim(0.0, 1.0)
    axes[1].set_xlim(0.0, 1.0)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def build_case_study_figures(
    *,
    n_train: int = 5,
    seed: int = 7,
    sample_method: str = "random",
    noise_sd: float = 0.0,
    prior_weight: float = 1.0,
    show: bool = False,
) -> Dict[str, object]:
    readouts = build_default_readouts(feature_key="x1")

    grid = torch.linspace(0.0, 1.0, 400, device=DEVICE, dtype=DTYPE).unsqueeze(-1)
    prior_means_data = compute_prior_means_data(readouts=readouts, grid=grid)
    fig_priors, ax_priors = plot_prior_means(readouts=readouts, grid=grid, show=show)
    fig_priors_stack, axes_priors_stack = plot_prior_means_separate(
        readouts=readouts, grid=grid, show=show
    )
    fig_prior_combined, ax_prior_combined = plot_prior_mean_single(
        readout=readouts["Combined"], grid=grid, show=show
    )

    gp_data = compute_gp_data(
        readout=readouts["Combined"],
        n_train=n_train,
        seed=seed,
        sample_method=sample_method,
        noise_sd=noise_sd,
        prior_weight=prior_weight,
    )
    fig_gp, axes_gp = plot_gp_comparison(
        readout=readouts["Combined"],
        n_train=n_train,
        seed=seed,
        sample_method=sample_method,
        noise_sd=noise_sd,
        prior_weight=prior_weight,
        show=show,
        gp_data=gp_data,
    )
    fig_ei, axes_ei = plot_ei_comparison(gp_data=gp_data, show=show)

    return {
        "readouts": readouts,
        "data": {
            "grid": gp_data["grid"],
            "prior_means": prior_means_data,
            "gp": gp_data,
        },
        "figure_priors": fig_priors,
        "axes_priors": ax_priors,
        "figure_priors_stack": fig_priors_stack,
        "axes_priors_stack": axes_priors_stack,
        "figure_prior_combined": fig_prior_combined,
        "axes_prior_combined": ax_prior_combined,
        "figure_gp": fig_gp,
        "axes_gp": axes_gp,
        "figure_ei": fig_ei,
        "axes_ei": axes_ei,
    }

#execution code
if __name__ == "__main__":
    build_case_study_figures(n_train=2,seed=26, sample_method="random", show=True,prior_weight=0.6)
#%%
