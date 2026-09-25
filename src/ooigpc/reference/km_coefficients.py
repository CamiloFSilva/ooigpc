"""
reference/km_coefficients.py — Kramers–Moyal (KM) coefficient estimation.

This module implements the reference identification method based on
Kramers–Moyal (KM) coefficients, following Noray & Schuermans (2022)
and the MATLAB scripts ``OOID_KMCoeff.m`` / ``OOID_KMCoeff_V2.m``.

The KM coefficients for a 1-D Markov process are defined as

    D^(n)(x₀) = lim_{τ→0}  ⟨(A(t+τ) − A(t))^n | A(t) = x₀⟩ / (n! τ)

In practice the limit is approximated by extrapolating the conditional
jump moments computed at finite τ to τ → 0, using an exponential fit
(see :func:`extrapolate_km`):

    M^(n)(x₀, τ) ≈ α · exp(β · τ)

so that  D^(n)(x₀) = α/n!  as τ → 0.

References
----------
Noray, A. & Schuermans, B. (2022). *Combustion noise and thermoacoustic
oscillations: Stochastic analysis and dynamical modelling.*  Combust.
Flame.

Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame. §4 (comparison).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "conditional_jump_moments",
    "extrapolate_km",
    "KMResult",
]

from dataclasses import dataclass, field


@dataclass
class KMResult:
    """Container for KM coefficient estimation output.

    Attributes
    ----------
    x_bins:
        Centres of the amplitude bins used for conditioning.
    D1:
        First KM coefficient (drift) D^(1)(x₀).
    D2:
        Second KM coefficient (diffusion) D^(2)(x₀).
    jump_moments_1:
        Conditional first jump moments at each τ, shape (n_tau, n_bins).
    jump_moments_2:
        Conditional second jump moments at each τ, shape (n_tau, n_bins).
    taus:
        Time delays used.
    """
    x_bins: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    jump_moments_1: np.ndarray
    jump_moments_2: np.ndarray
    taus: np.ndarray


def conditional_jump_moments(
    x: np.ndarray,
    tau_values: np.ndarray,
    dt_sim: float,
    x_bins: np.ndarray,
    bin_width: float,
    orders: tuple[int, ...] = (1, 2),
    i_start: int = 5000,
) -> dict[int, np.ndarray]:
    """Compute conditional jump moments M^(n)(x₀, τ).

    For each bin centre *x₀* and each time delay τ, averages

        (A(t+τ) − A(t))^n

    over all t where A(t) ∈ [x₀ − bin_width/2, x₀ + bin_width/2].

    Parameters
    ----------
    x:
        Amplitude envelope time series.
    tau_values:
        Delay times τ (seconds).
    dt_sim:
        Time step (seconds).
    x_bins:
        Bin centre values for the amplitude.
    bin_width:
        Width of each amplitude bin.
    orders:
        Moment orders to compute (typically (1, 2)).
    i_start:
        Transient skip.

    Returns
    -------
    dict mapping order → array of shape (n_tau, n_bins)
    """
    x = np.asarray(x, dtype=float)
    tk_vec = np.maximum(1, np.round(np.asarray(tau_values) / dt_sim).astype(int))
    n_tau = len(tk_vec)
    n_bins = len(x_bins)

    moments: dict[int, np.ndarray] = {n: np.zeros((n_tau, n_bins)) for n in orders}

    for idx, tk in enumerate(tk_vec):
        x1 = x[i_start : len(x) - tk]
        x2 = x[i_start + tk :]
        delta = x2 - x1

        for jb, xc in enumerate(x_bins):
            mask = np.abs(x1 - xc) < bin_width / 2.0
            if mask.sum() == 0:
                continue
            d = delta[mask]
            for n in orders:
                moments[n][idx, jb] = np.mean(d**n)

    return moments


def extrapolate_km(
    taus: np.ndarray,
    km_t: np.ndarray,
    t_min: float,
    init_growth_guess: float = -10.0,
) -> np.ndarray:
    """Extrapolate conditional jump moments to τ = 0.

    Fits  f(τ) = α · exp(β · τ)  to the tail  τ ≥ t_min  and
    evaluates at τ = 0.

    Matches the MATLAB function ``extrapolate_KM.m``.

    Parameters
    ----------
    taus:
        1-D array of time delays (seconds), length n_tau.
    km_t:
        2-D array of jump moments, shape (n_tau, n_bins).
    t_min:
        Minimum time delay above which the exponential model is fitted.
    init_growth_guess:
        Initial guess for β (decay rate, typically negative).

    Returns
    -------
    numpy.ndarray, shape (n_tau+1, n_bins)
        Extended array with τ = 0 prepended.
    """
    taus = np.asarray(taus, dtype=float)
    km_t = np.asarray(km_t, dtype=float)
    n_tau, n_bins = km_t.shape

    t_ext = np.concatenate([[0.0], taus])
    km_ext = np.zeros((n_tau + 1, n_bins))

    t_ind = np.searchsorted(taus, t_min, side="left")
    t_rest = taus[t_ind:]

    for ib in range(n_bins):
        to_ext = km_t[t_ind:, ib]

        def objective(coeffs: np.ndarray) -> float:
            a1, a2 = coeffs
            pred = a1 * np.exp(a2 * t_rest)
            return float(np.sum((to_ext - pred) ** 2))

        c0_start = km_t[t_ind, ib] * np.exp(0.1 * t_min)
        x0 = np.array([c0_start, init_growth_guess])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 2000 * n_tau, "xatol": 1e-10})
        a1, a2 = res.x
        km_ext[:, ib] = a1 * np.exp(a2 * t_ext)

    return km_ext
