"""
diffusion.py — Mean-square displacement (MSD) and diffusion coefficient estimation.

The acoustic amplitude envelope A(t) satisfies the Itô SDE (Eq. 1 of the paper):

    dA = (a·A - b·A³ + c/A) dt  +  √(2c) dW

The diffusion coefficient *c* is extracted from the MSD at short time delays:

    MSD(τ) = ⟨(A(t+τ) - A(t))²⟩                         (Eq. 5)

which is fitted by the model (Suppl. §4):

    MSD(τ) = Â · (1 - exp(-τ / T̂))                      (Eq. S-11)

giving

    c = Â / (2 T̂)                                         (Eq. 7)

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.
Supplementary material §4.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "compute_msd",
    "fit_msd",
    "estimate_diffusion",
]


# ---------------------------------------------------------------------------
# MSD computation
# ---------------------------------------------------------------------------


def compute_msd(
    x: np.ndarray,
    tau_values: np.ndarray,
    dt_sim: float,
    i_start: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean-square displacement for a vector of time delays.

    Parameters
    ----------
    x:
        1-D time-series of the amplitude envelope A(t), length N+1.
    tau_values:
        Delay times τ to evaluate (in seconds).  Shape (n_tau,).
    dt_sim:
        Simulation time step (seconds).
    i_start:
        Index of the first sample to use (discard initial transient).
        Matches ``x(5000:…)`` in the MATLAB script.

    Returns
    -------
    taus:
        Effective delay times (seconds), shape (n_tau,).  May differ
        slightly from *tau_values* due to rounding to integer steps.
    MSD:
        ⟨(A(t+τ) - A(t))²⟩ for each τ, shape (n_tau,).

    Notes
    -----
    Negative or zero values of the envelope are discarded before computing
    statistics, consistent with the MATLAB implementation.
    """
    x = np.asarray(x, dtype=float)
    N = len(x) - 1

    # Convert delay times to integer step counts
    tk_vec = np.maximum(1, np.round(np.asarray(tau_values) / dt_sim).astype(int))
    taus = tk_vec * dt_sim

    MSD = np.zeros(len(tk_vec))

    for idx, tk in enumerate(tk_vec):
        i_end = N - tk
        if i_end <= i_start:
            continue

        x1 = x[i_start : i_end + 1]
        x2 = x[i_start + tk : i_end + tk + 1]

        # Discard unphysical (non-positive) envelope values
        mask = (x1 > 0) & (x2 > 0)
        x1m, x2m = x1[mask], x2[mask]
        if len(x1m) == 0:
            continue

        delta = x1m - x2m
        MSD[idx] = np.dot(delta, delta) / len(delta)

    return taus, MSD


# ---------------------------------------------------------------------------
# MSD model and fitting
# ---------------------------------------------------------------------------


def _msd_model(tau: np.ndarray, A_hat: float, T_hat: float) -> np.ndarray:
    """Exponential MSD model:  MSD(τ) = Â·(1 - exp(-τ/T̂))."""
    return A_hat * (1.0 - np.exp(-tau / T_hat))


def fit_msd(
    taus: np.ndarray,
    MSD: np.ndarray,
    p0: tuple[float, float] = (0.1, 0.01),
    bounds: tuple = ((0, 0), (np.inf, np.inf)),
) -> tuple[float, float, np.ndarray]:
    """Fit the exponential model to MSD data.

    Uses ``scipy.optimize.curve_fit`` (nonlinear least squares).

    Parameters
    ----------
    taus:
        Time delays (seconds), shape (n_tau,).
    MSD:
        Mean-square displacement values, shape (n_tau,).
    p0:
        Initial guess ``(Â, T̂)``.  Defaults match the MATLAB ``StartPoint``.
    bounds:
        Parameter bounds passed to ``curve_fit``.

    Returns
    -------
    A_hat:
        Fitted plateau amplitude.
    T_hat:
        Fitted relaxation time (seconds).
    MSD_fit:
        Model MSD evaluated at *taus*, shape (n_tau,).

    Raises
    ------
    RuntimeError
        If ``curve_fit`` does not converge.
    """
    taus = np.asarray(taus, dtype=float)
    MSD = np.asarray(MSD, dtype=float)

    popt, _ = curve_fit(
        _msd_model,
        taus,
        MSD,
        p0=list(p0),
        bounds=bounds,
        maxfev=10_000,
    )
    A_hat, T_hat = float(popt[0]), float(popt[1])
    MSD_fit = _msd_model(taus, A_hat, T_hat)
    return A_hat, T_hat, MSD_fit


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------


def estimate_diffusion(
    x: np.ndarray,
    tau_values: np.ndarray,
    dt_sim: float,
    i_start: int = 5000,
    p0: tuple[float, float] = (0.1, 0.01),
) -> dict:
    """Compute MSD, fit the exponential model, and return the diffusion coefficient *c*.

    Parameters
    ----------
    x:
        Amplitude envelope time-series.
    tau_values:
        Delay times τ (seconds).
    dt_sim:
        Simulation time step (seconds).
    i_start:
        Transient skip index.
    p0:
        Initial guess ``(Â, T̂)`` for the MSD fit.

    Returns
    -------
    dict with keys:
        ``taus``      — effective delay times (s)
        ``MSD``       — empirical MSD
        ``MSD_fit``   — fitted MSD
        ``A_hat``     — plateau amplitude
        ``T_hat``     — relaxation time (s)
        ``c``         — diffusion coefficient  Â/(2T̂)  (Eq. 7)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # A short synthetic example (not a real Langevin signal)
    >>> x = np.abs(rng.normal(1.0, 0.2, 50_000))
    >>> tau = np.linspace(2.5e-4, 0.05, 40)
    >>> res = estimate_diffusion(x, tau, dt_sim=5e-5, i_start=100)
    >>> 'c' in res
    True
    """
    taus, msd = compute_msd(x, tau_values, dt_sim, i_start=i_start)
    A_hat, T_hat, msd_fit = fit_msd(taus, msd, p0=p0)
    c = A_hat / (2.0 * T_hat)

    return {
        "taus": taus,
        "MSD": msd,
        "MSD_fit": msd_fit,
        "A_hat": A_hat,
        "T_hat": T_hat,
        "c": c,
    }
