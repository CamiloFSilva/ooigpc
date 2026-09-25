"""
solvers/langevin.py — Discrete Langevin SDE integrator for the amplitude envelope.

The amplitude envelope A(t) of a self-excited thermoacoustic oscillator
satisfies the Itô SDE (Eq. 1):

    dA = (ν·A − (κ/8)·A³ + c/A) dt  +  √(2c) dW

Discretised with Euler–Maruyama (Eq. 3 in the paper / MATLAB V10):

    A_{n+1} = A_n  +  a·Δt·A_n  −  b·Δt·A_n³  +  (c/A_n)·Δt
              + √(2c·Δt) · ξ_n

where  a = ν,  b = κ/8,  c = Γ/(4ω₀²)  and  ξ_n ~ TN(0, 1; ±4.5)
(truncated normal to avoid extreme noise realisations).

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.  Eq. 3 and
Supplementary material §1.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "simulate_langevin",
    "langevin_params",
]


def langevin_params(
    nu: float,
    kappa: float,
    Gamma: float,
    omega: float,
) -> tuple[float, float, float]:
    """Convert physical parameters to internal Langevin coefficients.

    Parameters
    ----------
    nu:
        Linear growth rate ν (s⁻¹).
    kappa:
        Nonlinear damping coefficient κ (Pa⁻²·s⁻¹).
    Gamma:
        Noise intensity Γ (Pa²·s⁻³).
    omega:
        Angular eigenfrequency ω₀ = 2π·f₀ (rad/s).

    Returns
    -------
    a, b, c:
        Internal coefficients where  a = ν,  b = κ/8,  c = Γ/(4ω₀²).
    """
    a = float(nu)
    b = float(kappa) / 8.0
    c = float(Gamma) / (4.0 * float(omega) ** 2)
    return a, b, c


def simulate_langevin(
    nu: float,
    kappa: float,
    Gamma: float,
    omega: float,
    N: int = 2_000_000,
    dt: float = 5e-5,
    x0: float = 0.65,
    seed: int | None = None,
    noise_truncation: float = 4.5,
) -> np.ndarray:
    """Integrate the discrete Langevin SDE for the amplitude envelope A(t).

    Parameters
    ----------
    nu:
        Linear growth rate ν (s⁻¹).
    kappa:
        Nonlinear damping coefficient κ (Pa⁻²·s⁻¹).
    Gamma:
        Noise intensity Γ (Pa²·s⁻³).
    omega:
        Angular eigenfrequency ω₀ (rad/s).
    N:
        Number of time steps.
    dt:
        Time step Δt (seconds).
    x0:
        Initial amplitude A(0).
    seed:
        Random seed for reproducibility.
    noise_truncation:
        Truncation limit for the noise distribution (σ units).
        MATLAB uses ±4.5.

    Returns
    -------
    numpy.ndarray, shape (N+1,)
        Amplitude envelope time series A(t_0), A(t_1), …, A(t_N).

    Notes
    -----
    Noise is drawn from a truncated normal TN(0, 1; ±*noise_truncation*)
    using rejection sampling.  This is numerically equivalent to MATLAB's
    ``makedist('Normal'); truncate(pd, -4.5, 4.5)``.

    Examples
    --------
    >>> import numpy as np
    >>> x = simulate_langevin(nu=16, kappa=3.4, Gamma=1e7,
    ...                       omega=2*np.pi*120, N=100_000, dt=5e-5, seed=0)
    >>> x.shape
    (100001,)
    >>> x.min() > 0   # envelope stays positive for these parameters
    True
    """
    a, b, c = langevin_params(nu, kappa, Gamma, omega)
    s1 = float(np.sqrt(2.0 * c * dt))

    rng = np.random.default_rng(seed)

    # Truncated normal noise (rejection sampling)
    noise = _truncated_normal(rng, N, truncation=noise_truncation)

    x = np.empty(N + 1, dtype=float)
    x[0] = float(x0)

    for k in range(N):
        xk = x[k]
        x[k + 1] = (
            xk
            + a * dt * xk
            - b * dt * xk**3
            + (c / xk) * dt
            + s1 * noise[k]
        )

    return x


def _truncated_normal(
    rng: np.random.Generator,
    n: int,
    truncation: float = 4.5,
) -> np.ndarray:
    """Draw *n* samples from TN(0, 1; ±truncation) by rejection sampling."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        needed = n - filled
        # Oversample by ~5% to reduce loop iterations
        batch = rng.standard_normal(int(needed * 1.1) + 100)
        batch = batch[np.abs(batch) <= truncation]
        take = min(len(batch), needed)
        out[filled : filled + take] = batch[:take]
        filled += take
    return out
