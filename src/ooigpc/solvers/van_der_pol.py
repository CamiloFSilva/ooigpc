"""
solvers/van_der_pol.py — Stochastic Van der Pol oscillator integrator.

The stochastically forced Van der Pol (VdP) equation (Eq. 2 of the paper):

    η̈ − 2ν·η̇ + κ·η·η̇ + ω₀²·η = Ξ(t)

where Ξ(t) is Gaussian white noise with intensity Γ.

Written as a first-order system:

    d[η  ]   [η̇                        ]        [0         ]
    d[η̇  ] = [-ω₀²η + 2ν·η̇ − κ·η·η̇  ] dt + [√(2Γ) dW  ]

Discretised with the Euler–Maruyama scheme.

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.  Eq. 2.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "simulate_van_der_pol",
    "amplitude_envelope",
]


def simulate_van_der_pol(
    nu: float,
    kappa: float,
    Gamma: float,
    omega: float,
    N: int = 2_000_000,
    dt: float = 5e-5,
    eta0: float = 0.01,
    eta_dot0: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the stochastic Van der Pol ODE.

    Parameters
    ----------
    nu:
        Linear growth rate ν (s⁻¹).
    kappa:
        Nonlinear coupling coefficient κ.
    Gamma:
        White-noise intensity Γ (Pa²·s⁻³).
    omega:
        Angular eigenfrequency ω₀ = 2π·f₀ (rad/s).
    N:
        Number of time steps.
    dt:
        Time step Δt (seconds).
    eta0, eta_dot0:
        Initial displacement and velocity.
    seed:
        Random seed.

    Returns
    -------
    time:
        Time vector, shape (N+1,).
    eta:
        Displacement η(t), shape (N+1,).
    eta_dot:
        Velocity η̇(t), shape (N+1,).

    Examples
    --------
    >>> import numpy as np
    >>> t, eta, _ = simulate_van_der_pol(
    ...     nu=16, kappa=3.4, Gamma=1e7, omega=2*np.pi*120,
    ...     N=50_000, dt=5e-5, seed=0)
    >>> t.shape
    (50001,)
    """
    rng = np.random.default_rng(seed)
    sigma_noise = float(np.sqrt(2.0 * Gamma * dt))

    eta = np.empty(N + 1, dtype=float)
    eta_dot = np.empty(N + 1, dtype=float)
    time = np.linspace(0.0, dt * N, N + 1)

    eta[0] = float(eta0)
    eta_dot[0] = float(eta_dot0)

    noise = rng.standard_normal(N)

    omega2 = omega**2
    for k in range(N):
        e = eta[k]
        ed = eta_dot[k]
        eta_ddot = -omega2 * e + 2.0 * nu * ed - kappa * e * ed
        eta[k + 1] = e + ed * dt
        eta_dot[k + 1] = ed + eta_ddot * dt + sigma_noise * noise[k]

    return time, eta, eta_dot


def amplitude_envelope(
    eta: np.ndarray,
    eta_dot: np.ndarray,
    omega: float,
) -> np.ndarray:
    """Compute the instantaneous amplitude envelope from η and η̇.

    Uses the standard analytic-signal approximation:

        A(t) = √[ η(t)² + (η̇(t)/ω₀)² ]

    Parameters
    ----------
    eta:
        Displacement time series.
    eta_dot:
        Velocity time series.
    omega:
        Angular eigenfrequency ω₀ (rad/s).

    Returns
    -------
    numpy.ndarray
        Amplitude envelope A(t), same shape as *eta*.
    """
    return np.sqrt(np.asarray(eta, dtype=float)**2
                   + (np.asarray(eta_dot, dtype=float) / omega)**2)
