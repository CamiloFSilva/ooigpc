"""
solvers/van_der_pol.py — Stochastic Van der Pol oscillator integrator.

The stochastically forced Van der Pol (VdP) equation (Eq. 2 of the paper):

    η̈ − 2ν·η̇ + κ·η²·η̇ + ω₀²·η = Ξ(t)

where Ξ(t) is Gaussian white noise with intensity Γ.

Written as a first-order system:

    d[η  ]   [η̇                         ]        [0         ]
    d[η̇  ] = [-ω₀²η + 2ν·η̇ − κ·η²·η̇  ] dt + [√Γ dW     ]

Three time-stepping schemes are available:

``'symplectic'`` (default) — Semi-implicit (symplectic) Euler
    Velocity is updated first; then position is advanced with the *new*
    velocity.  This is symplectic for the conservative (Hamiltonian) part of
    the oscillator, so the scheme adds no spurious energy to the limit cycle.
    The observed mean amplitude matches the theoretical value A_eq = √(8ν/κ)
    to within ~0.1 %.

``'heun'`` — Explicit Runge-Kutta 2 (Heun / explicit trapezoidal)
    Two-stage scheme; det ≈ 1 + O(ω⁴Δt⁴) — effectively area-preserving.
    Matches the original paper MATLAB solver (vderPol_Solver.m).  Noise
    enters only in the corrector step (valid for small Δt).

``'euler-maruyama'`` — Standard Euler–Maruyama (explicit Euler)
    Position is advanced with the *old* velocity.  For a harmonic oscillator
    the explicit-Euler map is NOT area-preserving: it adds spurious energy
    ∝ (ωΔt)² per step, raising the effective growth rate to
    ν_eff = ν + ω²Δt/2 and shifting the limit-cycle amplitude to
    A_eq_eff = √(8ν_eff/κ) — about 37% above the true value for the default
    parameters (ω₀ = 2π·120 Hz, Δt = 5×10⁻⁵ s).  Keep this option only
    for backward-compatibility or comparison studies.

Two noise models are available (``noise_type`` parameter):

``'white'`` (default) — i.i.d. Gaussian white noise.
    Velocity increment per step: dΞ = √(Γ·Δt) · ξ,  ξ ~ N(0,1).

``'ou'`` — Ornstein-Uhlenbeck (colored) noise.
    dξ = −ξ/τ_c dt + √(Γ_drive)/τ_c dW.
    Γ_drive is calibrated so that the spectral power in the passband
    [f₀−BW/2, f₀+BW/2] equals Γ exactly:

        gamma_coeff = 2π·τ_c·BW / (atan(2π(f₀+BW/2)τ_c)
                                  − atan(2π(f₀−BW/2)τ_c))
        Γ_drive     = Γ / gamma_coeff · (1 + ω₀²τ_c²)

    Velocity increment per step: dΞ = ξ(t)·Δt  (force × timestep).
    Requires: tau_c and bandwidth_hz.

Note on the nonlinear term
--------------------------
The cubic term −κ·η²·η̇ is essential: it is the Van-der-Pol-type saturation
that produces the limit cycle at A_eq = √(8ν/κ) and, after slow-amplitude
averaging, yields the Langevin drift νA − (κ/8)A³ (Eq. 3 of the paper).
A quadratic term −κ·η·η̇ would average to zero over one oscillation cycle
and therefore cannot produce amplitude saturation.

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.  Eq. 2.
"""

from __future__ import annotations

import math

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
    method: str = "symplectic",
    noise_type: str = "white",
    tau_c: float = 5e-4,
    bandwidth_hz: float = 100.0,
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
    method:
        Time-stepping scheme: ``'symplectic'`` (default, fast, no energy
        drift), ``'heun'`` (RK2, matches original paper MATLAB solver), or
        ``'euler-maruyama'`` (legacy, biased amplitude).
    noise_type:
        Noise model: ``'white'`` (default, i.i.d. Gaussian) or ``'ou'``
        (Ornstein-Uhlenbeck colored noise with passband calibration).
    tau_c:
        OU correlation time τ_c (s).  Only used when ``noise_type='ou'``.
        Paper default: 5×10⁻⁴ s.
    bandwidth_hz:
        Full passband bandwidth BW (Hz) for OU noise calibration.  The OU
        driving intensity is set so that the spectral power in
        [f₀−BW/2, f₀+BW/2] equals Γ.  Only used when ``noise_type='ou'``.
        Paper default: 100 Hz (full bandwidth, i.e. ±50 Hz around f₀).

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
    omega2 = omega ** 2
    fn = omega / (2.0 * math.pi)   # eigenfrequency (Hz)
    nu2 = 2.0 * nu

    # ------------------------------------------------------------------ #
    #  Generate noise velocity increment  dxi[k]  for step k → k+1.      #
    #  For ALL integration schemes the increment is ADDED to eta_dot:     #
    #      eta_dot[k+1] += dxi[k]                                         #
    #  White noise: dxi ~ O(√dt);  OU: dxi = force × dt ~ O(dt).         #
    # ------------------------------------------------------------------ #
    noise_type = noise_type.lower()

    if noise_type == "white":
        # Noise coefficient √Γ (NOT √(2Γ) — the factor of 2 belongs to the
        # Langevin amplitude SDE and must not appear in the pressure equation).
        sigma_noise = math.sqrt(Gamma * dt)
        dxi = sigma_noise * rng.standard_normal(N)

    elif noise_type == "ou":
        # Calibrated OU forcing: Γ_drive chosen so that the spectral power
        # in [f₀−BW/2, f₀+BW/2] equals Γ.
        BW = float(bandwidth_hz)
        gamma_coeff = (
            2.0 * math.pi * tau_c * BW
            / (math.atan(2.0 * math.pi * (fn + BW / 2.0) * tau_c)
               - math.atan(2.0 * math.pi * (fn - BW / 2.0) * tau_c))
        )
        Gamma_drive = Gamma / gamma_coeff * (1.0 + omega2 * tau_c ** 2)
        sig_ou = math.sqrt(Gamma_drive) / tau_c * math.sqrt(dt)

        # Euler-Maruyama discretisation of  dξ = −ξ/τ_c dt + √Γ_drive/τ_c dW
        xi = np.empty(N, dtype=float)
        xi[0] = 0.0
        decay = 1.0 - dt / tau_c
        noise_raw = rng.standard_normal(N)
        for k in range(1, N):
            xi[k] = xi[k - 1] * decay + sig_ou * noise_raw[k]

        # velocity increment = force (acceleration) × Δt
        dxi = xi * dt

        print(
            f"  [OU noise]  gamma_coeff = {gamma_coeff:.4f},  "
            f"Gamma_drive = {Gamma_drive:.4e}"
        )

    else:
        raise ValueError(
            f"Unknown noise_type {noise_type!r}. "
            "Choose 'white' (default) or 'ou'."
        )

    # ------------------------------------------------------------------ #
    #  Time integration                                                    #
    # ------------------------------------------------------------------ #
    eta = np.empty(N + 1, dtype=float)
    eta_dot = np.empty(N + 1, dtype=float)
    time = np.linspace(0.0, dt * N, N + 1)

    eta[0] = float(eta0)
    eta_dot[0] = float(eta_dot0)

    method = method.lower()

    if method == "symplectic":
        # Semi-implicit (symplectic) Euler — area-preserving for harmonic part.
        # Velocity updated first with old position; position advanced with
        # the NEW velocity → no O((ωΔt)²) energy drift.
        for k in range(N):
            e = eta[k]
            ed = eta_dot[k]
            fdet = -omega2 * e + nu2 * ed - kappa * e ** 2 * ed
            eta_dot[k + 1] = ed + fdet * dt + dxi[k]
            eta[k + 1] = e + eta_dot[k + 1] * dt           # new velocity

    elif method == "heun":
        # Heun (RK2 / explicit trapezoidal).
        # Noise enters only the corrector (not the predictor); valid for small dt.
        for k in range(N):
            e1 = eta[k]
            ed1 = eta_dot[k]
            fdet1 = -omega2 * e1 + nu2 * ed1 - kappa * e1 ** 2 * ed1
            # Predictor (deterministic)
            e2 = e1 + dt * ed1
            ed2 = ed1 + dt * fdet1
            # Corrector (noise added once)
            fdet2 = -omega2 * e2 + nu2 * ed2 - kappa * e2 ** 2 * ed2
            eta[k + 1] = e1 + 0.5 * dt * (ed1 + ed2)
            eta_dot[k + 1] = ed1 + 0.5 * dt * (fdet1 + fdet2) + dxi[k]

    elif method == "euler-maruyama":
        # Standard Euler–Maruyama (explicit Euler for the deterministic part)
        # Position advanced with the OLD velocity — not symplectic.
        for k in range(N):
            e = eta[k]
            ed = eta_dot[k]
            fdet = -omega2 * e + nu2 * ed - kappa * e ** 2 * ed
            eta[k + 1] = e + ed * dt                        # old velocity
            eta_dot[k + 1] = ed + fdet * dt + dxi[k]

    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose 'symplectic' (default), 'heun', or 'euler-maruyama'."
        )

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
