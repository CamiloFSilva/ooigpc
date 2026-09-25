"""
igpc.py — Output-only identification via iterative gPC (igPC).

Given the amplitude envelope A(t) and the diffusion coefficient *c*
(from :mod:`ooigpc.diffusion`), this module recovers the three physical
parameters of the Langevin SDE

    dA = (ν·A  −  (κ/8)·A³  +  c/A) dt  +  √(2c) dW       (Eq. 1)

via the explicit identification formulae derived in the paper:

    ν  =  [9σ⁴ + 2μ²σ² − μ⁴]  /  [2μ²σ²(σ² − μ²)]  ×  c   (Eq. 10)

    κ/8 =  [3σ² − μ²]  /  [2μ²σ²(σ² − μ²)/2]  ×  c          (Eq. 11)

where μ = ⟨A⟩ and σ is the corrected standard deviation from
:mod:`ooigpc.correction`.

The physical parameters are then:
    ν   = a  (linear growth rate, s⁻¹)
    κ   = 8·b  (nonlinearity coefficient, Pa⁻²·s⁻¹)
    Γ   = c · 4ω₀²  (noise intensity, Pa²·s⁻³)

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.  Eqs. 10–11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .correction import CorrectionResult, apply_correction
from .diffusion import estimate_diffusion

__all__ = [
    "IdentificationResult",
    "identify",
    "identify_from_signal",
]


@dataclass
class IdentificationResult:
    """Container for the igPC identification output.

    Attributes
    ----------
    nu:
        Linear growth rate ν (s⁻¹).  Positive = linearly unstable.
    kappa:
        Nonlinearity coefficient κ (Pa⁻²·s⁻¹).
    Gamma:
        Noise intensity Γ (Pa²·s⁻³).  Requires *omega* to be provided.
    a:
        a = ν (internal Langevin coefficient).
    b:
        b = κ/8 (internal Langevin coefficient).
    c:
        Diffusion coefficient c = Γ/(4ω₀²).
    mu:
        Sample mean of the amplitude envelope.
    sigma:
        Corrected standard deviation entering Eqs. (10)–(11).
    correction:
        Full :class:`~ooigpc.correction.CorrectionResult` object.
    """
    nu: float
    kappa: float
    Gamma: float | None
    a: float
    b: float
    c: float
    mu: float
    sigma: float
    correction: CorrectionResult


def _identification_from_moments(
    mu: float,
    sigma: float,
    c: float,
) -> tuple[float, float]:
    """Apply Eqs. (10)–(11) given corrected moments and diffusion coefficient.

    Parameters
    ----------
    mu, sigma:
        Mean and corrected std of the amplitude envelope.
    c:
        Diffusion coefficient.

    Returns
    -------
    a, b:
        Langevin coefficients (a = ν, b = κ/8).

    Raises
    ------
    ValueError
        If ``σ² = μ²`` (denominator vanishes).
    """
    mu2 = mu**2
    sigma2 = sigma**2
    denom = mu2 * sigma2 * (sigma2 - mu2)

    if abs(denom) < 1e-30:
        raise ValueError(
            f"Denominator μ²σ²(σ²−μ²) ≈ 0 (μ={mu:.4g}, σ={sigma:.4g}). "
            "Cannot identify parameters — check that σ ≠ μ."
        )

    # Eq. (10): ν
    A1 = 0.5 * (9 * sigma2**2 + 2 * mu2 * sigma2 - mu2**2)
    a = (A1 / denom) * c

    # Eq. (11): κ/8
    B1 = 0.5 * (3 * sigma2 - mu2)
    b = (B1 / denom) * c

    return a, b


def identify(
    amplitudes: np.ndarray,
    c: float,
    omega: float | None = None,
) -> IdentificationResult:
    """Identify Langevin parameters from amplitude statistics and *c*.

    This is the core identification step.  It applies the two-level
    correction from :mod:`ooigpc.correction` and then Eqs. (10)–(11).

    Parameters
    ----------
    amplitudes:
        1-D array of amplitude envelope samples A(t) > 0.
    c:
        Diffusion coefficient estimated from the MSD (see
        :func:`ooigpc.diffusion.estimate_diffusion`).
    omega:
        Angular frequency ω₀ = 2π·f₀ (rad/s).  Required to compute Γ;
        pass ``None`` to skip the Γ → κ conversion.

    Returns
    -------
    IdentificationResult
    """
    corr = apply_correction(amplitudes)
    a, b = _identification_from_moments(corr.mu, corr.sigma, c)

    nu = a
    kappa = 8.0 * b
    Gamma = c * 4.0 * omega**2 if omega is not None else None

    return IdentificationResult(
        nu=nu,
        kappa=kappa,
        Gamma=Gamma,
        a=a,
        b=b,
        c=c,
        mu=corr.mu,
        sigma=corr.sigma,
        correction=corr,
    )


def identify_from_signal(
    x: np.ndarray,
    tau_values: np.ndarray,
    dt_sim: float,
    omega: float | None = None,
    i_start: int = 5000,
    msd_p0: tuple[float, float] = (0.1, 0.01),
) -> dict:
    """End-to-end identification from a raw amplitude-envelope time series.

    Runs :func:`~ooigpc.diffusion.estimate_diffusion` followed by
    :func:`identify` and returns a combined result dictionary.

    Parameters
    ----------
    x:
        1-D amplitude envelope time-series.
    tau_values:
        Time delays τ for the MSD (seconds).
    dt_sim:
        Time step of the simulation / measurement (seconds).
    omega:
        Angular frequency ω₀ (rad/s).  Pass ``None`` to skip Γ estimation.
    i_start:
        Samples to skip at the start (transient removal).
    msd_p0:
        Initial guess ``(Â, T̂)`` for the MSD exponential fit.

    Returns
    -------
    dict
        Contains all MSD fit results (``taus``, ``MSD``, ``MSD_fit``,
        ``A_hat``, ``T_hat``, ``c``) plus ``identification``
        (:class:`IdentificationResult`) and ``amplitudes_used`` (the
        positive-valued subset of *x*).

    Examples
    --------
    >>> from ooigpc.solvers.langevin import simulate_langevin
    >>> import numpy as np
    >>> x = simulate_langevin(nu=16, kappa=3.4, Gamma=1e7, omega=2*np.pi*120,
    ...                       N=500_000, dt=5e-5, seed=0)
    >>> tau = np.arange(2.5e-4, 0.05, 2.5e-4)
    >>> out = identify_from_signal(x, tau, dt_sim=5e-5, omega=2*np.pi*120)
    >>> print(f"ν = {out['identification'].nu:.1f} s⁻¹")
    """
    x = np.asarray(x, dtype=float)

    # Step 1: diffusion coefficient from MSD (Eq. 7)
    msd_result = estimate_diffusion(
        x, tau_values, dt_sim, i_start=i_start, p0=msd_p0
    )

    # Step 2: positive envelope values for statistics
    amp_positive = x[i_start:][x[i_start:] > 0]

    # Step 3: identification (Eqs. 10–11)
    ident = identify(amp_positive, msd_result["c"], omega=omega)

    return {**msd_result, "identification": ident, "amplitudes_used": amp_positive}
