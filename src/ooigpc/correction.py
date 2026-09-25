"""
correction.py — Two-level correction for the non-Gaussian amplitude PDF.

For a purely Gaussian ξ ~ N(0,1) the PC representation is exact.  In
practice the amplitude envelope A(t) is not Gaussian, so the raw sample
standard deviation std_A slightly under-estimates the corrected value σ that
enters Eqs. (10)–(11).

The correction separates two regimes (Fig. 4 in the paper):

Region 1 — marginally stable  (std_A > σ_crit = 0.280 μ)
    Linear interpolation between two empirical correction curves:

        c1_std = 0.144 · std_A + 0.0126    (upper bound, Eq. S-8)
        c2_std = −0.035 · std_A            (lower bound, Eq. S-9)

    with weights β = (std_A − σ₂) / (σ₁ − σ₂),
    where σ₁ = 0.423 μ and σ₂ = 0.280 μ.

    Result:  σ = std_A + std_correc

Region 2 — linearly unstable  (std_A ≤ σ_crit)
    The skewness of the PDF determines the third central moment:

        γ = skewness · std_A³

    which is used to solve the cubic (Eq. S-7):

        4 δ⁴ − 6 std_A² δ² + γ δ = 0  →  roots for δ (std_correc)

    Result:  σ = √(std_A² − 2 · std_correc²)

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.
Supplementary material §5 (correction factors, Eqs. S-7–S-9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import skew as scipy_skew

__all__ = [
    "CorrectionResult",
    "apply_correction",
]

# Empirical thresholds (Suppl. §5, Fig. 4)
_SIGMA_CRIT_RATIO = 0.280   # σ_crit / μ
_SIGMA_1_RATIO = 0.423      # σ₁ / μ   (upper boundary)
_SIGMA_2_RATIO = 0.280      # σ₂ / μ   (lower boundary) == _SIGMA_CRIT_RATIO


@dataclass
class CorrectionResult:
    """Container for the output of :func:`apply_correction`.

    Attributes
    ----------
    sigma:
        Corrected standard deviation entering Eqs. (10)–(11).
    std_correc:
        Additive (or sign-aware) correction that was applied.
    region:
        1 for marginally stable, 2 for linearly unstable.
    mu:
        Sample mean of the amplitude envelope.
    std_raw:
        Raw sample standard deviation before correction.
    """
    sigma: float
    std_correc: float
    region: int
    mu: float
    std_raw: float


def apply_correction(
    amplitudes: np.ndarray,
) -> CorrectionResult:
    """Apply the two-level statistical correction to the amplitude envelope.

    Parameters
    ----------
    amplitudes:
        1-D array of amplitude envelope samples A(t) > 0.

    Returns
    -------
    CorrectionResult

    Raises
    ------
    ValueError
        If *amplitudes* contains non-positive values or is too short.

    Notes
    -----
    The skewness in Region 2 is computed with ``scipy.stats.skew``
    (Fisher's definition, bias-corrected), matching MATLAB's ``skewness``
    function for large samples.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> A = np.abs(rng.normal(1.0, 0.1, 100_000))
    >>> res = apply_correction(A)
    >>> res.region
    2
    """
    amp = np.asarray(amplitudes, dtype=float)
    if np.any(amp <= 0):
        raise ValueError("All amplitude values must be positive (envelope).")
    if amp.size < 10:
        raise ValueError("Need at least 10 samples for the correction.")

    mu = float(np.mean(amp))
    std = float(np.std(amp, ddof=1))

    sigma_crit = _SIGMA_CRIT_RATIO * mu

    if std > sigma_crit:
        # ------------------------------------------------------------------
        # Region 1 — marginally stable  (Suppl. Eq. S-8, S-9)
        # ------------------------------------------------------------------
        c1_std = 0.144 * std + 0.0126
        c2_std = -0.035 * std

        sigma_1 = _SIGMA_1_RATIO * mu
        sigma_2 = _SIGMA_2_RATIO * mu

        beta = (std - sigma_2) / (sigma_1 - sigma_2)
        std_correc = (1.0 - beta) * c2_std + beta * c1_std

        sigma = std + std_correc
        region = 1

    else:
        # ------------------------------------------------------------------
        # Region 2 — linearly unstable  (Suppl. Eq. S-7)
        # ------------------------------------------------------------------
        sk = float(scipy_skew(amp, bias=False))
        gamma = sk * std**3

        # Cubic: 4δ⁴ - 6·std²·δ² + γ·δ = 0
        # Factoring out δ: δ · (4δ³ - 6·std²·δ + γ) = 0
        # Solve the depressed cubic:  4δ³ - 6·std²·δ + γ = 0
        poly_coeffs = [4.0, 0.0, -6.0 * std**2, gamma]
        roots = np.roots(poly_coeffs)

        # Take the third real root (index 2), matching MATLAB's ``x2(3)``
        # (roots sorted by numpy in descending magnitude order)
        real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-10 * std])
        if len(real_roots) == 0:
            real_roots = np.real(roots)

        # Consistent with MATLAB: use index 2 (third element)
        std_correc = 2.6 * float(np.real(roots[2]))

        var_corrected = std**2 - 2.0 * std_correc**2
        if var_corrected <= 0:
            # Fallback: clamp to a tiny positive value
            var_corrected = 1e-12
        sigma = float(np.sqrt(var_corrected))
        region = 2

    return CorrectionResult(
        sigma=sigma,
        std_correc=std_correc,
        region=region,
        mu=mu,
        std_raw=std,
    )
