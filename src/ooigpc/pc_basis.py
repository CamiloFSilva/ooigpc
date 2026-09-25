"""
pc_basis.py — Polynomial chaos (PC) basis for 1-D Gaussian random variable.

The random variable ξ ~ N(0, 1) is expanded in the probabilist's Hermite
polynomials (He_k), called Ψ_k throughout the paper:

    Ψ_0(ξ) = 1
    Ψ_1(ξ) = ξ
    Ψ_2(ξ) = ξ² - 1
    Ψ_3(ξ) = ξ³ - 3ξ
    …

They satisfy the orthogonality relation (inner product w.r.t. N(0,1)):

    ⟨Ψ_i, Ψ_j⟩ = δ_{ij} · i!          (Eq. S-1, Suppl. §2)

The gPC tensors needed by the identification algorithm are

    C_{ijk}   = ⟨Ψ_i Ψ_j Ψ_k⟩          3rd-order  (Eq. S-4, Suppl. §3)
    D_{ijkl}  = ⟨Ψ_i Ψ_j Ψ_k Ψ_l⟩      4th-order  (Eq. S-5, Suppl. §3)

Both are evaluated by Gauss–Hermite quadrature.

References
----------
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.
Supplementary material §2–3.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite_e import hermeval, hermegauss

__all__ = [
    "hermite_poly",
    "hermite_norm",
    "inner_product",
    "gpc_tensor_3",
    "gpc_tensor_4",
    "PCBasis",
]

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def hermite_poly(k: int, xi: np.ndarray) -> np.ndarray:
    """Evaluate the probabilist's Hermite polynomial Ψ_k at points *xi*.

    The coefficients for ``numpy.polynomial.hermite_e.hermeval`` are simply
    ``c = [0, 0, …, 1]`` with a 1 in position *k*.

    Parameters
    ----------
    k:
        Polynomial order (≥ 0).
    xi:
        Evaluation points (any shape).

    Returns
    -------
    numpy.ndarray
        Ψ_k(xi), same shape as *xi*.

    Examples
    --------
    >>> import numpy as np
    >>> xi = np.array([0.0, 1.0, -1.0])
    >>> hermite_poly(2, xi)   # ξ² - 1
    array([-1.,  0.,  0.])
    """
    if k < 0:
        raise ValueError(f"Polynomial order k must be ≥ 0, got {k}.")
    coeffs = np.zeros(k + 1)
    coeffs[k] = 1.0
    return hermeval(np.asarray(xi, dtype=float), coeffs)


def hermite_norm(k: int) -> float:
    """Return ‖Ψ_k‖² = k!   (Eq. S-1).

    Parameters
    ----------
    k:
        Polynomial order (≥ 0).
    """
    return float(math.factorial(k))


def inner_product(
    f: np.ndarray,
    g: np.ndarray,
    xi: np.ndarray,
    w: np.ndarray,
) -> float:
    """Compute ⟨f, g⟩ = ∫ f(ξ) g(ξ) f_ξ(ξ) dξ by Gauss–Hermite quadrature.

    ``hermegauss`` returns nodes *xi* and weights *w* normalised so that

        Σ_q w_q f(xi_q) ≈ ∫ f(ξ) exp(-ξ²/2) / sqrt(2π) dξ

    Parameters
    ----------
    f, g:
        Function values evaluated at the quadrature nodes *xi*.
    xi:
        Quadrature nodes (output of ``hermegauss``).
    w:
        Quadrature weights (output of ``hermegauss``).

    Returns
    -------
    float
        Approximation of ⟨f, g⟩.
    """
    return float(np.dot(w, f * g))


# ---------------------------------------------------------------------------
# Gauss–Hermite quadrature nodes and weights
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _gauss_hermite(n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature nodes and N(0,1)-normalised weights.

    ``numpy.polynomial.hermite_e.hermegauss`` returns weights for the
    physicist's inner product with weight exp(-x²).  We re-normalise so that
    the weights integrate against the N(0,1) PDF:

        w̃_q = w_q · exp(xi_q² / 2) / sqrt(2π)

    This matches the paper's definition  ⟨·,·⟩ = ∫ · f_ξ dξ.
    """
    xi, w = hermegauss(n_quad)
    # hermegauss weights satisfy: Σ w_q f(x_q) ≈ ∫ f(x) exp(-x²/2) dx
    # Our inner product uses the N(0,1) PDF = exp(-x²/2)/√(2π),
    # so divide by √(2π) to get the correct normalisation.
    w = w / np.sqrt(2.0 * np.pi)
    return xi, w


# ---------------------------------------------------------------------------
# Tensor computation
# ---------------------------------------------------------------------------


def gpc_tensor_3(
    n_modes: int,
    n_quad: int | None = None,
) -> np.ndarray:
    """Compute the 3rd-order gPC tensor  C_{ijk} = ⟨Ψ_i Ψ_j Ψ_k⟩.

    Parameters
    ----------
    n_modes:
        Number of PC modes (polynomial orders 0 … n_modes-1).
    n_quad:
        Number of Gauss–Hermite quadrature points.  Defaults to
        ``3 * n_modes`` which is exact for products up to order
        ``3*(n_modes-1)``.

    Returns
    -------
    numpy.ndarray, shape (n_modes, n_modes, n_modes)
        C[i, j, k] = ⟨Ψ_i Ψ_j Ψ_k⟩.

    Notes
    -----
    Corresponds to ``Ctensor(i,j,k)`` in the MATLAB script
    ``Tensor_1D_comput.m`` and Eq. (S-4) in the supplementary material.
    """
    if n_quad is None:
        n_quad = max(3 * n_modes, 10)

    xi, w = _gauss_hermite(n_quad)

    # Pre-evaluate all basis polynomials at the quadrature nodes
    psi = np.stack([hermite_poly(k, xi) for k in range(n_modes)], axis=0)
    # psi[k, q] = Ψ_k(xi_q)

    C = np.zeros((n_modes, n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                C[i, j, k] = np.dot(w, psi[i] * psi[j] * psi[k])

    return C


def gpc_tensor_4(
    n_modes: int,
    n_quad: int | None = None,
) -> np.ndarray:
    """Compute the 4th-order gPC tensor  D_{ijkl} = ⟨Ψ_i Ψ_j Ψ_k Ψ_l⟩.

    Parameters
    ----------
    n_modes:
        Number of PC modes.
    n_quad:
        Number of quadrature points.  Defaults to ``4 * n_modes``.

    Returns
    -------
    numpy.ndarray, shape (n_modes, n_modes, n_modes, n_modes)
        D[i, j, k, l] = ⟨Ψ_i Ψ_j Ψ_k Ψ_l⟩.

    Notes
    -----
    Corresponds to ``Dtensor{k,l}(i,j)`` in ``Tensor_1D_comput.m`` and
    Eq. (S-5) in the supplementary material.
    """
    if n_quad is None:
        n_quad = max(4 * n_modes, 12)

    xi, w = _gauss_hermite(n_quad)

    psi = np.stack([hermite_poly(k, xi) for k in range(n_modes)], axis=0)

    D = np.zeros((n_modes, n_modes, n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            ij = psi[i] * psi[j]
            for k in range(n_modes):
                ijk = ij * psi[k]
                for l in range(n_modes):
                    D[i, j, k, l] = np.dot(w, ijk * psi[l])

    return D


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------


class PCBasis:
    """Polynomial chaos basis for a single N(0,1) random variable.

    Wraps the tensor pre-computation and provides evaluation helpers so
    that modules downstream (``igpc.py``, ``correction.py``) can share a
    single pre-computed basis object.

    Parameters
    ----------
    n_modes:
        Number of PC modes to retain (orders 0 to n_modes-1).
        For the 1-D Langevin identification problem the paper uses
        ``n_modes = 4``.
    n_quad:
        Quadrature points for tensor integrals.  ``None`` → auto.

    Examples
    --------
    >>> basis = PCBasis(n_modes=4)
    >>> basis.norm          # [0! 1! 2! 3!]
    array([1., 1., 2., 6.])
    >>> basis.C[1, 1, 2]   # ⟨Ψ₁ Ψ₁ Ψ₂⟩ = 2
    2.0
    """

    def __init__(self, n_modes: int = 4, n_quad: int | None = None) -> None:
        self.n_modes = n_modes
        self.n_quad = n_quad

        # Norms  ‖Ψ_k‖² = k!
        self.norm = np.array([hermite_norm(k) for k in range(n_modes)])

        # Quadrature nodes / weights (cached)
        _nq = n_quad if n_quad is not None else max(4 * n_modes, 12)
        self.xi, self.w = _gauss_hermite(_nq)

        # Pre-evaluated basis at quadrature nodes,  shape (n_modes, n_quad)
        self.psi_quad = np.stack(
            [hermite_poly(k, self.xi) for k in range(n_modes)], axis=0
        )

        # gPC tensors (computed lazily via properties)
        self._C: np.ndarray | None = None
        self._D: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Tensor properties (lazy, cached on first access)
    # ------------------------------------------------------------------

    @property
    def C(self) -> np.ndarray:
        """3rd-order tensor  C[i,j,k] = ⟨Ψ_i Ψ_j Ψ_k⟩."""
        if self._C is None:
            self._C = gpc_tensor_3(self.n_modes, self.n_quad)
        return self._C

    @property
    def D(self) -> np.ndarray:
        """4th-order tensor  D[i,j,k,l] = ⟨Ψ_i Ψ_j Ψ_k Ψ_l⟩."""
        if self._D is None:
            self._D = gpc_tensor_4(self.n_modes, self.n_quad)
        return self._D

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def eval(self, k: int, xi: np.ndarray) -> np.ndarray:
        """Evaluate Ψ_k at arbitrary points *xi*."""
        return hermite_poly(k, xi)

    def inner(self, f: np.ndarray, g: np.ndarray) -> float:
        """⟨f, g⟩ using this basis's quadrature rule.

        *f* and *g* must be evaluated at ``self.xi``.
        """
        return inner_product(f, g, self.xi, self.w)

    def project(self, f: np.ndarray) -> np.ndarray:
        """Project function values *f* (at ``self.xi``) onto the PC basis.

        Returns
        -------
        numpy.ndarray, shape (n_modes,)
            PC coefficients a_k = ⟨f, Ψ_k⟩ / ‖Ψ_k‖².
        """
        f = np.asarray(f, dtype=float)
        coeffs = np.array(
            [self.inner(f, self.psi_quad[k]) / self.norm[k] for k in range(self.n_modes)]
        )
        return coeffs

    def __repr__(self) -> str:
        return f"PCBasis(n_modes={self.n_modes})"
