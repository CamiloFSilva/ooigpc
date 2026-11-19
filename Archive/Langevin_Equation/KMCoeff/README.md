# KMCoeff — Kramers-Moyal identification scripts

This folder contains two scripts that implement the same KM identification
pipeline for the Langevin amplitude equation, but differ in **how the
transition moments are computed** from the simulated time series.

---

## The common pipeline

Both scripts follow the four-step procedure of Noiray & Schuermans (2013)
and Boujo & Noiray (2017):

1. Simulate the Itô Euler-Maruyama amplitude equation for a set of growth
   rates `ν` (100 realisations each).
2. Compute the first and second conditional transition moments M₁(τ, A)
   and M₂(τ, A) at a range of time lags τ.
3. Extrapolate M₁ and M₂ to τ→0 (via `extrapolate_KM_3coeff` and
   `extrapolate_KM`) to obtain the KM coefficients D₁(A) and D₂(A).
4. Average D₂ to estimate Γ; fit a least-squares Langevin model to D₁ to
   recover ν and κ.

They share identical physical parameters (κ = 3.4, Γ = 1e7, ω₀ = 2π×120),
time-step (dt = 5×10⁻⁵ s), binning (nbin = 30), lag set
(τ = 0.005 : 0.01 : 0.1 s), and extrapolation helpers.  Results are saved
to `../DATA/` with the same variable names (`params_true`, `params_hat`).

---

## The fundamental difference: how transition moments are computed

### `OOID_KM_Direct.m` — direct conditional-statistics

For each lag τ and each amplitude bin i, the script **directly collects
the set of conditional jumps**

    Δ_i(τ) = { A(t+τ) − A(t)  |  A(t) ∈ bin i }

using `discretize()` to assign every sample to a bin, then `find()` to
gather the matching pairs.  The transition moments follow immediately:

    M₁(τ, i) = mean(Δ_i) / τ
    M₂(τ, i) = mean(Δ_i²) / τ

**No intermediate matrix is stored.**  At each lag τ the script iterates
over bins only (nbin inner iterations), not over all sample pairs.

### `OOID_KM_PDFMatrix.m` — conditional PDF matrix

This script first constructs the full **conditional-jump probability
density matrix**

    PA(τ, i, j) = P( A(t+τ) ∈ bin j  |  A(t) ∈ bin i )

by counting all (reference, delayed) amplitude pairs and normalising each
row.  The transition moments are then computed as weighted sums over the
second index:

    M₁(τ, i) = Σⱼ (Abin_j − Abin_i) · PA(τ, i, j) / τ
    M₂(τ, i) = Σⱼ (Abin_j − Abin_i)² · PA(τ, i, j) / τ

**An explicit PDF matrix PA of size (nτ × nbin × nbin) is stored** before
any moments are computed.

---

## Practical comparison

| Aspect | `OOID_KM_Direct.m` | `OOID_KM_PDFMatrix.m` |
|---|---|---|
| Moment computation | Direct sample mean / variance | Weighted sums over PA matrix |
| Intermediate storage | None (bin index vector only) | PA(nτ, nbin, nbin) matrix |
| Inner loop over | Bins (nbin) | All pairs, then bins² for moments |
| Bin reference in model fit | Bin right-edges (Abin) | Bin right-edges (Abin) |
| Code origin | Original hand-written script | AI-assisted generalisation |

Both approaches are **mathematically equivalent** for a stationary ergodic
process and converge to the same KM coefficients with sufficient data.

**When to prefer the direct variant (`OOID_KM_Direct.m`):** the PA matrix is
never needed downstream; the direct approach avoids allocating it and is
slightly simpler to follow.

**When to prefer the PDF-matrix variant (`OOID_KM_PDFMatrix.m`):** if the
full conditional PDF PA is itself of interest (e.g. for visualisation or
higher-order moment checks), building it once and reusing it is more
efficient than recomputing conditional sets for each quantity.

---

## References

- Siegert, Friedrich, Peinke (1998). *Phys. Lett. A* 243, 275–280.
- Noiray & Schuermans (2013). *Int. J. Non-linear Mech.* 50, 152–163.
- Boujo & Noiray (2017). *Proc. R. Soc. A* 473, 20160894.
- Silva, C. F. (2026). *Output-only identification of thermoacoustic
  systems using polynomial chaos expansions.* Combust. Flame.
