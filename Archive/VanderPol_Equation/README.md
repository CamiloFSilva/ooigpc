# VanderPol_Equation

MATLAB scripts for simulating the stochastic Van der Pol oscillator and identifying its parameters via the igPC and Kramers-Moyal methods.

## Physical model

The Van der Pol oscillator with additive coloured noise:

```
η̈ − 2ν η̇ + κ η² η̇ + ω₀² η = Ξ(t)
```

where `Ξ(t)` is Ornstein-Uhlenbeck (OU) coloured noise, `ω₀ = 2π × 120 rad/s`, and the growth rate `ν` and saturation coefficient `κ` are the parameters to be identified.

The instantaneous amplitude envelope `A(t)` is extracted from the signal `η(t)` via the Hilbert transform of the band-pass-filtered signal.  In the slow-flow approximation, `A(t)` obeys the Itô SDE (Langevin amplitude equation):

```
dA/dt = ν A − b A³ + c/A + ζ(t),   b = κ/8,   c = Γ/(4ω₀²)
```

## File overview

### Simulation

| File | Description |
|------|-------------|
| `vderPol_Solver.m` | Integrates the VdP SDE (Heun/RK2 scheme), extracts the amplitude envelope, and saves `Data_nu<nu>.mat` to `DATA/`. |
| `BP_filter_rect.m` | Band-pass filter utility used by `vderPol_Solver.m`. |

### Identification

| File | Description |
|------|-------------|
| `igPC/OOID_igPC.m` | igPC identification: estimates `c` via MSD fit, applies optional two-level `σ` correction, then recovers `ν` and `κ` from closed-form Eqs. 10-11. |
| `KMCoeff/OOID_KM_PDFMatrix.m` | Kramers-Moyal — PDF matrix method: builds the full conditional-jump PDF matrix `PA(τ, A_i, A_j)`, extrapolates drift and diffusion KM coefficients to `τ→0`, and fits the amplitude model. |
| `KMCoeff/OOID_KM_Direct.m` | Kramers-Moyal — direct method: bins the amplitude via `discretize`, computes conditional mean and variance for each bin directly, extrapolates to `τ→0`, and fits the amplitude model. |
| `KMCoeff/extrapolate_KM.m` | Exponential extrapolation utility for the 2nd KM coefficient. |
| `KMCoeff/extrapolate_KM_3coeff.m` | Three-coefficient exponential extrapolation for the 1st KM coefficient. |

### Diagnostics and plotting

| File | Description |
|------|-------------|
| `Plotting/Plot_MSD_StepConvergence.m` | Checks sensitivity of recovered `c`, `μ`, `σ`, `γ` to the MSD step size `dt = k·dt_sim`. Runs on VdP or experimental data. |
| `Plotting/Plot_MSD_TauMaxConvergence.m` | Checks sensitivity of recovered `c`, `μ`, `σ`, `γ` to the MSD fit window `τ_max`. Runs on VdP or experimental data. |
| `Plotting/Plot_EnsembleStats.m` | Collects mean and std of `μ` and `σ` across the `NumSim` replicates for each `ν` case; prints ensemble statistics. |
| `Plotting/Plot_AmplitudeHistograms.m` | Plots per-case marginal amplitude PDFs and a concatenated amplitude time series across all `ν` cases. |

## Data flow

```
vderPol_Solver.m
    └─► DATA/Data_nu<nu>.mat  (A, dt, Gamma, wn)
           ├─► igPC/OOID_igPC.m
           │       └─► DATA/nu_<nu>_<CASE>_<BIN>_fromigPC.mat
           │               ├─► Plotting/Plot_EnsembleStats.m
           │               └─► Plotting/Plot_AmplitudeHistograms.m
           ├─► KMCoeff/OOID_KM_PDFMatrix.m
           │       └─► DATA/nu_<nu>_fromKMCoeffs.mat
           └─► KMCoeff/OOID_KM_Direct.m
                   └─► DATA/nu_<nu>_fromKMDirect.mat
```

## Key conventions

- **Factor of 2**: The second transition moment satisfies `M₂(τ,A) → 2·D₂(A) = 2c` as `τ→0`. The raw PDF-weighted average of the KM second coefficient therefore estimates `2c`; dividing by 2 recovers the physical diffusion coefficient `c`.
- **Gamma formula**: `Γ = c · 4·ω₀²` (not `2·ω₀²`).
- **Growth rate units**: `ν` is in rad/s throughout.
- **MSD fit**: `MSD(τ) = MSD_amp · (1 − exp(−τ/τ_drift))`, then `c = MSD_amp / (2·τ_drift)`.

## Typical workflow

1. Set `nu_vec` and `NumSim` in `vderPol_Solver.m`; run to populate `DATA/`.
2. Run `igPC/OOID_igPC.m` (or `KMCoeff/OOID_KM_PDFMatrix.m`) for the same `nu_vec`.
3. Use the `Plotting/` scripts to inspect histograms, ensemble statistics, and MSD convergence.

## Reference

Silva, C. F. (2026). *Output-only identification of thermoacoustic systems using polynomial chaos expansions.* Combustion and Flame.
