# Langevin\_Equation

Numerical validation of output-only identification methods for the stochastic
Langevin amplitude equation

    dA/dt = ν·A − (κ/8)·A³ + c/A + ζ(t)

where ν is the linear growth rate, κ the nonlinear saturation coefficient,
c = Γ/(4ω₀²) the diffusion coefficient, and ζ a white-noise forcing.
Two identification methods are compared: **igPC** (polynomial-chaos moment
matching) and **KM** (Kramers-Moyal coefficients). Both are tested across
seven growth rates ν ∈ {0, 4, 8, 12, 16, 20, 24} rad/s with 100
Monte-Carlo realisations each.

Paper: Silva, C. F. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.

---

## Folder layout

```
Langevin_Equation/
├── igPC/           identification via igPC
├── KMCoeff/        identification via Kramers-Moyal coefficients
├── Plotting/       post-processing and figure scripts
├── DATA/           saved ensemble results (.mat, one file per ν and method)
└── DATA_2DMap/     2-D parameter-space sweep data for correction maps
```

---

## Scripts

### igPC/

| File | Description |
|---|---|
| `OOID_igPC.m` | Main igPC identification loop. Simulates the Langevin equation, fits the MSD to extract *c*, then applies the closed-form igPC formulas (Eqs. 10–11) to recover ν and κ. Optional two-level σ correction. Saves results to `DATA/`. |
| `OOID_CorrectionFactors.m` | Derives the two-level σ correction. Runs a dense 2-D sweep over (ν, κ, c) space in `'Calculate'` mode and plots the correction and error maps in `'Figures'` mode. Saves results to `DATA_2DMap/`. |
| `OOID_CorrectionFactors_Validation.m` | Validates the two-level correction on 120 randomly sampled (ν, κ) pairs using the analytical stationary PDF. No data files are read or written. |

### KMCoeff/

| File | Description |
|---|---|
| `OOID_KM_Direct.m` | KM identification via direct conditional statistics. For each time lag τ and amplitude bin, collects conditional jumps directly with `discretize` + `find`. No intermediate PDF matrix is stored. |
| `OOID_KM_PDFMatrix.m` | KM identification via the conditional-jump PDF matrix PA(τ, i, j). Builds the full matrix first, then computes moments as weighted sums. Equivalent results to the Direct variant; preferred when PA itself is of interest. |
| `extrapolate_KM.m` | Exponential extrapolation to τ→0 for the 2nd KM coefficient (diffusion). |
| `extrapolate_KM_3coeff.m` | Three-coefficient exponential extrapolation for the 1st KM coefficient (drift). |
| `README.md` | Detailed comparison of the two KM variants. |

### Plotting/

| File | Description |
|---|---|
| `Plot_Violin.m` | Violin plots of identified ν, κ, c comparing igPC and KM across all growth rates. Loads from `DATA/`. |
| `Plot_TimeSeriesHisto.m` | Sample time series and marginal amplitude PDFs for selected growth rates. Loads from `DATA/`. |

---

## Paper figure mapping

| Paper figure | Script | Output file(s) in `Plotting/` |
|---|---|---|
| Fig. 3(b) | `OOID_CorrectionFactors.m` | `map_nu_ref_c.pdf`, `map_kappa_ref_c.pdf` |
| Fig. 3(c) | `OOID_CorrectionFactors.m` (id = 1) | `map_Error1_nu.pdf`, `map_Error1_kappa.pdf` |
| Fig. 3(d) | `OOID_CorrectionFactors.m` (id = 2) | `map_Error2_nu.pdf`, `map_Error2_kappa.pdf` |
| Fig. 4(a) | `OOID_CorrectionFactors.m` | `map_CorrFactor_ref.pdf` |
| Fig. 4(b) | `OOID_CorrectionFactors.m` | `map_CorrFactor_model_Konst_2.6.pdf` |
| Fig. 4(c) | `OOID_CorrectionFactors.m` | `map_ErrorCorrFactor_2.6.pdf` |
| Fig. 5(a) | `OOID_CorrectionFactors.m` | `map_CorrFactor2_ref.pdf` |
| Fig. 5(b) | `OOID_CorrectionFactors.m` | `map_CorrFactor_model.pdf` |
| Fig. 7 | `Plot_TimeSeriesHisto.m` | `time_series0.pdf` |
| Fig. 8 | `OOID_igPC.m` | `MSD_Langevin_16.pdf`, `Extrapolation_Langevin_16.pdf` |
| Fig. 9 | `Plot_TimeSeriesHisto.m` | `Marg_PDF_*.pdf` |
| Fig. 10(a–c) | `Plot_Violin.m` | `param_nu_CorrectionTwoLevels.pdf`, `param_b_CorrectionTwoLevels.pdf`, `param_c_CorrectionTwoLevels.pdf` |
| Supp. Fig. 1 | `OOID_CorrectionFactors_Validation.m` | `Validation_nu_Supp1.pdf` |
| Supp. Fig. 2 | `OOID_CorrectionFactors_Validation.m` | `Validation_kappa_Supp2.pdf` |

---

## DATA/ naming convention

Files follow the pattern `nu_<ν>_<tag>_<method>.mat`:

| Tag | Meaning |
|---|---|
| `_fromKMCoeffs.mat` | Identified by `OOID_KM_Direct.m` |
| `_fromKMCoeffs_PDFMatrix.mat` | Identified by `OOID_KM_PDFMatrix.m` |
| `_CorrectionTwoLevels_fromigPC.mat` | Identified by `OOID_igPC.m` with two-level σ correction |
| `_fromigPC.mat` | Identified by `OOID_igPC.m` without correction |

Each `.mat` file stores two variables: `params_true` (ground truth) and
`params_hat` (ensemble of identified values, one row per realisation).
Columns are [ν, b, c].
