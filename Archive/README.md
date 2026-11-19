# OOigPC — Output-Only Identification via Polynomial Chaos

MATLAB code accompanying:

> Silva, C. F. (2026).  
> *Output-only identification of thermoacoustic systems using polynomial chaos expansions.*  
> Combustion and Flame.

The repository demonstrates and validates two output-only system-identification methods — **igPC** (polynomial-chaos moment matching) and **KM** (Kramers-Moyal coefficients) — applied to stochastic amplitude equations arising in thermoacoustic systems.  Validation proceeds from a pedagogical tutorial through synthetic simulations to real combustion data.

---

## Repository layout

```
Matlab/
├── Examples/            PC algebraic operations tutorial (Supplementary Material)
├── Langevin_Equation/   Validation on the stochastic Langevin amplitude SDE
├── VanderPol_Equation/  Validation on the stochastic Van der Pol oscillator
└── Experiments/         Application to Noiray & Denisov (2017) experimental data
```

Each folder (except `Examples`) follows the same sub-structure:

```
<Folder>/
├── igPC/        identification via igPC
├── KMCoeff/     identification via Kramers-Moyal coefficients
├── Plotting/    post-processing and figure scripts
└── DATA/        saved results (.mat, written by the identification scripts)
```

Detailed per-folder documentation is in `Langevin_Equation/README.md` and
`VanderPol_Equation/README.md`.

---

## Paper figure index

| Figure | Script | Folder |
|--------|--------|--------|
| Fig. 1 | `Examples.m` | `Examples/` |
| Fig. 2 | `vderPol_Solver.m` | `VanderPol_Equation/` |
| Fig. 3(b–d) | `igPC/OOID_CorrectionFactors.m` | `Langevin_Equation/` |
| Fig. 4(a–c) | `igPC/OOID_CorrectionFactors.m` | `Langevin_Equation/` |
| Fig. 5(a–b) | `igPC/OOID_CorrectionFactors.m` | `Langevin_Equation/` |
| Fig. 7 | `Plotting/Plot_TimeSeriesHisto.m` | `Langevin_Equation/` |
| Fig. 8(a) | `igPC/OOID_igPC.m` | `Langevin_Equation/` |
| Fig. 8(b) | `igPC/OOID_igPC.m` | `VanderPol_Equation/` |
| Fig. 8(c–d) | `igPC/OOID_igPC.m` | `Experiments/` |
| Fig. 9 | `Plotting/Plot_TimeSeriesHisto.m` | `Langevin_Equation/` |
| Fig. 10(a) | `Plotting/Plot_Violin.m` | `Langevin_Equation/` |
| Fig. 10(b) | `Plotting/Plot_Violin.m` | `VanderPol_Equation/` |
| Fig. 11 | `Plotting/Plot_AmplitudePDF.m` | `Experiments/` |
| Fig. 12 | `Plotting/Plot_BarResults.m` | `Experiments/` |
| Supp. Fig. 1–2 | `igPC/OOID_CorrectionFactors_Validation.m` | `Langevin_Equation/` |

---

## Typical workflow

### Examples (tutorial)
1. Navigate to `Examples/`.
2. Open `Examples.m`, set `Example` to one of `'base'`, `'sum'`, `'multiplication'`, `'division'`, or `'axb'`.
3. Run. Figures are saved to `Examples/Figs/`.

### Langevin_Equation
1. Run `igPC/OOID_igPC.m` (or the KM variants) for all growth rates in `nu_vec`.
2. Run `igPC/OOID_CorrectionFactors.m` to compute and plot correction maps (Figs. 3–5).
3. Use `Plotting/Plot_Violin.m` and `Plotting/Plot_TimeSeriesHisto.m` for Figs. 7, 9–10.

### VanderPol_Equation
1. Run `vderPol_Solver.m` to simulate and save amplitude envelopes to `DATA/`.
2. Run `igPC/OOID_igPC.m` (or `KMCoeff/OOID_KM_Direct.m`) for the same `nu_vec`.
3. Run `Plotting/Plot_Violin.m` for Fig. 10(b).

### Experiments
1. Run `readExpData.m` to band-pass filter the raw recordings and save amplitude envelopes to `DATA/`.
2. Run `igPC/OOID_igPC.m` to identify parameters via igPC (Figs. 8c–d).
3. Run `KMCoeff/OOID_KM_Direct.m` (or `OOID_KM_PDFMatrix.m`) to identify via KM.
4. Run `Plotting/Plot_AmplitudePDF.m` for Fig. 11 and `Plotting/Plot_BarResults.m` for Fig. 12.

---

## Key conventions

**Amplitude SDE (Itô, slow-flow)**

    dA/dt = ν·A − b·A³ + c/A + ζ(t)

where `b = κ/8` and `c = Γ/(4·ω₀²)`.  The identified parameter vector is always `[ν, b, c]`.

**Factor of 2** — The second transition moment satisfies `M₂(τ,A)/τ → 2·D₂(A) = 2c` as `τ→0`.  The PDF-weighted average of the KM second coefficient estimates `2c`; dividing by 2 recovers the physical diffusion coefficient `c`.

**Gamma formula** — `Γ = c · 4·ω₀²`.

**MSD fit** — `MSD(τ) = MSD_amp · (1 − exp(−τ/τ_drift))`, then `c = MSD_amp / (2·τ_drift)`.

---

## External data

Experimental pressure recordings: Noiray, N. & Denisov, A. (2017). *A method for evaluating the acoustic limit cycle in turbulent combustors and its application to gas turbines.* Proc. Combust. Inst. 36, 3843–3850.

---

## Dependencies

- MATLAB R2019b or later (uses `discretize`, `histcounts`, `fminsearch`)
- Signal Processing Toolbox (Hilbert transform, band-pass filter)
- Optimization Toolbox (`fminsearch` falls back to built-in if absent)
