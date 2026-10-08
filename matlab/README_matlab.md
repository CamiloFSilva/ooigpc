# MATLAB archive — `ooigpc`

Official MATLAB implementation accompanying the paper:

> Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic systems
> using polynomial chaos expansions.* Combustion and Flame.

This folder is the archival reference for the results reported in the paper.
The Python package (`src/ooigpc/`) is a simplified port of these scripts.

---

## Files

| File | Description | Paper reference |
|------|-------------|-----------------|
| `simulate_langevin.m` | Euler–Maruyama integrator for the amplitude SDE | Eq. 3–4 |
| `simulate_van_der_pol.m` | Stochastic Van der Pol ODE integrator | Eq. 2 |
| `compute_msd.m` | Mean-square displacement MSD(τ) | Eq. 5 |
| `fit_msd.m` | Exponential MSD fit → diffusion coefficient *c* | Eq. 7, S-11 |
| `apply_correction.m` | Two-level non-Gaussian correction → σ | Suppl. §5, Eqs. S-7–S-9 |
| `identify_igpc.m` | igPC identification formulae → ν, κ, Γ | Eqs. 10–12 |
| `main_demo.m` | End-to-end demonstration reproducing §5.1 | — |

---

## Quick start

```matlab
% Add this folder to the MATLAB path (run once per session)
addpath(fileparts(mfilename('fullpath')));

% Run the full demo
main_demo
```

Or step by step:

```matlab
omega = 2*pi*120;          % 120 Hz mode

% 1. Simulate
A = simulate_langevin(16, 3.4, 1e7, omega, 2e6, 5e-5, 'Seed', 0);

% 2. MSD and diffusion coefficient
tau = (2.5e-4 : 2.5e-4 : 0.2)';
[taus, MSD]            = compute_msd(A, tau, 5e-5);
[A_hat, T_hat, ~]      = fit_msd(taus, MSD);
c_est                  = A_hat / (2*T_hat);

% 3. Identify
res = identify_igpc(A(5001:end), c_est, 'Omega', omega);
fprintf('nu = %.1f s^{-1},  kappa = %.2f,  Gamma = %.2e\n', ...
    res.nu, res.kappa, res.Gamma);
```

---

## Required toolboxes

- **Optimization Toolbox** — `lsqcurvefit` (used in `fit_msd.m`)
- **Statistics and Machine Learning Toolbox** — `skewness` (used in `apply_correction.m` Region 2)

---

## Using your own data

The identification requires the **amplitude envelope** A(t) of the pressure
signal, not the raw pressure.  In MATLAB:

```matlab
% Band-pass around the thermoacoustic mode
p_bp = bandpass(p, [f0 - 20, f0 + 20], fs);

% Hilbert-transform envelope
A = abs(hilbert(p_bp));

% Then call identify_igpc as shown above
```
