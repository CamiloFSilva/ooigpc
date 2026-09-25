# ooigpc

**Output-only identification of thermoacoustic systems using polynomial chaos expansions**

Python implementation accompanying the paper:

> Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic systems using polynomial chaos expansions.* Combustion and Flame.

---

## Overview

`ooigpc` recovers the three parameters of a self-excited thermoacoustic
oscillator from measured pressure data **without requiring any external
forcing** (output-only identification).  The oscillator model is the
Langevin SDE for the amplitude envelope A(t):

$$
\mathrm{d}A = \bigl(\nu A - \tfrac{\kappa}{8} A^3 + \tfrac{c}{A}\bigr)\,\mathrm{d}t
              + \sqrt{2c}\,\mathrm{d}W
$$

The three identified parameters are:

| Symbol | Meaning | Unit |
|--------|---------|------|
| ν | Linear growth rate | s⁻¹ |
| κ | Nonlinear damping coefficient | Pa⁻²·s⁻¹ |
| Γ | Noise intensity | Pa²·s⁻³ |

The method uses iterative generalized Polynomial Chaos (igPC) expansions to
derive explicit algebraic formulae (Eqs. 10–11 of the paper) that express ν
and κ directly in terms of the mean μ, corrected standard deviation σ, and
diffusion coefficient c of the amplitude PDF.

---

## Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate ooigpc

# Install the package in editable mode
pip install -e .
```

Or with pip only:

```bash
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from ooigpc.solvers.langevin import simulate_langevin
from ooigpc.igpc import identify_from_signal

# Physical parameters (paper §5.1)
omega = 2 * np.pi * 120     # 120 Hz thermoacoustic mode
nu_true    = 16.0           # linear growth rate (s⁻¹)
kappa_true = 3.4            # nonlinear coefficient
Gamma_true = 1e7            # noise intensity (Pa²·s⁻³)

# Simulate the Langevin SDE (2 × 10⁶ steps)
x = simulate_langevin(nu=nu_true, kappa=kappa_true, Gamma=Gamma_true,
                      omega=omega, N=2_000_000, dt=5e-5, seed=0)

# Time-delay vector for the MSD (Eq. 5)
tau = np.arange(2.5e-4, 0.201, 2.5e-4)

# Identify parameters (Eqs. 7 and 10–11)
result = identify_from_signal(x, tau, dt_sim=5e-5, omega=omega)
ident  = result['identification']

print(f"Identified  ν = {ident.nu:.1f} s⁻¹   (true: {nu_true})")
print(f"Identified  κ = {ident.kappa:.2f}      (true: {kappa_true})")
print(f"Identified  Γ = {ident.Gamma:.2e}  (true: {Gamma_true:.2e})")
```

---

## Package structure

```
src/ooigpc/
├── __init__.py
├── pc_basis.py         # Hermite PC basis, norms, 3rd/4th-order tensors
├── diffusion.py        # MSD computation and exponential fit → c  (Eq. 7)
├── correction.py       # Two-level non-Gaussian correction (Suppl. §5)
├── igpc.py             # Main identification: Eqs. 10–11
├── solvers/
│   ├── langevin.py     # Discrete Langevin SDE integrator  (Eq. 3)
│   └── van_der_pol.py  # Stochastic Van der Pol ODE  (Eq. 2)
├── signal/
│   ├── bandpass.py     # Butterworth / rectangular band-pass filter
│   └── envelope.py     # Hilbert-transform amplitude envelope
└── reference/
    └── km_coefficients.py  # Kramers–Moyal reference method
```

---

## Jupyter tutorial

The notebook `notebooks/tutorial.ipynb` reproduces the results of the
supplementary material step by step, with equations cross-referenced to the
paper.

```bash
jupyter notebook notebooks/tutorial.ipynb
```

---

## Citation

```bibtex
@article{Silva2026ooigpc,
  author  = {Silva, Camilo F. and others},
  title   = {Output-only identification of thermoacoustic systems using
             polynomial chaos expansions},
  journal = {Combustion and Flame},
  year    = {2026},
}
```

---

## License

MIT — see `LICENSE`.