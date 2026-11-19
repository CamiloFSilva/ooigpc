# Examples — PC Algebraic Operations Tutorial

This folder contains the MATLAB code that accompanies **Section 1 of the Supplementary
Material** to the paper:

> Silva, C. F.  (2026).  
> *Output-only identification of thermoacoustic systems using polynomial chaos expansions.*  
> Combustion and Flame.

The script `Examples.m` illustrates the five fundamental PC algebraic operations needed
to understand the igPC identification method.  Run it by setting `Example` to one of the
five strings listed below.

---

## Files

| File | Purpose |
|------|---------|
| `Examples.m` | Main tutorial script (see below) |
| `Tensor_1D_comput.m` | Pre-computes C and D tensors for the 1-D Hermite basis; saves `Ctensor2D.mat`, `Dtensor2D.mat`, `norm2D.mat` |
| `Tensor_2D_comput.m` | 2-D Kronecker-product version of the tensor computation |
| `DisplayCtensor.m` | Prints selected entries of the C (triple-product) tensor |
| `DisplayDtensor.m` | Prints selected entries of the D (quadruple-product) tensor |
| `Plot_Results_Histo.m` | Plots the marginal PDF of the amplitude envelope from igPC output data |
| `ForKurtosis.m` | Symbolic computation of variance, skewness and kurtosis for a PC expansion |

---

## PC Basis (Hermite Polynomials)

The script uses the **probabilist's Hermite polynomials** up to second order as basis functions:

```
ψ₀(ξ) = 1
ψ₁(ξ) = ξ
ψ₂(ξ) = ξ² − 1
```

where ξ ~ N(0,1) is a standard Gaussian random variable.

A scalar random variable x is then represented as:

```
x = x₀ψ₀ + x₁ψ₁ + x₂ψ₂
```

The variance follows from orthogonality:  `Var[x] = x₁²‹ψ₁²› + x₂²‹ψ₂²› = x₁² + 2x₂²`

The norms are  `‹ψ₀²› = 1`,  `‹ψ₁²› = 1`,  `‹ψ₂²› = 2`.

---

## Examples — set `Example = '<name>'`

### `'base'` — Single 1-D PC random variable  
*(Supplementary Material, Section 1.1)*

Defines  x = 5ψ₀ + 0.6ψ₁ + 0.06ψ₂  and plots:
- the marginal PDFs of each individual mode (ψ₀-only, ψ₁-only, ψ₂-only)
- the total PDF of x
- the histograms of the basis functions ψ₀, ψ₁, ψ₂ themselves

**Output figures** (`Figs/`):  
`Marg_pdfo_base.pdf`, `Marg_pdf1_base.pdf`, `Marg_pdf2_base.pdf`, `Marg_pdf_base.pdf`,  
`psi0_base.pdf`, `psi1_base.pdf`, `psi2_base.pdf`

---

### `'sum'` — Addition of two 2-D PC random variables  
*(Supplementary Material, Section 1.2)*

Defines x (driven by ξ₁) and y (driven by ξ₂) and computes z = x + y analytically in PC
space by simple coefficient addition.  Monte Carlo validation is included.

**Output figures**: `Marg_x_sum.pdf`, `Marg_y_sum.pdf`, `Marg_z_sum.pdf`

---

### `'multiplication'` — Multiplication of two 2-D PC random variables  
*(Supplementary Material, Section 1.3)*

Computes z = x · y by expanding the product in the 2-D Hermite basis.  All 9 PC
coefficients are obtained via direct algebra (no linear system required).

**Output figures**: `Marg_x_multiplication.pdf`, `Marg_y_multiplication.pdf`, `Marg_z_multiplication.pdf`

---

### `'division'` — Inversion of a 1-D PC random variable  
*(Supplementary Material, Section 1.4 — Galerkin projection)*

Solves  a·x = 1  (i.e., x = 1/a) where a is a PC random variable.  The Galerkin
projection transforms the equation into a 3×3 linear system using the C tensor:
`Σⱼ Σₖ aₖ Cᵢⱼₖ xⱼ = δᵢ₀`

**Output figure**: `Marg_a_division.pdf`

---

### `'axb'` — General linear PC equation ax = b  
*(Supplementary Material, Section 1.5)*

Solves  a·x = b  where both a and b are PC random variables from independent Gaussians
(ξ₁ and ξ₂ respectively).  The result x = b/a is obtained via a 4×4 Galerkin system.

**Output figure**: `Marg_xmt_axb.pdf`

---

## Running the script

1. Open MATLAB and navigate to this folder.
2. Open `Examples.m`.
3. Set the `Example` variable (line 7) to one of the five strings above.
4. Run the script.  Figures are saved to `./Figs/`.

The precomputed tensor files (`Ctensor2D.mat`, `Dtensor2D.mat`, `norm2D.mat`) are
provided so that `Examples.m` can run without first executing `Tensor_1D_comput.m`,
which requires the MATLAB Symbolic Math Toolbox.

---

## Link to igPC identification

The same PC algebraic operations used here underlie the igPC identification method
(Eqs. 10–11 of the paper): given the moments (μ, σ, c) of the amplitude envelope,
the nonlinear damping parameters ν and κ are recovered by solving a small PC-algebraic
system.  The `Tensor_1D_comput.m` script computes the tensors that appear in those
equations.
