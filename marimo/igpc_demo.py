import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.stats import skew
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    return FancyArrowPatch, curve_fit, mpatches, np, plt, skew


# ---------------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    mo.md(r"""
    # igPC — Output-only identification of thermoacoustic systems

    **Reference:** Silva, C.F. et al. (2026). *Output-only identification of
    thermoacoustic systems using polynomial chaos expansions.* Combustion and Flame.

    ---

    A self-excited thermoacoustic oscillator can be modelled by the van der Pol equation (Eq. 2).
    Applying a slow-time averaging, the amplitude envelope $A(t)$ satisfies the Langevin equation
    (Eq. 3):

    $$\frac{dA}{dt} = a\,A \;-\; b\,A^3 \;+\; \frac{c}{A} \;+\; \zeta(t) \tag{3}$$

    where $a = \nu$, $b = \kappa/8$, $c = \Gamma/(4\omega_0^2)$, and $\zeta(t)$ is Gaussian
    white noise with intensity $\sqrt{2c}$.

    The three physical parameters to identify are:

    | Symbol | Physical meaning | Unit |
    |--------|-----------------|------|
    | ν | Linear growth rate (positive = linearly unstable) | s⁻¹ |
    | κ | Nonlinear saturation coefficient | — |
    | Γ | Noise intensity | Pa²·s⁻³ |

    **The method uses only the measured signal A(t). No external forcing is needed.**

    Adjust the sliders below. The identification results update automatically.
    """)
    return


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    nu_sl    = mo.ui.slider(2.0,   40.0,  value=16.0,           step=0.5,   label="ν  (s⁻¹)")
    kappa_sl = mo.ui.slider(0.5,    8.0,  value=3.4,            step=0.1,   label="κ")
    Gamma_sl = mo.ui.slider(2e6,   3e7,   value=1e7,            step=1e6,   label="Γ  (Pa²·s⁻³)")
    omega_sl = mo.ui.slider(100.0, 400.0, value=120.0,          step=10.0,  label="f₀  (Hz)")
    N_sl     = mo.ui.slider(100_000, 600_000, value=300_000,    step=50_000,label="N  (time steps)")
    mo.vstack([nu_sl, kappa_sl, Gamma_sl, omega_sl, N_sl])
    return Gamma_sl, N_sl, kappa_sl, nu_sl, omega_sl


@app.cell
def __(Gamma_sl, N_sl, kappa_sl, mo, np, nu_sl, omega_sl):
    nu    = nu_sl.value
    kappa = kappa_sl.value
    Gamma = Gamma_sl.value
    f0    = omega_sl.value
    N     = N_sl.value
    dt    = 5e-5               # time step (s)

    omega  = 2.0 * np.pi * f0  # angular frequency ω₀ (rad/s)
    a      = nu                 # a = ν
    b      = kappa / 8.0       # b = κ/8
    c_true = Gamma / (4.0 * omega**2)   # c = Γ/(4ω₀²)
    T_signal = N * dt           # physical duration (s)
    n_cycles = T_signal * f0    # number of oscillation cycles

    mo.md(f"""
    **Derived quantities:**

    | | |
    |---|---|
    | ω₀ = 2π·f₀ | {omega:.1f}  rad/s |
    | c = Γ / (4ω₀²) | {c_true:.5f} |
    | Signal duration | {T_signal:.1f} s  ({n_cycles:.0f} oscillation cycles) |
    """)
    return (
        N, T_signal, a, b, c_true, dt, f0, kappa, n_cycles, nu, omega,
    )


# ---------------------------------------------------------------------------
# Simulation — Euler–Maruyama integration
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    reseed_btn = mo.ui.button(
        label   = "🔀  New realisation  (different random seed)",
        value   = 0,
        on_click= lambda v: v + 1,
    )
    mo.vstack([
        mo.callout(
            mo.md(
                "**Stochastic signal.** "
                "The amplitude envelope is a random process: every run is a *different realisation* "
                "of the same noise-driven oscillator. "
                "Identified parameters therefore vary slightly from run to run — this is the "
                "finite-sample uncertainty of the method, not a flaw. "
                "Click the button to generate a new realisation with the same physical parameters "
                "and observe the spread in results."
            ),
            kind="info",
        ),
        reseed_btn,
    ])
    return (reseed_btn,)


@app.cell
def __(mo):
    mo.md(r"""
    ---
    ## Simulation — Euler–Maruyama discretisation of the Langevin equation

    The continuous Langevin equation (Eq. 3) is discretised as (Eq. 4):

    $$A_{n+1} = A_n \;+\; \underbrace{a\,A_n\,\Delta t}_{\text{linear growth}}
               \;-\; \underbrace{b\,A_n^3\,\Delta t}_{\text{nonlinear saturation}}
               \;+\; \underbrace{\dfrac{c}{A_n}\,\Delta t}_{\text{diffusion correction}}
               \;+\; \underbrace{\sqrt{2c\,\Delta t}\;\xi_n}_{\text{noise}} \tag{4}$$

    with $\xi_n \sim \mathcal{N}(0,1)$ independent at each step.
    """)
    return


@app.cell
def __(N, a, b, c_true, dt, np, reseed_btn):
    # Euler–Maruyama integration (Eq. 4)
    # seed = button click count  →  each click gives a fresh realisation
    rng       = np.random.default_rng(reseed_btn.value)
    A         = np.empty(N + 1)
    A[0]      = float(np.sqrt(a / b)) if b > 0 else 0.5   # near deterministic fixed point
    sqrt2c_dt = np.sqrt(2.0 * c_true * dt)

    for _n in range(N):
        _An = max(A[_n], 1e-8)
        A[_n + 1] = max(
            _An
            + a * _An * dt          # linear growth
            - b * _An**3 * dt       # nonlinear saturation
            + c_true / _An * dt     # diffusion correction
            + sqrt2c_dt * rng.standard_normal(),   # noise
            1e-8,
        )

    time = np.arange(N + 1) * dt
    return A, rng, sqrt2c_dt, time


@app.cell
def __(A, N, dt, np, plt, time):
    fig1, ax1 = plt.subplots(1, 2, figsize=(11, 3.5))

    n_show = min(10_000, N)
    ax1[0].plot(time[:n_show], A[:n_show], lw=0.6, color="steelblue")
    ax1[0].set(xlabel="Time (s)", ylabel="A(t)", title="Amplitude envelope (excerpt)")

    A_pos = A[5000:][A[5000:] > 0]
    ax1[1].hist(A_pos, bins=100, density=True, color="steelblue", alpha=0.7, edgecolor="none")
    ax1[1].set(xlabel="A", ylabel="PDF", title="Amplitude PDF")

    plt.tight_layout()
    fig1
    return A_pos, ax1, fig1, n_show


# ---------------------------------------------------------------------------
# Step 1 — MSD and diffusion coefficient c
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    mo.md(r"""
    ---
    ## Step 1 — Diffusion coefficient *c* from the mean-square displacement

    The multi-lag mean-square displacement is (see Sec. 3.1):

    $$\text{MSD}(\tau) = \bigl\langle [A(t+\tau) - A(t)]^2 \bigr\rangle$$

    Since the autocovariance decays approximately as $C(\tau) \approx C(0)\,e^{-\lambda\tau}$,
    the MSD is modelled by (Eq. 40):

    $$\text{MSD}_\text{mod}(\tau) = 2\,C(0)\,\bigl[1 - e^{-\lambda\tau}\bigr] \tag{40}$$

    Fitting this model to the data gives $C(0)$ and $\lambda$.
    The diffusion coefficient then follows from Eq. (14):

    $$c = \lim_{\tau\to 0}\frac{\text{MSD}(\tau)}{2\tau} = C(0)\cdot\lambda \tag{14}$$
    """)
    return


@app.cell
def __(A, N, dt, np):
    # Compute MSD(τ) for a range of time delays
    i_start    = 5000
    tau_values = np.arange(2.5e-4, 0.201, 2.5e-4)
    tk_vec     = np.maximum(1, np.round(tau_values / dt).astype(int))
    taus       = tk_vec * dt

    MSD = np.zeros(len(taus))
    for _idx, _tk in enumerate(tk_vec):
        _ie = N - _tk
        if _ie <= i_start:
            continue
        _x1   = A[i_start : _ie + 1]
        _x2   = A[i_start + _tk : _ie + _tk + 1]
        _mask = (_x1 > 0) & (_x2 > 0)
        if _mask.sum() == 0:
            continue
        _d        = _x1[_mask] - _x2[_mask]
        MSD[_idx] = float(np.dot(_d, _d) / len(_d))

    return MSD, i_start, taus, tau_values, tk_vec


@app.cell
def __(MSD, curve_fit, np, taus):
    # Fit MSD_mod(τ) = 2·C0·(1 − exp(−λ·τ))   [Eq. 40]
    # Free parameters: C0 (autocovariance at τ=0) and λ (relaxation rate)
    #
    # scipy.optimize.curve_fit uses nonlinear least squares (Levenberg–Marquardt).
    # p0 = initial guess for [C0, λ].  bounds ensure both parameters stay positive.

    def _msd_model(tau, C0, lam):
        return 2.0 * C0 * (1.0 - np.exp(-lam * tau))

    _popt, _pcov = curve_fit(
        _msd_model,
        taus,
        MSD,
        p0    = [0.05, 10.0],          # initial guess: [C0, λ]
        bounds= ((0, 0), (np.inf, np.inf)),
        maxfev= 10_000,
    )
    C0  = float(_popt[0])   # autocovariance C(0)
    lam = float(_popt[1])   # relaxation rate λ  (s⁻¹)

    MSD_fit = _msd_model(taus, C0, lam)

    # Diffusion coefficient  c = C(0) · λ   [Eq. 14]
    c = C0 * lam
    return C0, MSD_fit, c, lam


@app.cell
def __(C0, MSD, MSD_fit, c, c_true, lam, mo, plt, taus):
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.5))

    ax2[0].plot(taus, MSD,     "rx", ms=3, label="MSD data")
    ax2[0].plot(taus, MSD_fit, "b-", lw=2, label="2·C(0)·[1−exp(−λτ)]  [Eq. 40]")
    ax2[0].set(xlabel="τ (s)", ylabel="MSD(τ)",
               title=f"MSD fit:  C(0) = {C0:.5f},  λ = {lam:.2f} s⁻¹")
    ax2[0].legend(fontsize=9)

    ax2[1].plot(taus, MSD     / (2 * taus), "rx", ms=3, label="MSD / (2τ)  from data")
    ax2[1].plot(taus, MSD_fit / (2 * taus), "b-", lw=2, label="model")
    ax2[1].axhline(c,      color="k", ls="--", lw=1.5, label=f"c identified = {c:.5f}")
    ax2[1].axhline(c_true, color="r", ls=":",  lw=1.5, label=f"c true       = {c_true:.5f}")
    ax2[1].set(xlabel="τ (s)", ylabel="MSD(τ) / (2τ)",
               title="Extrapolation to τ → 0  gives  c = C(0)·λ")
    ax2[1].legend(fontsize=9)

    plt.tight_layout()
    mo.vstack([
        mo.md(f"**c identified = {c:.5f}**   (true: {c_true:.5f},  error: {abs(c-c_true)/c_true*100:.1f}%)"),
        fig2,
    ])
    return ax2, fig2


# ---------------------------------------------------------------------------
# Step 2 — Correction for non-Gaussian PDF  (Fig. 6 of the paper)
# ---------------------------------------------------------------------------

@app.cell
def __(plt):
    # Reproduce the workflow of Fig. 6 from the paper as a flow diagram

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis("off")

    def box(ax, x, y, w, h, text, color="#dce8f7", fontsize=9):
        ax.add_patch(plt.Rectangle((x - w/2, y - h/2), w, h,
                                   facecolor=color, edgecolor="steelblue", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                wrap=True, multialignment="center")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    # Top boxes
    box(ax6, 5.0, 9.3, 6.0, 0.8, "Extract envelope from acoustic signal A(t)", fontsize=9)
    arrow(ax6, 5.0, 8.9, 5.0, 8.3)
    box(ax6, 5.0, 7.9, 5.5, 0.7, "Compute  μ = mean(A),  σ = std(A),  γ = skewness·σ³", fontsize=9)
    arrow(ax6, 5.0, 7.55, 5.0, 7.0)
    box(ax6, 5.0, 6.6, 4.0, 0.7, "Determine regime from μ and σ", fontsize=9)

    # Branch arrows (from regime box at y=6.25)
    arrow(ax6, 3.1, 6.25, 1.8, 5.55)   # to left
    arrow(ax6, 5.0, 6.25, 5.0, 5.55)   # to middle
    arrow(ax6, 6.9, 6.25, 8.4, 5.55)   # to right

    # Left branch: Unstable / Weakly unstable
    # x=1.8, w=2.9  →  spans 0.35 – 3.25  (no overlap with middle starting at 3.55)
    box(ax6, 1.8, 5.2, 2.9, 0.6, "Unstable / Weakly unstable\nσ ≤ 0.280 μ",
        color="#fde8d8", fontsize=8)
    box(ax6, 1.8, 4.2, 2.9, 1.3,
        "Solve cubic (Eq. 20):\n4x₂³ − 6σ²x₂ + γ = 0\n"
        "δ_σ = 2.6 · x₂  (Eq. 25)\n"
        "σ_proj = √(σ² − 2δ_σ²)  (Eq. 22)",
        color="#fff3ec", fontsize=8)
    arrow(ax6, 1.8, 4.9, 1.8, 4.85)
    arrow(ax6, 1.8, 3.55, 1.8, 2.95)

    # Middle branch: Marginally stable
    # x=5.2, w=3.0  →  spans 3.7 – 6.7
    box(ax6, 5.2, 5.2, 3.0, 0.6, "Marginally stable\n0.280 μ ≤ σ ≤ 0.423 μ",
        color="#e8f5e9", fontsize=8)
    box(ax6, 5.2, 4.0, 3.2, 1.5,
        "Interpolation coord. β  (Eq. 32)\n"
        "δ_corr,(0) = 0.144σ + 0.0126  (Eq. 30)\n"
        "δ_corr,(1) = −0.035σ           (Eq. 31)\n"
        "δ_corr = (1−β)·δ_corr,(1) + β·δ_corr,(0)  (Eq. 33)\n"
        "σ_proj = σ + δ_corr  (Eq. 29)",
        color="#f1f8f1", fontsize=8)
    arrow(ax6, 5.2, 4.9, 5.2, 4.75)
    arrow(ax6, 5.2, 3.25, 5.2, 2.95)

    # Right branch: Stable
    box(ax6, 8.4, 5.2, 2.5, 0.6, "Stable\nσ > 0.423 μ",
        color="#f5f5f5", fontsize=8)
    box(ax6, 8.4, 4.2, 2.5, 0.6, "Outside scope\nof this study",
        color="#f5f5f5", fontsize=8)

    # Bottom: final step
    arrow(ax6, 1.8, 2.95, 3.5, 2.45)
    arrow(ax6, 5.2, 2.95, 5.2, 2.45)
    box(ax6, 4.5, 2.1, 5.5, 0.7,
        "Use  μ  and  σ_proj  in Eqs. (10) and (11)  →  ν, κ",
        color="#dce8f7", fontsize=9)

    ax6.set_title("Fig. 6 — Workflow for the correction of the non-Gaussian envelope PDF",
                  fontsize=10, pad=4)
    plt.tight_layout()
    fig6
    return ax6, box, arrow, fig6


@app.cell
def __(A_pos, np, skew):
    # Two-level correction — implements the workflow of Fig. 6
    mu  = float(np.mean(A_pos))
    std = float(np.std(A_pos, ddof=1))
    gam = float(skew(A_pos, bias=False)) * std**3   # third central moment γ

    sigma_0 = 0.423 * mu   # upper boundary σ_(0)
    sigma_1 = 0.280 * mu   # lower boundary σ_(1)

    if std <= sigma_1:
        # ── Region: Unstable / Weakly unstable ───────────────────────────────
        # Solve depressed cubic:  4x₂³ − 6σ²·x₂ + γ = 0   (Eq. 20)
        roots      = np.roots([4.0, 0.0, -6.0 * std**2, gam])
        x2         = float(np.real(roots[2]))          # admissible root (index 2)
        delta_sig  = 2.6 * x2                          # Eq. 25
        var_proj   = max(std**2 - 2.0 * delta_sig**2, 1e-12)
        sigma_proj = float(np.sqrt(var_proj))          # Eq. 22
        region     = "Unstable / Weakly unstable  (σ ≤ 0.280 μ)"

    elif std <= sigma_0:
        # ── Region: Marginally stable ─────────────────────────────────────────
        beta          = (std - sigma_1) / (sigma_0 - sigma_1)   # Eq. 32
        delta_corr_0  =  0.144 * std + 0.0126                   # Eq. 30
        delta_corr_1  = -0.035 * std                            # Eq. 31
        delta_corr    = (1.0 - beta) * delta_corr_1 + beta * delta_corr_0  # Eq. 33
        sigma_proj    = std + delta_corr                         # Eq. 29
        region        = "Marginally stable  (0.280 μ ≤ σ ≤ 0.423 μ)"

    else:
        # ── Region: Stable — outside scope ───────────────────────────────────
        sigma_proj = std
        region     = "Stable  (σ > 0.423 μ) — outside scope of this method"

    return beta, delta_corr, delta_corr_0, delta_corr_1, gam, mu, region, sigma_proj, std


@app.cell
def __(mo, mu, region, sigma_proj, std):
    mo.md(f"""
    **Correction result:**

    | | |
    |---|---|
    | Regime | {region} |
    | μ | {mu:.4f} |
    | σ_raw (before correction) | {std:.4f} |
    | σ_proj (corrected) | {sigma_proj:.4f} |
    """)
    return


# ---------------------------------------------------------------------------
# Step 3 — Identification: Eqs. (10) and (11)
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    mo.md(r"""
    ---
    ## Step 3 — Parameter identification  (Eqs. 10 and 11)

    Given the corrected $\sigma_\text{proj}$, the mean $\mu$, and the diffusion coefficient $c$,
    the linear growth rate and nonlinear coefficient follow directly from:

    $$\nu = \frac{9\sigma^4 + 2\mu^2\sigma^2 - \mu^4}{2\,\mu^2\sigma^2(\sigma^2-\mu^2)}\cdot c \tag{10}$$

    $$\kappa = \frac{8\,(3\sigma^2 - \mu^2)}{2\,\mu^2\sigma^2(\sigma^2-\mu^2)}\cdot c \tag{11}$$

    The noise intensity is recovered from $c = \Gamma/(4\omega_0^2)$:

    $$\Gamma = 4\,\omega_0^2\cdot c$$
    """)
    return


@app.cell
def __(c, c_true, f0, kappa, mo, mu, n_cycles, nu, omega, region, sigma_proj):
    mu2    = mu**2
    sig2   = sigma_proj**2
    denom  = mu2 * sig2 * (sig2 - mu2)

    # Eq. (10) — linear growth rate ν
    nu_id = c * 0.5 * (9*sig2**2 + 2*mu2*sig2 - mu2**2) / denom

    # Eq. (11) — nonlinear saturation coefficient κ
    kappa_id = 8.0 * c * 0.5 * (3*sig2 - mu2) / denom

    # Noise intensity  Γ = 4ω₀²·c
    Gamma_id   = c      * 4.0 * omega**2
    Gamma_true = c_true * 4.0 * omega**2

    err_nu    = abs(nu_id    - nu)          / abs(nu)          * 100
    err_kappa = abs(kappa_id - kappa)       / abs(kappa)       * 100
    err_Gamma = abs(Gamma_id - Gamma_true)  / abs(Gamma_true)  * 100
    err_c     = abs(c        - c_true)      / abs(c_true)      * 100

    mo.md(f"""
    ### Identification results

    Regime detected: **{region}**

    | Parameter | True | Identified | Error |
    |-----------|------|-----------|-------|
    | ν  (s⁻¹) | {nu:.2f} | {nu_id:.2f} | {err_nu:.1f} % |
    | κ | {kappa:.3f} | {kappa_id:.3f} | {err_kappa:.1f} % |
    | Γ  (Pa²·s⁻³) | {Gamma_true:.3e} | {Gamma_id:.3e} | {err_Gamma:.1f} % |
    | c | {c_true:.5f} | {c:.5f} | {err_c:.1f} % |

    ---
    **Signal used:** {n_cycles:.0f} oscillation cycles at f₀ = {f0:.0f} Hz.

    The paper reports results for 100 s signals (≈ 12,000 cycles at 120 Hz).
    Errors above 5–10 % are expected for shorter signals, particularly for ν in the
    marginally stable regime. Increase N to reduce statistical scatter.
    """)
    return (
        Gamma_id, Gamma_true, denom, err_Gamma, err_c, err_kappa, err_nu,
        kappa_id, mu2, nu_id, sig2,
    )


# ---------------------------------------------------------------------------
# Upload your own signal
# ---------------------------------------------------------------------------

@app.cell
def __(mo):
    mo.md(r"""
    ---
    ## Apply igPC to your own signal

    Upload either a **raw pressure signal** η(t) or a pre-extracted
    **amplitude envelope** A(t).

    **Raw pressure signal** (e.g. exported from MATLAB with `writematrix([t, eta], 'signal.csv')`):
    the app will band-pass filter around f₀ and compute the Hilbert envelope automatically.

    **Amplitude envelope** (already computed, e.g. from `abs(hilbert(eta_bp))`):
    fed directly into the igPC pipeline.

    **File formats accepted:**

    | Format | Content |
    |--------|---------|
    | CSV / TXT — 1 column | η(t) or A(t), one sample per row |
    | CSV / TXT — 2 columns | [t, η(t)] or [t, A(t)] — time column is used to infer Δt |
    | NumPy .npy | 1-D array η or A |
    """)
    return


@app.cell
def __(mo):
    sig_type_rb = mo.ui.radio(
        options=["Raw pressure signal η(t)  →  band-pass + Hilbert",
                 "Amplitude envelope A(t)  →  direct identification"],
        value  ="Raw pressure signal η(t)  →  band-pass + Hilbert",
        label  ="Signal type",
    )
    user_file  = mo.ui.file(
        label    = "Signal file  (.csv, .txt or .npy)",
        filetypes= [".csv", ".txt", ".npy"],
    )
    dt_user_sl = mo.ui.slider(
        1e-5, 5e-3, value=5e-5, step=1e-5,
        label="Sampling interval Δt  (s)  [ignored when file has a time column]",
    )
    f0_user_sl = mo.ui.slider(
        20.0, 2000.0, value=120.0, step=10.0,
        label="Fundamental frequency f₀  (Hz)",
    )
    bw_user_sl = mo.ui.slider(
        5.0, 200.0, value=50.0, step=5.0,
        label="Band-pass half-bandwidth Δf  (Hz)  [only for raw pressure]",
    )
    mo.vstack([sig_type_rb, user_file, dt_user_sl, f0_user_sl, bw_user_sl])
    return bw_user_sl, dt_user_sl, f0_user_sl, sig_type_rb, user_file


@app.cell
def __(bw_user_sl, curve_fit, dt_user_sl, f0_user_sl,
       mo, np, plt, sig_type_rb, skew, user_file):
    import io as _io
    from scipy.signal import butter as _butter, filtfilt as _filtfilt
    from scipy.signal import hilbert as _hilbert

    # ── helper ───────────────────────────────────────────────────────────────
    def _msd_model_u(tau, C0, lam):
        return 2.0 * C0 * (1.0 - np.exp(-lam * tau))

    # ── guard: no file uploaded yet ──────────────────────────────────────────
    if not user_file.value:
        _out = mo.md("*No file uploaded yet.  Use the widget above to upload your signal.*")
    else:
        try:
            # ── 0. Parse ─────────────────────────────────────────────────────
            _fi  = user_file.value[0]
            if _fi.name.lower().endswith(".npy"):
                _raw = np.load(_io.BytesIO(_fi.contents)).ravel()
            else:
                import re as _re
                # utf-8-sig strips the BOM that MATLAB sometimes writes
                _text = _fi.contents.decode("utf-8-sig", errors="replace")
                _lines = [l.strip() for l in _text.splitlines() if l.strip()]

                # Try common delimiter/decimal combos in order.
                # MATLAB on German/EU locale → semicolon delimiter, comma decimal.
                # MATLAB on EN locale or newer → comma delimiter, dot decimal.
                _raw = None
                for _sep, _fix_dec in [
                    (";",  True),   # EU MATLAB: 0,123 separated by ;
                    (",",  False),  # standard CSV: 0.123 separated by ,
                    ("\t", True),   # tab + comma decimal
                    ("\t", False),  # tab + dot decimal
                    (None, False),  # whitespace-separated
                ]:
                    try:
                        if _fix_dec:
                            _body = "\n".join(
                                _re.sub(r"(?<=\d),(?=\d)", ".", l) for l in _lines
                            )
                        else:
                            _body = "\n".join(_lines)
                        _kw = {"delimiter": _sep} if _sep is not None else {}
                        _raw = np.loadtxt(_io.StringIO(_body), **_kw)
                        break
                    except Exception:
                        continue
                if _raw is None:
                    raise ValueError(
                        "Could not parse the file. Expected a CSV/TSV with one or "
                        "two numeric columns (time, signal)."
                    )

            _raw = np.atleast_2d(_raw.astype(float))
            if _raw.shape[0] < _raw.shape[1]:
                _raw = _raw.T   # ensure (N, ncols)

            _dt_u = dt_user_sl.value
            if _raw.shape[1] >= 2:
                # two-column file: [t, signal]
                _dt_u = float(_raw[1, 0] - _raw[0, 0])
                _sig  = _raw[:, 1]
            else:
                _sig  = _raw[:, 0]

            _f0_u  = f0_user_sl.value
            _bw_u  = bw_user_sl.value
            _om_u  = 2.0 * np.pi * _f0_u
            _fs_u  = 1.0 / _dt_u
            _is_raw = "Raw pressure" in sig_type_rb.value

            # ── 1. Preprocessing ─────────────────────────────────────────────
            if _is_raw:
                # Band-pass filter η(t) → Hilbert → A(t)
                _nyq   = _fs_u / 2.0
                _fl, _fh = _f0_u - _bw_u, _f0_u + _bw_u
                if _fl <= 0 or _fh >= _nyq:
                    raise ValueError(
                        f"Band-pass [{_fl:.1f}, {_fh:.1f}] Hz out of range "
                        f"for fs = {_fs_u:.0f} Hz.  Reduce Δf or adjust f₀."
                    )
                _bb, _ab = _butter(4, [_fl/_nyq, _fh/_nyq], btype="bandpass")
                _eta_bp  = _filtfilt(_bb, _ab, _sig)
                _A_raw   = np.abs(_hilbert(_eta_bp))
                _prep_label = f"band-pass [{_fl:.0f}–{_fh:.0f} Hz] + Hilbert"
            else:
                _A_raw = _sig.copy()
                if np.any(_A_raw <= 0):
                    raise ValueError(
                        f"The envelope contains {np.sum(_A_raw<=0):,} non-positive "
                        "values.  Please upload a positive amplitude envelope."
                    )
                _prep_label = "pre-extracted envelope (no filtering)"

            _Nstep = len(_A_raw) - 1

            # ── 2. MSD ───────────────────────────────────────────────────────
            _i0_u  = min(5000, _Nstep // 10)
            _tau_u = np.arange(2.5e-4,
                                min(0.25, _Nstep * _dt_u * 0.12),
                                2.5e-4)
            if len(_tau_u) < 5:
                _tau_u = np.linspace(_dt_u * 2, _Nstep * _dt_u * 0.05, 40)

            _tk_u   = np.maximum(1, np.round(_tau_u / _dt_u).astype(int))
            _taus_u = _tk_u * _dt_u
            _MSD_u  = np.zeros(len(_taus_u))

            for _idx_u, _tk in enumerate(_tk_u):
                _ie_u = _Nstep - _tk
                if _ie_u <= _i0_u:
                    continue
                _x1_u = _A_raw[_i0_u : _ie_u + 1]
                _x2_u = _A_raw[_i0_u + _tk : _ie_u + _tk + 1]
                _mk_u = (_x1_u > 0) & (_x2_u > 0)
                if _mk_u.sum() == 0:
                    continue
                _d_u           = _x1_u[_mk_u] - _x2_u[_mk_u]
                _MSD_u[_idx_u] = float(np.dot(_d_u, _d_u) / len(_d_u))

            _p0_u = [float(np.var(_A_raw[_i0_u:])) * 0.1, 10.0]
            _popt_u, _ = curve_fit(
                _msd_model_u, _taus_u, _MSD_u,
                p0=_p0_u, bounds=((0, 0), (np.inf, np.inf)), maxfev=10_000,
            )
            _C0_u      = float(_popt_u[0])
            _lam_u     = float(_popt_u[1])
            _cu        = _C0_u * _lam_u
            _MSD_fit_u = _msd_model_u(_taus_u, _C0_u, _lam_u)

            # ── 3. Correction (Fig. 6) ────────────────────────────────────────
            _Ap_u  = _A_raw[_i0_u:]
            _mu_u  = float(np.mean(_Ap_u))
            _std_u = float(np.std(_Ap_u, ddof=1))
            _gam_u = float(skew(_Ap_u, bias=False)) * _std_u**3
            _s0_u  = 0.423 * _mu_u
            _s1_u  = 0.280 * _mu_u

            if _std_u <= _s1_u:
                _roots_u = np.roots([4.0, 0.0, -6.0 * _std_u**2, _gam_u])
                _x2_u    = float(np.real(_roots_u[2]))
                _dsig_u  = 2.6 * _x2_u
                _sigp_u  = float(np.sqrt(max(_std_u**2 - 2.0 * _dsig_u**2, 1e-12)))
                _reg_u   = "Unstable / Weakly unstable  (σ ≤ 0.280 μ)"
            elif _std_u <= _s0_u:
                _beta_u  = (_std_u - _s1_u) / (_s0_u - _s1_u)
                _dc0_u   =  0.144 * _std_u + 0.0126
                _dc1_u   = -0.035 * _std_u
                _dc_u    = (1.0 - _beta_u) * _dc1_u + _beta_u * _dc0_u
                _sigp_u  = _std_u + _dc_u
                _reg_u   = "Marginally stable  (0.280 μ < σ ≤ 0.423 μ)"
            else:
                _sigp_u  = _std_u
                _reg_u   = "Stable  (σ > 0.423 μ) — outside scope of this method"

            # ── 4. Identification (Eqs. 10–11) ────────────────────────────────
            _mu2_u  = _mu_u**2
            _sig2_u = _sigp_u**2
            _den_u  = _mu2_u * _sig2_u * (_sig2_u - _mu2_u)
            _nu_u   = _cu * 0.5 * (9*_sig2_u**2 + 2*_mu2_u*_sig2_u - _mu2_u**2) / _den_u
            _kap_u  = 8.0 * _cu * 0.5 * (3*_sig2_u - _mu2_u) / _den_u
            _Gam_u  = _cu * 4.0 * _om_u**2
            _dur_u  = _Nstep * _dt_u
            _cyc_u  = _dur_u * _f0_u

            # ── Plots ─────────────────────────────────────────────────────────
            _nrows = 2 if _is_raw else 1
            _figU, _axU = plt.subplots(_nrows, 3, figsize=(13, 3.5 * _nrows),
                                        squeeze=False)

            if _is_raw:
                # Top row: raw η(t) excerpt and filtered signal
                _nsh = min(10_000, _Nstep + 1)
                _t_u = np.arange(_nsh) * _dt_u
                _axU[0][0].plot(_t_u, _sig[:_nsh], lw=0.5, color="steelblue",
                                label="η(t)")
                _axU[0][0].set(xlabel="Time (s)", ylabel="η(t)",
                               title=f"Raw pressure — {_fi.name}")
                _axU[0][1].plot(_t_u, _eta_bp[:_nsh], lw=0.5, color="tomato",
                                label="η_bp(t)")
                _axU[0][1].set(xlabel="Time (s)", ylabel="η_bp(t)",
                               title=f"Band-passed  ({_fl:.0f}–{_fh:.0f} Hz)")
                _axU[0][2].plot(_t_u, _A_raw[:_nsh], lw=0.7, color="darkorange")
                _axU[0][2].set(xlabel="Time (s)", ylabel="A(t)",
                               title="Hilbert amplitude envelope")
                _bot = _axU[1]
            else:
                _bot = _axU[0]
                _nsh = min(10_000, _Nstep + 1)
                _t_u = np.arange(_nsh) * _dt_u
                _bot[0].plot(_t_u, _A_raw[:_nsh], lw=0.6, color="steelblue")
                _bot[0].set(xlabel="Time (s)", ylabel="A(t)",
                            title=f"Envelope excerpt — {_fi.name}")

            _bot[1].plot(_taus_u, _MSD_u,     "rx", ms=3, label="MSD data")
            _bot[1].plot(_taus_u, _MSD_fit_u, "b-", lw=2,
                         label="2·C(0)·[1−exp(−λτ)]")
            _bot[1].set(xlabel="τ (s)", ylabel="MSD(τ)",
                        title=f"MSD fit:  C(0)={_C0_u:.4f},  λ={_lam_u:.2f} s⁻¹")
            _bot[1].legend(fontsize=8)

            _bot[2].hist(_Ap_u, bins=80, density=True,
                         color="steelblue", alpha=0.7, edgecolor="none")
            _bot[2].set(xlabel="A", ylabel="PDF", title="Amplitude PDF")

            plt.tight_layout()

            # ── Results table ─────────────────────────────────────────────────
            _summary = mo.md(f"""
            ### igPC results for `{_fi.name}`

            Preprocessing: **{_prep_label}**

            Signal: **{_Nstep+1:,} samples**  ·  Δt = {_dt_u:.2e} s
            ·  duration = {_dur_u:.1f} s  ({_cyc_u:.0f} cycles at {_f0_u:.0f} Hz)

            **Step 1 — Diffusion coefficient**

            | | |
            |---|---|
            | C(0) | {_C0_u:.5f} |
            | λ | {_lam_u:.2f} s⁻¹ |
            | **c = C(0)·λ** | **{_cu:.5f}** |

            **Step 2 — Correction**

            | | |
            |---|---|
            | Regime | {_reg_u} |
            | μ | {_mu_u:.4f} |
            | σ_raw | {_std_u:.4f} |
            | σ_proj | {_sigp_u:.4f} |

            **Step 3 — Identified parameters**

            | Parameter | Value |
            |-----------|-------|
            | ν  (s⁻¹) | {_nu_u:.3f} |
            | κ | {_kap_u:.4f} |
            | Γ  (Pa²·s⁻³) | {_Gam_u:.4e} |
            | c | {_cu:.5f} |
            """)

            _out = mo.vstack([_summary, _figU])

        except Exception as _err:
            _out = mo.callout(
                mo.md(f"**Error processing file:** {_err}"),
                kind="danger",
            )

    _out


if __name__ == "__main__":
    app.run()
