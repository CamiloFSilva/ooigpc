%% main_demo_vdp.m — igPC identification from a Van der Pol signal
%
% End-to-end pipeline identical to main_demo.m, but the input signal is
% produced by the full stochastic Van der Pol oscillator rather than the
% reduced-order Langevin amplitude SDE.  The pressure-like signal η(t) is
% first band-pass filtered, then the amplitude envelope A_H(t) is extracted
% via the Hilbert transform before identification.
%
% Steps
%   1. Simulate the Van der Pol ODE  (Heun + OU noise, matching original solver)
%   2. Band-pass filter η → isolate the thermoacoustic mode
%   3. Extract the Hilbert amplitude envelope  A_H(t)
%   4. Compute MSD(τ) and fit the exponential model → c             (§3.2)
%   5. Apply the two-level non-Gaussian correction → σ               (Suppl. §5)
%   6. Identify ν, κ, Γ via Eqs. (10)–(12)                         (§3.3)
%   7. Plot results
%
% KEY BANDWIDTH RULE
%   The band-pass half-bandwidth BW_HZ must be large enough to pass all
%   amplitude-modulation sidebands (rule of thumb: BW_HZ ≥ 3·ν so that the
%   amplitude relaxation timescale ν⁻¹ fits within the band), yet small
%   enough to exclude harmonics (BW_HZ << f₀).
%   For f₀ = 120 Hz, ν = 16 s⁻¹: recommended BW_HZ = 50 Hz (full BW = 100 Hz).
%   Using BW_HZ = 10–20 Hz clips the sidebands → the Hilbert envelope shows
%   sub-cycle ripple → MSD/(2τ) has a spurious peak instead of monotone decay
%   → c is overestimated by up to a factor of 2.
%
% Controllable parameters
%   SEED       – integer seed for the random noise (0, 1, 2, …).
%   BW_HZ      – half-bandwidth of the band-pass filter (Hz).
%                Paper default: 50 Hz  (full bandwidth 100 Hz).
%   USE_OU     – true  → Heun + calibrated OU noise (matches paper solver)
%                false → symplectic Euler + white noise (faster, small bias)
%
% Required MATLAB toolboxes
%   Signal Processing Toolbox       (butter, filtfilt, hilbert)
%   Curve Fitting Toolbox or lsqcurvefit / fit (for fit_msd)
%   Statistics and Machine Learning (skewness)
%
% -------------------------------------------------------------------------
clear; close all; clc;

%% =========================================================================
%  User-adjustable settings
% ==========================================================================
SEED   = 0;      % random seed  (change for a different realisation)
BW_HZ  = 100;     % band-pass HALF-bandwidth around f₀  (Hz)  — paper: 50
USE_OU = false;   % true = Heun + OU noise (paper), false = symplectic + white

%% =========================================================================
%  1.  Physical parameters  (paper §5.1)
% ==========================================================================
omega      = 2*pi*120;    % angular eigenfrequency  ω₀ = 2π·f₀  (f₀ = 120 Hz)
nu_true    = 16.0;        % linear growth rate ν  (s⁻¹)
kappa_true = 3.4;         % nonlinear damping coefficient κ
Gamma_true = 1e7;         % noise intensity Γ  (Pa²·s⁻³)
c_true     = Gamma_true / (4 * omega^2);   % theoretical diffusion coefficient

N  = 6e6;                 % number of time steps  (paper: 6 e6 → T = 300 s)
dt = 5e-5;                % time step Δt (s)
fs = 1 / dt;              % sampling frequency (Hz)
f0 = omega / (2*pi);      % eigenfrequency  (Hz)

% OU noise parameters (only used when USE_OU = true)
tc_ou = 5e-4;             % OU correlation time τ_c (s)   — paper: 5e-4
BW_full = 2 * BW_HZ;      % full band-pass bandwidth (Hz), used for OU calibration

fprintf('========================================================\n');
fprintf('  igPC identification — Van der Pol signal\n');
fprintf('========================================================\n');
fprintf('  Seed       : %d\n',          SEED);
fprintf('  Bandwidth  : ±%g Hz  (%.0f–%.0f Hz)\n', BW_HZ, f0-BW_HZ, f0+BW_HZ);
fprintf('  Noise      : %s\n',          ternary(USE_OU, 'OU (calibrated, Heun)', 'white (symplectic Euler)'));
fprintf('  N, dt      : %g,  %.0e s  (T = %.0f s)\n', N, dt, N*dt);
fprintf('--------------------------------------------------------\n');
fprintf('True parameters\n');
fprintf('  nu    = %g s^{-1}\n',       nu_true);
fprintf('  kappa = %g\n',              kappa_true);
fprintf('  Gamma = %g Pa^2/s^3\n',     Gamma_true);
fprintf('  c     = %.6f  (= Gamma/(4 omega^2))\n\n', c_true);

%% =========================================================================
%  2.  Simulate the Van der Pol ODE
% ==========================================================================
if USE_OU
    fprintf('Simulating VdP (Heun + OU noise, seed=%d) ...\n', SEED);
    [t_vdp, eta, ~, ~] = simulate_van_der_pol( ...
        nu_true, kappa_true, Gamma_true, omega, N, dt, ...
        'Method', 'heun', ...
        'NoiseType', 'ou', ...
        'TcNoise',     tc_ou, ...
        'BandwidthHz', BW_full, ...
        'Seed', SEED);
else
    fprintf('Simulating VdP (symplectic Euler + white noise, seed=%d) ...\n', SEED);
    [t_vdp, eta, ~, ~] = simulate_van_der_pol( ...
        nu_true, kappa_true, Gamma_true, omega, N, dt, ...
        'Method', 'symplectic', ...
        'NoiseType', 'white', ...
        'Seed', SEED);
end
fprintf('  Done.  size(eta) = [%d 1]\n\n', numel(eta));

%% =========================================================================
%  3.  Band-pass filter  →  Hilbert amplitude envelope
% ==========================================================================
f_low  = f0 - BW_HZ;
f_high = f0 + BW_HZ;
fprintf('Band-pass filtering: %.1f–%.1f Hz (Butterworth, order 4) ...\n', ...
    f_low, f_high);

nyq   = fs / 2;
[b_bp, a_bp] = butter(4, [f_low/nyq, f_high/nyq], 'bandpass');
eta_bp = filtfilt(b_bp, a_bp, double(eta));

fprintf('Hilbert envelope extraction ...\n');
A_hil = abs(hilbert(eta_bp));
A_eq  = sqrt(8*nu_true/kappa_true);
fprintf('  Done.  mean(A_H) = %.4f   (A_eq = sqrt(8nu/kappa) = %.4f)\n\n', ...
    mean(A_hil(5001:end)), A_eq);

%% =========================================================================
%  4.  Mean-square displacement  (Eq. 5)  and exponential fit  (Eq. 7)
% ==========================================================================
tau = (2.5e-4 : 2.5e-4 : 0.2)';   % delay times τ  (s)

fprintf('Computing MSD of Hilbert envelope ...\n');
[taus, MSD] = compute_msd(A_hil, tau, dt);

fprintf('Fitting exponential model  MSD(τ) = A_hat·(1 − exp(−τ/T_hat)) ...\n');
[A_hat, T_hat, MSD_fit] = fit_msd(taus, MSD);
c_est = A_hat / (2 * T_hat);

fprintf('  A_hat = %.4f\n',  A_hat);
fprintf('  T_hat = %.4f s\n', T_hat);
fprintf('  c_est = %.6f  (c_true = %.6f,  ratio = %.3f)\n\n', ...
    c_est, c_true, c_est/c_true);

%% =========================================================================
%  5.  Identification  (Eqs. 10–12)
% ==========================================================================
% Discard transient; use only positive-amplitude samples
A_use = A_hil(5001:end);
A_use = A_use(A_use > 0);

res = identify_igpc(A_use, c_est, 'Omega', omega);

fprintf('--------------------------------------------------------\n');
fprintf('Identified parameters\n');
fprintf('  nu    = %7.2f s^{-1}  (true: %g,   error: %+.1f %%)\n', ...
    res.nu,    nu_true,    100*(res.nu    - nu_true)   /abs(nu_true));
fprintf('  kappa = %7.3f         (true: %g,  error: %+.1f %%)\n', ...
    res.kappa, kappa_true, 100*(res.kappa - kappa_true)/abs(kappa_true));
fprintf('  Gamma = %7.3e         (true: %g,  error: %+.1f %%)\n', ...
    res.Gamma, Gamma_true, 100*(res.Gamma - Gamma_true)/abs(Gamma_true));
fprintf('\n  Correction region : %d  (1 = marginally stable, 2 = unstable)\n', ...
    res.region);
fprintf('  mu                : %.4f\n', res.mu);
fprintf('  std_raw           : %.4f\n', res.std_raw);
fprintf('  sigma (corrected) : %.4f\n', res.sigma);
fprintf('========================================================\n\n');

%% =========================================================================
%  6.  Plots
% ==========================================================================
noise_label = ternary(USE_OU, 'Heun+OU', 'Sympl+White');

figure('Name', 'igPC — Van der Pol signal', ...
       'Position', [80 80 1400 820]);

% ---- (a) η(t) excerpt -----------------------------------------------
subplot(2,3,1);
i_show = 1:2000;
plot(t_vdp(i_show)*1e3, eta(i_show), 'Color', [0.2 0.5 0.8], 'LineWidth', 0.7);
xlabel('Time  (ms)');
ylabel('\eta  (Pa)');
title('(a)  VdP pressure signal  \eta(t)');
grid on; box on;

% ---- (b) Amplitude envelope excerpt ---------------------------------
subplot(2,3,2);
plot(t_vdp(i_show)*1e3, A_hil(i_show), 'Color', [0.85 0.33 0.10], 'LineWidth', 0.9);
hold on;
yline(A_eq, 'k--', 'LineWidth', 1.2, ...
      'Label', sprintf('A_{eq}=%.2f', A_eq), 'LabelHorizontalAlignment', 'left');
xlabel('Time  (ms)');
ylabel('A_H  (Pa)');
title(sprintf('(b)  Hilbert envelope  (BW=\\pm%g Hz)', BW_HZ));
grid on; box on;

% ---- (c) MSD / (2τ) — diagnostic plot ------------------------------
subplot(2,3,3);
plot(taus, MSD ./ (2*taus), 'r.', 'MarkerSize', 3); hold on;
plot(taus, MSD_fit ./ (2*taus), 'b-', 'LineWidth', 1.5);
yline(c_est,  'k--', 'LineWidth', 1.2, ...
      'Label', sprintf('c_{est}=%.4f', c_est), 'LabelHorizontalAlignment', 'left');
yline(c_true, 'r:', 'LineWidth', 1.2, ...
      'Label', sprintf('c_{true}=%.4f', c_true), 'LabelHorizontalAlignment', 'left');
xlabel('\tau  (s)');
ylabel('MSD(\tau) / (2\tau)');
title('(c)  MSD/(2\tau) — extrapolation \tau\rightarrow0');
legend('Data', 'Fit (Eq. 7)', 'Location', 'northeast');
grid on; box on;

% ---- (d) MSD and fit ------------------------------------------------
subplot(2,3,4);
plot(taus*1e3, MSD,     'b.', 'MarkerSize', 4); hold on;
plot(taus*1e3, MSD_fit, 'r-', 'LineWidth', 1.5);
xlabel('\tau  (ms)');
ylabel('MSD  (Pa^2)');
title('(d)  MSD and exponential fit');
legend('Empirical', 'Model fit', 'Location', 'southeast');
grid on; box on;

% ---- (e) Amplitude PDF  with Fokker–Planck overlay ------------------
subplot(2,3,5);
[counts, edges] = histcounts(A_use, 100, 'Normalization', 'pdf');
centres = 0.5*(edges(1:end-1) + edges(2:end));
bar(centres, counts, 1, 'FaceColor', [0.7 0.85 1], 'EdgeColor', 'none'); hold on;

% Theoretical Fokker-Planck PDF: p(A) ∝ A · exp(a·A²/(2c) − b·A⁴/(4c))
a_fp  = nu_true;
b_fp  = kappa_true / 8;
A_grd = linspace(min(A_use)*0.5, max(A_use)*1.15, 500);
logp  = log(A_grd) + (a_fp*A_grd.^2)/(2*c_true) - (b_fp*A_grd.^4)/(4*c_true);
logp  = logp - max(logp);
p_fp  = exp(logp);
p_fp  = p_fp / trapz(A_grd, p_fp);
plot(A_grd, p_fp, 'r-', 'LineWidth', 1.8);
xline(A_eq, 'r--', 'LineWidth', 1.2);
xline(res.mu, 'k-', 'LineWidth', 1.2, ...
      'Label', sprintf('\\mu=%.2f', res.mu), 'LabelVerticalAlignment', 'bottom');
xlabel('A  (Pa)');
ylabel('PDF  (Pa^{-1})');
title('(e)  Amplitude PDF');
legend('Hilbert envelope', 'Fokker–Planck (true params)', 'Location', 'northeast');
grid on; box on;

% ---- (f) Identification summary bar ---------------------------------
subplot(2,3,6);
params_true = [nu_true; kappa_true; Gamma_true];
params_est  = [res.nu;  res.kappa;  res.Gamma];
err_pct     = 100*(params_est - params_true)./abs(params_true);

bar_h = bar(err_pct, 'FaceColor', 'flat');
bar_h.CData(err_pct >= 0, :) = repmat([0.2 0.6 0.4], sum(err_pct >= 0), 1);
bar_h.CData(err_pct <  0, :) = repmat([0.8 0.3 0.2], sum(err_pct <  0), 1);
yline(0, 'k-', 'LineWidth', 1);
set(gca, 'XTickLabel', {'\nu', '\kappa', '\Gamma'});
ylabel('Relative error  (%)');
title('(f)  Identification errors');
grid on; box on;

for k = 1:3
    text(k, err_pct(k) + sign(err_pct(k))*0.5, ...
         sprintf('%+.1f%%', err_pct(k)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end

sgtitle(sprintf(['igPC — Van der Pol   %s   seed=%d,  BW=\\pm%g Hz\n' ...
    '\\nu_{id}=%.1f s^{-1},  \\kappa_{id}=%.3f,  \\Gamma_{id}=%.2e'], ...
    noise_label, SEED, BW_HZ, res.nu, res.kappa, res.Gamma), 'FontSize', 11);

%% =========================================================================
%  Helper
% ==========================================================================
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
