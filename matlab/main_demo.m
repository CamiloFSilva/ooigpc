%% main_demo.m — igPC identification: end-to-end demonstration
%
% Reproduces the numerical example of Section 5.1 of the paper:
%
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame.
%
% Steps
%   1. Simulate the Langevin SDE amplitude envelope A(t)             (§3.1)
%   2. Compute the mean-square displacement MSD(τ) and fit the
%      exponential model → diffusion coefficient c                   (§3.2)
%   3. Apply the two-level non-Gaussian correction → σ               (Suppl. §5)
%   4. Identify ν, κ, Γ via Eqs. (10)–(12)                         (§3.3)
%   5. Plot results
%
% All functions called here are in this same matlab/ folder.
% Required MATLAB toolboxes: Optimization Toolbox (lsqcurvefit),
%                            Statistics and Machine Learning Toolbox (skewness)
%
% -------------------------------------------------------------------------
clear; close all; clc;

%% =========================================================================
%  1.  Physical parameters  (paper §5.1)
% ==========================================================================
omega   = 2*pi*120;        % angular eigenfrequency  ω₀ = 2π·f₀  (f₀ = 120 Hz)
nu_true    = 16.0;         % linear growth rate ν  (s⁻¹)
kappa_true = 3.4;          % nonlinear damping coefficient κ
Gamma_true = 1e7;          % noise intensity Γ  (Pa²·s⁻³)

N  = 2e6;                  % number of time steps
dt = 5e-5;                 % time step Δt (s)

fprintf('True parameters\n');
fprintf('  nu    = %g s^{-1}\n', nu_true);
fprintf('  kappa = %g\n',        kappa_true);
fprintf('  Gamma = %g Pa^2/s^3\n\n', Gamma_true);

%% =========================================================================
%  2.  Simulate the Langevin SDE  (Eq. 3–4)
% ==========================================================================
fprintf('Simulating Langevin SDE  (%d steps) ...\n', N);
A = simulate_langevin(nu_true, kappa_true, Gamma_true, omega, N, dt, 'Seed', 0);
fprintf('  Done.  size(A) = [%d 1]\n\n', numel(A));

%% =========================================================================
%  3.  Mean-square displacement  (Eq. 5) and exponential fit  (Eq. 7)
% ==========================================================================
tau = (2.5e-4 : 2.5e-4 : 0.2)';   % delay times τ  (s)

fprintf('Computing MSD ...\n');
[taus, MSD] = compute_msd(A, tau, dt);

fprintf('Fitting exponential model  MSD(τ) = A_hat·(1 − exp(−τ/T_hat)) ...\n');
[A_hat, T_hat, MSD_fit] = fit_msd(taus, MSD);
c_est = A_hat / (2 * T_hat);

fprintf('  A_hat = %.4f\n',  A_hat);
fprintf('  T_hat = %.4f s\n', T_hat);
fprintf('  c_est = %.4f  (c_true = %.4f)\n\n', ...
    c_est, Gamma_true/(4*omega^2));

%% =========================================================================
%  4.  Identification  (Eqs. 10–12)
% ==========================================================================
% Use the positive-amplitude samples after the transient
A_use = A(5001:end);
A_use = A_use(A_use > 0);

res = identify_igpc(A_use, c_est, 'Omega', omega);

fprintf('Identified parameters\n');
fprintf('  nu    = %.2f s^{-1}  (true: %g,  error: %.1f %%)\n', ...
    res.nu,    nu_true,    100*abs(res.nu    - nu_true)   /abs(nu_true));
fprintf('  kappa = %.3f         (true: %g,  error: %.1f %%)\n', ...
    res.kappa, kappa_true, 100*abs(res.kappa - kappa_true)/abs(kappa_true));
fprintf('  Gamma = %.3e         (true: %g,  error: %.1f %%)\n', ...
    res.Gamma, Gamma_true, 100*abs(res.Gamma - Gamma_true)/abs(Gamma_true));
fprintf('\n  Correction region: %d  (1 = marginally stable, 2 = unstable)\n', ...
    res.region);
fprintf('  mu        = %.4f\n', res.mu);
fprintf('  std_raw   = %.4f\n', res.std_raw);
fprintf('  sigma     = %.4f  (corrected)\n\n', res.sigma);

%% =========================================================================
%  5.  Plots
% ==========================================================================

figure('Name', 'igPC identification results', 'Position', [100 100 1200 400]);

% --- (a) Signal excerpt ------------------------------------------------
subplot(1,3,1);
t_show = (0:999) * dt;
plot(t_show * 1e3, A(1:1000), 'b', 'LineWidth', 0.8);
xlabel('\tau  (ms)');
ylabel('A  (Pa)');
title('(a)  Amplitude envelope  A(t)');
grid on; box on;

% --- (b) MSD and fit ---------------------------------------------------
subplot(1,3,2);
plot(taus * 1e3, MSD,     'b.', 'MarkerSize', 4); hold on;
plot(taus * 1e3, MSD_fit, 'r-', 'LineWidth', 1.5);
xlabel('\tau  (ms)');
ylabel('MSD  (Pa^2)');
title('(b)  MSD and exponential fit');
legend('Empirical', 'Model fit', 'Location', 'southeast');
grid on; box on;

% --- (c) Amplitude PDF -------------------------------------------------
subplot(1,3,3);
[counts, edges] = histcounts(A_use, 80, 'Normalization', 'pdf');
centres = 0.5 * (edges(1:end-1) + edges(2:end));
bar(centres, counts, 1, 'FaceColor', [0.7 0.85 1], 'EdgeColor', 'none'); hold on;

% Rayleigh-like reference PDF: p(A) ∝ A·exp(−A²/(2σ_R²))
A_plot  = linspace(0, max(A_use)*1.1, 300);
mu_A    = mean(A_use);
sigma_R = mu_A / sqrt(pi/2);          % Rayleigh scale from the mean
p_ray   = (A_plot / sigma_R^2) .* exp(-A_plot.^2 / (2*sigma_R^2));
plot(A_plot, p_ray, 'r--', 'LineWidth', 1.5);

xlabel('A  (Pa)');
ylabel('PDF  (Pa^{-1})');
title('(c)  Amplitude PDF');
legend('Empirical', 'Rayleigh ref.', 'Location', 'northeast');
grid on; box on;

sgtitle(sprintf('igPC identification  (\\nu=%.1f, \\kappa=%.2f, \\Gamma=%.1e)', ...
    res.nu, res.kappa, res.Gamma));
