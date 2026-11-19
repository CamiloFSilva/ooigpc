% =========================================================================
% Plot_MSD_TauMaxConvergence.m  —  MSD tau_max convergence diagnostic
%
% Description:
%   Checks how the recovered diffusion coefficient c and the envelope
%   statistics (mu, sigma, gamma) depend on the maximum lag tau_max used
%   in the MSD exponential fit.
%
%   For each tau_max value in kvec, the script:
%     1. Builds the MSD curve over tau_val = [dt : dt : tau_max] with
%        a fixed step dt = 10*dt_sim
%     2. Fits MSD(tau) ≈ MSD_amp*(1 − exp(−tau/T_drift)) → c = MSD_amp / (2*T_drift)
%     3. Records mu, sigma, skewness-related gamma from the signal window
%
%   All quantities are normalised to their value at the LARGEST tau_max
%   and plotted together so that sensitivity to tau_max can be judged.
%
%   The script can be run on either VdP simulation data or experimental
%   envelope data (see DATA SOURCE section below).
%
%   Output figure: ConvergenceStatistics_taufit_<tag>.pdf
%
% Input files:
%   VdP simulation:  ../DATA/Data_nu<nu>.mat  → A (cell), dt, Gamma, wn
%   Experimental:    ../../Experiments/DATA/envelopeRef_B<bw>Exp.mat
%                    → A (cell), dt_vec, fn_vec, time_vec
%
% Author : C. F. Silva  (camilo.silva@tum.de)
%          Professorship of Thermofluiddynamics, TU Munich
% Paper  : Silva et al. (2026). Output-only identification of
%          thermoacoustic systems using polynomial chaos expansions.
%          Combust. Flame.
% =========================================================================

clear all

% =========================================================================
% USER OPTIONS
% =========================================================================

use_experimental = true;    % true  → load experimental data
                            % false → load VdP simulation data

% --- Experimental data settings (used when use_experimental = true) ---
bandwidth = 40;             % Filter bandwidth [Hz]
ns_exp    = 5;              % Operating-condition index (1–5)

% --- VdP simulation settings (used when use_experimental = false) ---
nu_vdp = 24;                % Growth rate [1/s]
ns_vdp = 1;                 % Simulation replicate index

% --- tau_max sweep ---
kvec     = 0.05 : 0.01 : 0.5;   % tau_max values to test [s]
k_fixed  = 10;                   % Fixed step multiplier (dt = k_fixed*dt_sim)
n_skip   = 5000;                 % Samples to skip at start (burn-in)

save_figs = true;           % true  → save PDF figure

% =========================================================================
% DATA SOURCE
% =========================================================================

if use_experimental
    file = ['../../Experiments/DATA/envelopeRef_B', num2str(bandwidth), 'Exp.mat'];
    tmp  = load(file, 'A', 'dt_vec');
    env_cell = tmp.A;
    dt_sim   = tmp.dt_vec(ns_exp);
    tag      = ['Expns', num2str(ns_exp)];
else
    file     = ['../DATA/Data_nu', num2str(nu_vdp), '.mat'];
    tmp      = load(file, 'A', 'dt');
    env_cell = tmp.A;
    dt_sim   = tmp.dt;
    tag      = ['nu', num2str(nu_vdp)];
end

% Select the envelope replicate consistent with the chosen data source
if use_experimental
    x = env_cell{ns_exp};
else
    x = env_cell{ns_vdp};
end
N  = length(x);
dt = k_fixed * dt_sim;   % Fixed MSD step

% =========================================================================
% tau_max SWEEP
% =========================================================================

n_kvec     = length(kvec);
c_recov    = zeros(1, n_kvec);
mu_global  = zeros(1, n_kvec);
std_global = zeros(1, n_kvec);
gamma_3    = zeros(1, n_kvec);

ft = fittype('MSD_sat * (1 - exp(-taus / T_drift))', ...
    'independent', 'taus', ...
    'coefficients', {'MSD_sat', 'T_drift'});

for kv = 1 : n_kvec

    tau_max = kvec(kv);
    tau_val = dt : dt : tau_max;
    tk_vec  = floor(tau_val / dt_sim);

    MSD = zeros(1, length(tk_vec));

    for kk = 1 : length(tk_vec)
        tk       = tk_vec(kk);
        xvec1    = x(n_skip : N - tk)';
        xvec2    = x(n_skip + tk : N)';
        deltax   = xvec1 - xvec2;
        MSD(kk)  = sum(deltax.^2) / length(deltax);
    end

    taus = tk_vec * dt_sim;
    f    = fit(taus(:), MSD(:), ft, 'StartPoint', [0.1, 0.01]);

    MSD_amp  = f.MSD_sat;
    T_drift  = f.T_drift;
    c_recov(kv)    = MSD_amp / (2 * T_drift);
    mu_global(kv)  = mean(xvec2);
    std_global(kv) = std(xvec2);
    skew_val       = skewness(xvec2);
    gamma_3(kv)    = skew_val * std_global(kv)^3;

end

% =========================================================================
% PLOT — normalised statistics vs. tau_max
% =========================================================================

figure(1); clf
plot(kvec, mu_global  / mu_global(end),  'rx', 'LineWidth', 2, 'MarkerSize', 5); hold on
plot(kvec, std_global / std_global(end), 'b^', 'LineWidth', 2, 'MarkerSize', 5)
plot(kvec, gamma_3    / gamma_3(end),    'ms', 'LineWidth', 2, 'MarkerSize', 5)
plot(kvec, c_recov    / c_recov(end),    'ko', 'LineWidth', 2, 'MarkerSize', 5)
hold off
legend('\mu / \mu_{end}', '\sigma / \sigma_{end}', '\gamma / \gamma_{end}', ...
    'c / c_{end}', 'Location', 'best')
xlabel('\tau_{max}  [s]');  ylabel('Normalised statistic')
axis([0, kvec(end), 0.9, 1.1])
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 20], ...
    'PaperPosition', [0 0 20 20])
if save_figs
    saveas(gcf, ['./ConvergenceStatistics_taufit_', tag, '.pdf'])
end
