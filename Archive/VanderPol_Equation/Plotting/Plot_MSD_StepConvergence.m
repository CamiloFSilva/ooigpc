% =========================================================================
% Plot_MSD_StepConvergence.m  —  MSD step-size convergence diagnostic
%
% Description:
%   Checks how the recovered diffusion coefficient c and the envelope
%   statistics (mu, sigma, gamma) depend on the MSD time step dt = k*dt_sim.
%
%   For each step multiplier k in kvec, the script:
%     1. Builds the MSD curve over tau_val = [dt : dt : 0.5 s]
%     2. Fits MSD(tau) ≈ MSD_amp*(1 − exp(−tau/T_drift)) → c = MSD_amp / (2*T_drift)
%     3. Records mu, sigma, skewness-related gamma from the signal window
%
%   All quantities are normalised to their value at the smallest step size
%   and plotted together so that sensitivity to k can be judged visually.
%
%   The script can be run on either VdP simulation data or experimental
%   envelope data (see DATA SOURCE section below).
%
%   Output figure: ConvergenceStatistics3_<tag>.pdf
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
nu_vdp = 4;                 % Growth rate [1/s]
ns_vdp = 1;                 % Simulation replicate index

% --- MSD sweep ---
kvec    = 5 : 10 : 205;    % Step-size multipliers k (dt = k*dt_sim)
n_skip  = 5000;            % Samples to skip at start of signal (burn-in)
tau_max = 0.5;             % Maximum lag for MSD fit [s]

save_figs = true;           % true  → save PDF figure

% =========================================================================
% DATA SOURCE
% =========================================================================

if use_experimental
    file     = ['../../Experiments/DATA/envelopeRef_B', num2str(bandwidth), 'Exp.mat'];
    tmp      = load(file, 'A', 'dt_vec');
    env_cell = tmp.A;
    dt_sim   = tmp.dt_vec(ns_exp);
    tag      = ['Expns', num2str(ns_exp)];
    x        = env_cell{ns_exp};
else
    file     = ['../DATA/Data_nu', num2str(nu_vdp), '.mat'];
    tmp      = load(file, 'A', 'dt');
    env_cell = tmp.A;
    dt_sim   = tmp.dt;
    tag      = ['nu', num2str(nu_vdp)];
    x        = env_cell{ns_vdp};
end

N = length(x);

% =========================================================================
% MSD STEP-SIZE SWEEP
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

    k    = kvec(kv);
    dt   = k * dt_sim;
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
% PLOT — normalised statistics vs. step size
% =========================================================================

t_axis = kvec * dt_sim;

figure(1); clf
plot(t_axis, mu_global  / mu_global(1),  'rx', 'LineWidth', 2, 'MarkerSize', 5); hold on
plot(t_axis, std_global / std_global(1), 'b^', 'LineWidth', 2, 'MarkerSize', 5)
plot(t_axis, gamma_3    / gamma_3(1),    'ms', 'LineWidth', 2, 'MarkerSize', 5)
plot(t_axis, c_recov    / c_recov(1),    'ko', 'LineWidth', 2, 'MarkerSize', 5)
hold off
legend('\mu / \mu_0', '\sigma / \sigma_0', '\gamma / \gamma_0', 'c / c_0', ...
    'Location', 'best')
xlabel('dt  [s]');  ylabel('Normalised statistic')
axis([0, t_axis(end), 0.9995, 1.0005])
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 20], ...
    'PaperPosition', [0 0 20 20])
if save_figs
    saveas(gcf, ['./ConvergenceStatistics3_', tag, '.pdf'])
end
