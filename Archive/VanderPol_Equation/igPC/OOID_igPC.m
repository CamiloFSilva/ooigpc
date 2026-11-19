% OOID_igPC.m  (Van der Pol version)
% =========================================================================
% Output-only identification of Van der Pol data via igPC
%
% DESCRIPTION
%   Loads pre-computed amplitude envelope data produced by vderPol_Solver.m
%   and identifies the system parameters using the igPC method.
%
%   The identification follows the same steps as the Langevin version:
%     1. Estimate diffusion coefficient  c  via Mean Square Displacement
%     2. Apply optional two-level correction to  σ
%     3. Recover  ν  and  κ  from closed-form igPC formulas (Eqs. 10-11)
%
%   An optional amplitude-binning step (BIN = 'YES') replaces the
%   continuous amplitude trace with a step function over discrete bins.
%   This reduces MSD noise at the cost of some spectral resolution and
%   is currently not used in the published results.
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
%
% DATA INPUT
%   ../DATA/Data_nu<nu>.mat  — produced by vderPol_Solver.m
%   Expected variables: A (cell array of envelopes), dt, Gamma, wn
%
% SEE ALSO
%   vderPol_Solver.m          — generates the input data
%   OOID_CorrectionFactors.m  — derives and validates the σ correction
%   OOID_KM_PDFMatrix.m       — Kramers-Moyal coefficient benchmark
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

% Identification mode
%   'NoCorrection'        — raw σ fed directly into igPC
%   'CorrectionTwoLevels' — two-level σ correction applied before igPC
%CASE = 'NoCorrection';
CASE = 'CorrectionTwoLevels';

% Amplitude binning (reduces MSD noise by discretising the amplitude axis)
%   'YES' — bin amplitude into nbin intervals before MSD computation
%   'NOP' — use the continuous amplitude trace (standard, used in paper)
%BIN = 'YES';
BIN = 'NOP';

save_figs = false;   % true  → save MSD diagnostic figures to ../Plotting/
                    % false → display figures without saving

save_data = true;   % true  → save identified parameters to ../DATA/
                    % false → run without writing .mat file

% =========================================================================
%% PHYSICAL PARAMETERS  (ground truth for validation)
% =========================================================================
%
%   Loaded from file: omega (wn), Gamma.
%   Set below as defaults; overwritten after load() if present in file.

kappa = 3.4;        % [m⁻²·s⁻¹]  nonlinear saturation coefficient
                    % (used only for displaying true vs. identified values)

% =========================================================================
%% SCAN PARAMETERS
% =========================================================================

% Time-lag vector for MSD estimation  [s]
tau_val = 0.00025 : 0.00025 : 0.2;
%tau_val = 0.001 : 0.001 : 0.031;      % coarser range

NumSim = 100;    % number of independent realisations (must match vderPol_Solver.m)

% Growth rates to loop over — must match the saved DATA/ files  [rad/s]
%nu_vec = [16];
nu_vec = [16,20,24];

% =========================================================================
%% IDENTIFICATION LOOP
% =========================================================================

for nk = 1:length(nu_vec)

    nu_true = nu_vec(nk);

    % Pre-allocate output vectors
    nu_hat_vec = zeros(NumSim, 1);
    b_hat_vec  = zeros(NumSim, 1);
    c_hat_vec  = zeros(NumSim, 1);
    xstats     = zeros(NumSim, 2);

    for ns = 1:NumSim

        fprintf('\n--- Realisation %d / %d  (ν_true = %g rad/s) ---\n', ...
                ns, NumSim, nu_true);

        % ------------------------------------------------------------------
        % 1. Load pre-computed envelope data
        % ------------------------------------------------------------------
        % Load the cell array of amplitude envelopes from vderPol_Solver.m.
        % IMPORTANT: rename to env_cell immediately to avoid collision with
        % the MSD fit amplitude  MSD_amp = f.MSD_sat  computed below.

        file = ['../DATA/Data_nu', num2str(nu_true), '.mat'];
        load(file, 'A', 'dt', 'Gamma', 'wn')
        env_cell = A;    % A{ns} = envelope from realisation ns
        clear A          % free the name  A  for use by the MSD fitter

        x      = env_cell{ns};    % [m]  amplitude envelope
        dt_sim = dt;              % [s]  time step of the VdP simulation
        N      = length(x);
        omega  = wn;              % [rad/s] natural angular frequency

        b_true = kappa / 8;
        c_true = Gamma / (4*omega^2);

        % ------------------------------------------------------------------
        % 2. Stationary statistics  (skip initial transient: first 5000 pts)
        % ------------------------------------------------------------------
        x_stat = x(5000:end);
        mu_A   = mean(x_stat);
        std_A  = std(x_stat);
        var_A  = var(x_stat);     % kept for reference

        if isnan(mu_A) || length(x_stat) < 100
            warning('Realisation %d: empty or NaN envelope — skipping.', ns);
            continue
        end

        % ------------------------------------------------------------------
        % 3. Mean Square Displacement (MSD) estimation of  c
        % ------------------------------------------------------------------
        % MSD(τ) = E[(A(t+τ) − A(t))²]
        % Saturates as an exponential for the Langevin process:
        %   MSD(τ) ≈ MSD_amp · [1 − exp(−τ/τ_drift)],   c = MSD_amp/(2·τ_drift)

        tk_vec = floor(tau_val / dt_sim);    % lag indices

        MSD  = zeros(1, length(tk_vec));
        dMSD = zeros(1, length(tk_vec));

        for kk = 1:length(tk_vec)
            tk     = tk_vec(kk);
            dt_lag = tk * dt_sim;            % [s] physical time lag

            xvec1 = x(5000 : N-tk)';        % reference segment
            xvec2 = x(5000+tk : N)';        % delayed segment

            % --- Optional amplitude binning ---
            if strcmp(BIN, 'YES')
                nbin = 30;
                Amin  = min(env_cell{ns});
                Amax  = max(env_cell{ns});
                dA    = (Amax - Amin) / nbin;
                edges = [Amin, linspace(Amin+dA, Amax, nbin)];

                idx    = discretize(env_cell{ns}, edges);
                Afloor = edges(idx);

                xvec1 = Afloor(5000 : N-tk);
                xvec2 = Afloor(5000+tk : N);
            end

            deltax   = xvec1 - xvec2;
            MSD(kk)  = mean(deltax.^2);
            dMSD(kk) = MSD(kk) / dt_lag;
        end

        if any(isnan(MSD)) || all(MSD == 0)
            warning('Realisation %d: MSD is NaN or zero — skipping.', ns);
            continue
        end

        % ------------------------------------------------------------------
        % 4. Fit exponential MSD model → extract  c_hat
        % ------------------------------------------------------------------
        taus = tk_vec * dt_sim;   % physical time lags [s]

        ft = fittype('MSD_sat * (1 - exp(-taus/T_drift))', ...
            'independent', 'taus', ...
            'coefficients', {'MSD_sat', 'T_drift'});
        f = fit(taus(:), MSD(:), ft, 'StartPoint', [0.1, 0.01]);

        MSD_amp   = f.MSD_sat;    % [m²]  MSD saturation amplitude
        tau_drift = f.T_drift;    % [s]   exponential time constant
        c_hat     = MSD_amp / 2 / tau_drift;   % [m²·s⁻¹] diffusion coefficient

        if isnan(c_hat) || c_hat <= 0
            warning('Realisation %d: unphysical c_hat = %g — skipping.', ns, c_hat);
            continue
        end

        MSDfit  = MSD_amp * (1 - exp(-taus / tau_drift));
        dMSDfit = MSDfit ./ taus;
        dMSD    = MSD    ./ taus;    % re-compute for plotting

        fprintf('  c estimated:  c_hat = %.6g   (c_true = %.6g)\n', c_hat, c_true);

        % ------------------------------------------------------------------
        % 5. MSD diagnostic figures  (see paper for context)
        % ------------------------------------------------------------------
        if save_figs

            % Figure 20 — MSD(τ) with exponential fit
            figure(20); clf
            plot(taus, MSD,    'rx', 'MarkerSize', 4, 'LineWidth', 1); hold on
            plot(taus, MSDfit, 'b-', 'MarkerSize', 8, 'LineWidth', 2)
            xlabel('\tau  [s]');  ylabel('MSD  [m^2]')
            title(['MSD fit — \nu = ' num2str(nu_true) ' rad/s'])
            grid on
            ax = gca;  ax.YGrid = 'on';  ax.YMinorGrid = 'on';  ax.XGrid = 'off';
            axis([0 0.2 0 0.3])
            ax.XAxis.Exponent = 0;
            set(gca, 'FontSize', 20)
            set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                     'PaperPosition', [0 0 10 10])
            saveas(gcf, '../Plotting/MSD_VdP_16.pdf')

            % Figure 21 — D₂(τ) = dMSD/2  (extrapolated to τ→0 gives  c)
            figure(21); clf
            plot(taus, dMSD/2,    'rx', 'MarkerSize', 4, 'LineWidth', 1); hold on
            plot(taus, dMSDfit/2, 'b-', 'MarkerSize', 8, 'LineWidth', 2)
            xlabel('\tau  [s]');  ylabel('D_2(\tau)  [m^2 s^{-1}]')
            title(['Diffusion coefficient extrapolation — \nu = ' num2str(nu_true)])
            grid on
            ax = gca;  ax.YGrid = 'on';  ax.YMinorGrid = 'on';  ax.XGrid = 'off';
            axis([0 0.2 0 4.5])
            ax.XAxis.Exponent = 0;
            set(gca, 'FontSize', 20)
            set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                     'PaperPosition', [0 0 10 10])
            saveas(gcf, '../Plotting/Extrapolation_VdP_16.pdf')

        end

        % ------------------------------------------------------------------
        % 6. Optional correction to σ before igPC identification
        % ------------------------------------------------------------------
        % The two-level correction compensates for the bias that the PC
        % variance formula  Var[A] = x₁² + 2x₂²  introduces in σ.
        % See OOID_CorrectionFactors.m for derivation.

        switch CASE

            case 'NoCorrection'
                delta_sigma = 0;
                sigma_eff   = std_A;

            case 'CorrectionTwoLevels'
                std_crit = 0.28 * mu_A;    % boundary between Region 1 and 2

                if std_A > std_crit
                    % --- Region 2  (σ > 0.28·μ, high-noise regime) -------
                    c1_std = 0.144 * std_A + 0.0126;   % upper boundary correction
                    c2_std = -0.035 * std_A;            % lower boundary correction

                    std_hi = 0.423 * mu_A;
                    std_lo = 0.280 * mu_A;   % = std_crit

                    beta_interp = (std_A - std_lo) / (std_hi - std_lo);
                    delta_sigma = (1 - beta_interp)*c2_std + beta_interp*c1_std;
                    sigma_eff   = std_A + delta_sigma;
                    disp('Correction region: 2 (high noise)')

                else
                    % --- Region 1  (σ ≤ 0.28·μ, low-noise regime) --------
                    %   4x³ − 6σ²x + γ₃ = 0,  γ₃ = skew·σ³
                    %   δσ = 2.6 · Re(x₂)
                    skew_A     = skewness(x_stat);
                    gamma_skew = skew_A * std_A^3;

                    x2_roots    = roots([4,  0,  -6*std_A^2,  gamma_skew]);
                    delta_sigma = 2.6 * real(x2_roots(3));
                    sigma_eff   = sqrt(std_A^2 - 2*delta_sigma^2);
                    disp('Correction region: 1 (low noise)')
                end

        end   % switch CASE

        % ------------------------------------------------------------------
        % 7. igPC identification  (Eqs. 10-11, Silva et al. 2026)
        % ------------------------------------------------------------------
        [nu_hat, b_hat] = igpc_identify(mu_A, sigma_eff, c_hat);

        kappa_hat = 8 * b_hat;
        Gamma_hat = c_hat * 4 * omega^2;

        % Store results
        nu_hat_vec(ns) = nu_hat;
        b_hat_vec(ns)  = b_hat;
        c_hat_vec(ns)  = c_hat;
        xstats(ns,1)   = mu_A;
        xstats(ns,2)   = std_A;

        fprintf('  ν    identified: %8.4f   true: %8.4f  [rad/s]\n',   nu_hat,    nu_true);
        fprintf('  κ    identified: %8.4f   true: %8.4f  [m⁻²s⁻¹]\n', kappa_hat, kappa);
        fprintf('  Γ    identified: %8.3g   true: %8.3g  [m²s⁻³]\n',  Gamma_hat, Gamma);

    end   % ns loop

    % Save identified parameters (controlled by save_data flag above)
    params_true = [nu_true,  b_true,  c_true];
    params_hat  = [nu_hat_vec, b_hat_vec, c_hat_vec];

    if save_data
        file = ['../DATA/nu_', num2str(nu_true), '_', CASE, '_', BIN, '_fromigPC.mat'];
        save(file, 'params_true', 'params_hat', 'x_stat', 'xstats')
        fprintf('Data saved to: %s\n', file);
    end

end   % nk loop

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function [nu_hat, b_hat] = igpc_identify(mu, sigma, c)
% igpc_identify  Closed-form igPC identification of the Langevin parameters
%
%   [nu_hat, b_hat] = igpc_identify(mu, sigma, c)
%
%   INPUT
%     mu     — mean amplitude  E[A]
%     sigma  — (corrected) standard deviation  std[A]
%     c      — diffusion coefficient  c = Γ/(4ω₀²)
%
%   OUTPUT
%     nu_hat — identified linear growth rate  ν  [rad/s]
%     b_hat  — identified saturation parameter  b = κ/8  [m⁻²·s⁻¹]
%
%   EQUATIONS (10-11), Silva et al. (2026):
%     A1 = ½(9σ⁴ + 2μ²σ² − μ⁴)
%     B1 = ½(3σ² − μ²)
%     D  = μ²σ²(σ² − μ²)
%     ν̂  = (A1/D) · c
%     b̂  = (B1/D) · c

    A1    = 0.5*(9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
    B1    = 0.5*(3*sigma^2 - mu^2);
    D     = mu^2 * sigma^2 * (sigma^2 - mu^2);
    nu_hat = (A1/D) * c;
    b_hat  = (B1/D) * c;
end
