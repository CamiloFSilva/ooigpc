% OOID_igPC.m
% =========================================================================
% Output-only identification of the Langevin amplitude equation via igPC
%
% DESCRIPTION
%   Integrates the stochastic amplitude (Langevin) equation via Euler-
%   Maruyama and identifies the system parameters using the igPC method.
%
%   The amplitude equation is:
%     dA/dt = ν·A − b·A³ + c/A + ζ(t),   b = κ/8,  c = Γ/(4ω₀²)
%
%   The diffusion coefficient  c  is estimated from the Mean Square
%   Displacement (MSD) of the amplitude signal:
%     MSD(τ) → MSD_amp·[1 − exp(−τ/τ_drift)],   c = MSD_amp/(2·τ_drift)
%
%   Growth rate  ν  and saturation  b = κ/8  are then recovered from
%   the closed-form igPC formulas (Eqs. 10-11, Silva et al. 2026).
%
%   An optional two-level correction can be applied to the measured σ
%   before entering the igPC formulas; see OOID_CorrectionFactors.m
%   for the derivation and validation of the correction.
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
%
% SEE ALSO
%   OOID_CorrectionFactors.m   — derives and validates the σ correction
%   OOID_KM_PDFMatrix.m        — Kramers-Moyal coefficient benchmark
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

save_figs = false;   % true  → save MSD diagnostic figures to ../Plotting/
                    % false → display figures without saving

save_data = true;  % true  → save identified parameters to ../DATA/
                    % false → run without writing .mat file

% =========================================================================
%% PHYSICAL PARAMETERS  (ground truth for validation)
% =========================================================================
%
%   Van der Pol → amplitude equation:  dA/dt = ν·A − b·A³ + c/A + ζ(t)
%     ν   = (β − α)/2             [rad/s]  linear growth rate
%     b   = κ/8                   [m⁻²·s⁻¹]  saturation parameter
%     c   = Γ/(4ω₀²)             [m²·s⁻¹]  diffusion coefficient
%
% Set ν via nu_vec below; kappa, Gamma and omega are fixed here.

kappa = 3.4;        % [m⁻²·s⁻¹]  nonlinear saturation coefficient
Gamma = 1e7;        % [m²·s⁻³]   white-noise forcing intensity
omega = 2*pi*120;   % [rad/s]     natural angular frequency  (f₀ = 120 Hz)

% =========================================================================
%% SCAN PARAMETERS
% =========================================================================

% Time-lag vector for MSD estimation  [s]
%tau_val = 0.00025 : 0.00025 : 0.2;
tau_val = 0.00025 : 0.00025 : 0.05;  % shorter range

NumSim = 100;    % number of independent realisations (1 → single run)

% Growth rates to loop over  [rad/s]
%nu_vec = [16];
nu_vec = [0, 4, 8, 12, 16, 20, 24];

% =========================================================================
%% IDENTIFICATION LOOP
% =========================================================================

for nk = 1:length(nu_vec)

    nu_true = nu_vec(nk);
    b_true  = kappa / 8;
    c_true  = Gamma / (4*omega^2);

    % Pre-allocate output vectors
    nu_hat_vec = zeros(NumSim, 1);
    b_hat_vec  = zeros(NumSim, 1);
    c_hat_vec  = zeros(NumSim, 1);
    xstats     = zeros(NumSim, 2);

    for ns = 1:NumSim

        fprintf('\n--- Simulation %d / %d  (ν_true = %g rad/s) ---\n', ...
                ns, NumSim, nu_true);

        % ------------------------------------------------------------------
        % 1. Simulate the discrete Langevin amplitude equation
        % ------------------------------------------------------------------
        % Itô Euler-Maruyama step:
        %   A_{k+1} = A_k + ν·A_k·dt − b·A_k³·dt + (c/A_k)·dt + √(2c·dt)·ε_k
        % where ε_k ~ Truncated-Normal(0,1) on [−4.5, 4.5].

        N      = 2000000;               % number of Euler steps
        dt_sim = 5e-5;                  % [s] time step
        s1     = sqrt(2*c_true*dt_sim); % noise amplitude per step

        % Truncated-normal noise (avoids outlier kicks)
        pd    = makedist('Normal');
        pd_tr = truncate(pd, -4.5, 4.5);
        noise = random(pd_tr, N, 1);

        x    = zeros(1, N);
        x(1) = 0.65;   % initial condition (arbitrary positive value)

        for k = 1:N
            x(k+1) = x(k) ...
                   + nu_true  * dt_sim * x(k) ...
                   - b_true   * dt_sim * x(k)^3 ...
                   + (c_true  / x(k)) * dt_sim ...
                   + s1 * noise(k);
        end

        % ------------------------------------------------------------------
        % 2. Stationary statistics  (skip initial transient: first 5000 pts)
        % ------------------------------------------------------------------
        % Compute from the full stationary signal, not from a shifted window,
        % so that stats do not depend on the MSD time lag.

        x_stat = x(5000:end);
        x_stat = x_stat(x_stat > 0);   % discard unphysical negative values

        mu_A  = mean(x_stat);
        std_A = std(x_stat);
        var_A = var(x_stat);           % kept for reference; not used in igPC

        % ------------------------------------------------------------------
        % 3. Mean Square Displacement (MSD) estimation of  c
        % ------------------------------------------------------------------
        % MSD(τ) = E[(A(t+τ) − A(t))²]
        % For the Langevin process this saturates as an exponential,
        % and the initial slope gives the diffusion coefficient:
        %   c = MSD_amp / (2·τ_drift)

        tk_vec = floor(tau_val / dt_sim);   % lag indices

        for kk = 1:length(tk_vec)
            tk     = tk_vec(kk);
            dt_lag = tk * dt_sim;           % [s] physical time lag

            xvec1  = x(5000 : N-tk)';      % reference segment
            xvec2  = x(5000+tk : N)';      % delayed segment

            % Discard unphysical negative values
            pos    = xvec1 > 0;
            xvec1  = xvec1(pos);
            xvec2  = xvec2(pos);

            if isempty(xvec1) || length(xvec1) ~= length(xvec2)
                MSD(kk)  = 0;
                dMSD(kk) = 0;
                continue
            end

            deltax   = xvec1 - xvec2;
            MSD(kk)  = mean(deltax.^2);
            dMSD(kk) = MSD(kk) / dt_lag;
        end

        if isnan(mu_A) || MSD(1) == 0
            warning('Simulation %d: invalid MSD — skipping.', ns);
            continue
        end

        % ------------------------------------------------------------------
        % 4. Fit exponential MSD model → extract  c_hat
        % ------------------------------------------------------------------
        taus = tk_vec * dt_sim;   % physical time-lag vector [s]

        ft = fittype('MSD_sat * (1 - exp(-taus/T_drift))', ...
            'independent', 'taus', ...
            'coefficients', {'MSD_sat', 'T_drift'});
        f = fit(taus(:), MSD(:), ft, 'StartPoint', [0.1, 0.01]);

        MSD_amp   = f.MSD_sat;    % [m²]  MSD saturation level
        tau_drift = f.T_drift;    % [s]   exponential time constant
        c_hat     = MSD_amp / 2 / tau_drift;   % [m²·s⁻¹] diffusion coefficient

        MSDfit  = MSD_amp * (1 - exp(-taus / tau_drift));
        dMSDfit = MSDfit ./ taus;
        dMSD    = MSD    ./ taus;   % re-compute for plotting

        fprintf('  c estimated:  c_hat = %.6g   (c_true = %.6g)\n', c_hat, c_true);

        % ------------------------------------------------------------------
        % 5. MSD diagnostic figures  (see paper Fig. 6 for context)
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
            ax.XAxis.Exponent = 0;
            set(gca, 'FontSize', 20)
            set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                     'PaperPosition', [0 0 10 10])
            saveas(gcf, '../Plotting/MSD_Langevin_16.pdf')

            % Figure 21 — D₂(τ) = dMSD/2  (extrapolated to τ→0 gives  c)
            figure(21); clf
            plot(taus, dMSD/2,    'rx', 'MarkerSize', 4, 'LineWidth', 1); hold on
            plot(taus, dMSDfit/2, 'b-', 'MarkerSize', 8, 'LineWidth', 2)
            xlabel('\tau  [s]');  ylabel('D_2(\tau)  [m^2 s^{-1}]')
            title(['Diffusion coefficient extrapolation — \nu = ' num2str(nu_true)])
            grid on
            ax = gca;  ax.YGrid = 'on';  ax.YMinorGrid = 'on';  ax.XGrid = 'off';
            ax.XAxis.Exponent = 0;
            set(gca, 'FontSize', 20)
            set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                     'PaperPosition', [0 0 10 10])
            saveas(gcf, '../Plotting/Extrapolation_Langevin_16.pdf')

        end

        % ------------------------------------------------------------------
        % 6. Optional correction to σ before igPC identification
        % ------------------------------------------------------------------
        % The two-level correction compensates for the bias that the PC
        % variance formula  Var[A] = x₁² + 2x₂²  introduces in σ.
        % See OOID_CorrectionFactors_V5.m for derivation.

        switch CASE

            case 'NoCorrection'
                % No correction; use measured standard deviation directly
                delta_sigma = 0;
                sigma_eff   = std_A;

            case 'CorrectionTwoLevels'
                std_crit = 0.28 * mu_A;   % boundary between Region 1 and 2

                if std_A > std_crit
                    % --- Region 2  (σ > 0.28·μ, high-noise regime) -------
                    % Additive interpolation between two linear correction curves
                    c1_std = 0.144 * std_A + 0.0126;   % upper boundary correction
                    c2_std = -0.035 * std_A;            % lower boundary correction

                    std_hi = 0.423 * mu_A;   % σ at upper correction boundary
                    std_lo = 0.280 * mu_A;   % σ at lower boundary (= std_crit)

                    beta_interp = (std_A - std_lo) / (std_hi - std_lo);
                    delta_sigma = (1 - beta_interp)*c2_std + beta_interp*c1_std;
                    sigma_eff   = std_A + delta_sigma;
                    disp('Correction region: 2 (high noise)')

                else
                    % --- Region 1  (σ ≤ 0.28·μ, low-noise regime) --------
                    % Cubic-root correction from the third central moment γ₃:
                    %   4·x³ − 6σ²·x + γ₃ = 0,   γ₃ = skew·σ³
                    %   δσ = 2.6 · Re(x₂)
                    skew_A     = skewness(x_stat);
                    gamma_skew = skew_A * std_A^3;   % third central moment

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
    params_true = [nu_true, b_true, c_true];
    params_hat  = [nu_hat_vec, b_hat_vec, c_hat_vec];

    if save_data
        file = ['../DATA/nu_', num2str(nu_true), '_', CASE, '_fromigPC.mat'];
        save(file, 'params_true', 'params_hat', 'xstats')
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
