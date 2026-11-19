% OOID_igPC.m  (Experiments version)
% =========================================================================
% Output-only identification of experimental thermoacoustic data via igPC
%
% DESCRIPTION
%   Loads pre-computed amplitude envelopes produced by readExpData.m and
%   identifies the Langevin amplitude-equation parameters (ν, b, c) for
%   each operating condition using the igPC method.
%
%   The identification follows the same steps as the VdP/Langevin versions:
%     1. Estimate the diffusion coefficient  c  via Mean Square Displacement
%     2. Apply optional two-level σ correction (Regions 1 & 2)
%     3. Recover  ν  and  b  from closed-form igPC formulas (Eqs. 10-11)
%
%   Unlike the simulation scripts, there is no ensemble of realisations and
%   no ground-truth parameter.  Each operating condition (index ns = 1…5)
%   corresponds to a different equivalence ratio / heat-release rate.
%
%   An optional amplitude-binning step (BIN = 'YES') replaces the
%   continuous amplitude trace with a step function over discrete bins
%   before MSD computation.  This reduces noise at the cost of resolution;
%   set BIN = 'NOP' for the standard (published) workflow.
%
% INPUT
%   ../DATA/envelopeRef_B<bandwidth>Exp.mat   — produced by readExpData.m
%       Variables:  A        — {1×5} amplitude envelopes  [Pa]
%                   dt_vec   — [1×5] sampling time steps  [s]
%                   fn_vec   — [1×5] dominant frequencies [Hz]
%
% OUTPUT  (one file per operating condition)
%   ../DATA/ns_<ns>_<CASE>_<BIN>_fromigPC.mat
%       Variables:  params_hat — [1×3]  (nu_hat, b_hat, c_hat)
%                   ns         — scalar  operating-condition index
%                   wn         — scalar  angular frequency  [rad/s]
%                   x_stat     — amplitude time series used for statistics
%                   xstats     — [1×2]  (mu_A, std_A)
%
% DATA REFERENCE
%   Noiray, N. & Denisov, A. (2017).  Proc. Combust. Inst. 36, 3843-3850.
%
% METHOD REFERENCES
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
%   Boujo, E. & Noiray, N. (2017).  Proc. R. Soc. A 473, 20160894.
%
% SEE ALSO
%   readExpData.m               — generates the input envelope file
%   OOID_KM_PDFMatrix.m         — KM benchmark for the same data
%   OOID_KM_Direct.m            — KM direct method
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

bandwidth = 40;    % [Hz]  band-pass filter half-width; must match readExpData.m
%bandwidth = 10;   % alternative

ns_vec = 1:5;      % operating conditions to process (1 = lowest, 5 = highest ER)
%ns_vec = [3 4 5]; % process a subset only

% Identification mode
%   'NoCorrection'        — raw σ fed directly into igPC
%   'CorrectionTwoLevels' — two-level σ correction applied before igPC
%CASE = 'NoCorrection';
CASE = 'CorrectionTwoLevels';

% Amplitude binning before MSD (reduces noise, not used in published results)
%   'YES' — bin amplitude into nbin intervals
%   'NOP' — use the continuous amplitude trace (standard)
%BIN = 'YES';
BIN = 'NOP';

% Transient skip: ignore the first ind_trans samples when computing
% stationary statistics and MSD segments
ind_trans = 50;    % [samples]  (≈ first 50 / Fs seconds of data)

save_figs = true;  % true  → save MSD diagnostic figures to ../Plotting/
                   % false → display figures without saving
save_data = true;  % true  → save params_hat to ../DATA/
                   % false → run without writing .mat files

% =========================================================================
%% KM ESTIMATION PARAMETERS
% =========================================================================

% Time-lag vector for MSD estimation  [s]
tau_val = 0.0005 : 0.0005 : 0.2;
%tau_val = 0.005  : 0.005  : 0.16;   % coarser alternative
%tau_val = 0.03   : 0.02   : 0.19;   % coarsest alternative

% =========================================================================
%% LOAD ENVELOPE DATA
% =========================================================================

filename = ['../DATA/envelope_BW', num2str(bandwidth), 'Exp.mat'];
load(filename, 'A', 'dt_vec', 'fn_vec')

% =========================================================================
%% IDENTIFICATION LOOP (over operating conditions)
% =========================================================================

for nn = 1:length(ns_vec)

    ns = ns_vec(nn);
    fprintf('\n=== Operating condition %d / %d ===\n', ns, max(ns_vec));

    % ------------------------------------------------------------------
    % 1. Extract envelope data for this condition
    % ------------------------------------------------------------------
    % IMPORTANT: rename the envelope to  ampl  immediately so the name  A
    % remains free for the MSD fitter  (fit() returns a struct with field
    % MSD_sat, not A — but renaming avoids any future ambiguity).

    ampl   = A{ns};          % [Pa]  amplitude envelope time series
    dt_sim = dt_vec(ns);     % [s]   sampling time step
    fn     = fn_vec(ns);     % [Hz]  dominant frequency
    omega  = 2*pi*fn;        % [rad/s]  angular frequency
    N      = length(ampl);

    % ------------------------------------------------------------------
    % 2. Stationary statistics  (skip initial transient)
    % ------------------------------------------------------------------

    x_stat = ampl(ind_trans : end);
    mu_A   = mean(x_stat);
    std_A  = std(x_stat);

    fprintf('  μ_A  = %.4g  Pa\n', mu_A);
    fprintf('  σ_A  = %.4g  Pa\n', std_A);

    % ------------------------------------------------------------------
    % 3. Mean Square Displacement (MSD) estimation of  c
    %    MSD(τ) = E[(A(t+τ) − A(t))²]
    %    For the Langevin amplitude process:
    %      MSD(τ) ≈ MSD_amp · [1 − exp(−τ/τ_drift)],
    %      c = MSD_amp / (2·τ_drift)
    % ------------------------------------------------------------------

    tk_vec_msd = floor(tau_val / dt_sim);    % lag indices

    for kk = 1:length(tk_vec_msd)
        tk     = tk_vec_msd(kk);
        dt_lag = tk * dt_sim;                % [s] physical time lag

        xvec1 = ampl(ind_trans : N-tk)';    % reference segment
        xvec2 = ampl(ind_trans+tk : N)';    % delayed segment

        % --- Optional amplitude binning (BIN = 'YES') ---
        if strcmp(BIN, 'YES')
            nbin  = 30;
            Amin  = min(ampl);
            Amax  = max(ampl);
            dA    = (Amax - Amin)/nbin;
            edges = [Amin, linspace(Amin+dA, Amax, nbin)];

            idx    = discretize(ampl, edges);
            Afloor = edges(idx);

            xvec1 = Afloor(ind_trans : N-tk);
            xvec2 = Afloor(ind_trans+tk : N);
        end

        deltax   = xvec1 - xvec2;
        MSD(kk)  = mean(deltax.^2);
        dMSD(kk) = MSD(kk) / dt_lag;
    end

    % ------------------------------------------------------------------
    % 4. Fit exponential MSD model → extract  c_hat
    % ------------------------------------------------------------------

    taus = tk_vec_msd * dt_sim;   % physical time lags [s]

    ft = fittype('MSD_sat * (1 - exp(-taus/T_drift))', ...
        'independent', 'taus', ...
        'coefficients', {'MSD_sat', 'T_drift'});
    f = fit(taus(:), MSD(:), ft, 'StartPoint', [0.1, 0.01]);

    MSD_amp   = f.MSD_sat;    % [Pa²]  MSD saturation amplitude
    tau_drift = f.T_drift;    % [s]    exponential time constant
    c_hat     = MSD_amp / 2 / tau_drift;   % [Pa²·s⁻¹] diffusion coefficient

    MSDfit  = MSD_amp * (1 - exp(-taus / tau_drift));
    dMSDfit = MSDfit ./ taus;
    dMSD    = MSD    ./ taus;    % re-compute for plotting

    fprintf('  MSD_amp  = %.4g  Pa²\n',   MSD_amp);
    fprintf('  τ_drift  = %.4g  s\n',     tau_drift);
    fprintf('  c_hat    = %.4g  Pa²/s\n', c_hat);

    % ------------------------------------------------------------------
    % 5. MSD diagnostic figures
    % ------------------------------------------------------------------

    if save_figs

        % Figure 20 — MSD(τ) with exponential fit
        figure(20); clf
        plot(taus, MSD,    'rx', 'MarkerSize', 4, 'LineWidth', 1); hold on
        plot(taus, MSDfit, 'b-', 'MarkerSize', 8, 'LineWidth', 2)
        xlabel('\tau  [s]');  ylabel('MSD  [Pa^2]')
        title(['MSD fit — operating condition ', num2str(ns)])
        grid on
        ax = gca;  ax.YGrid = 'on';  ax.YMinorGrid = 'on';  ax.XGrid = 'off';
        ax.XAxis.Exponent = 0;
        set(gca, 'FontSize', 20)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        saveas(gcf, ['../Plotting/MSD_Exp_ns', num2str(ns), '.pdf'])

        % Figure 21 — D₂(τ) = dMSD/2  (extrapolates to c at τ→0)
        figure(21); clf
        plot(taus, dMSD/2,    'rx', 'MarkerSize', 4, 'LineWidth', 1); hold on
        plot(taus, dMSDfit/2, 'b-', 'MarkerSize', 8, 'LineWidth', 2)
        xlabel('\tau  [s]');  ylabel('D_2(\tau)  [Pa^2 s^{-1}]')
        title(['Diffusion coefficient extrapolation — ns = ', num2str(ns)])
        grid on
        ax = gca;  ax.YGrid = 'on';  ax.YMinorGrid = 'on';  ax.XGrid = 'off';
        ax.XAxis.Exponent = 0;
        set(gca, 'FontSize', 20)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        saveas(gcf, ['../Plotting/Extrapolation_Exp_ns', num2str(ns), '.pdf'])

    end

    % ------------------------------------------------------------------
    % 6. Optional correction to σ before igPC identification
    %    The two-level correction compensates for the bias that the PC
    %    variance formula  Var[A] = x₁² + 2x₂²  introduces in σ.
    %    See OOID_CorrectionFactors_V5.m for derivation.
    % ------------------------------------------------------------------

    switch CASE

        case 'NoCorrection'
            delta_sigma = 0;
            sigma_eff   = std_A;

        case 'CorrectionTwoLevels'
            std_crit = 0.28 * mu_A;    % boundary between Region 1 and 2

            if std_A > std_crit
                % --- Region 2  (σ > 0.28·μ, high-noise regime) ---
                c1_std = 0.144 * std_A + 0.0126;   % upper boundary correction
                c2_std = -0.035 * std_A;            % lower boundary correction

                std_hi = 0.423 * mu_A;
                std_lo = 0.280 * mu_A;   % = std_crit

                beta_interp = (std_A - std_lo) / (std_hi - std_lo);
                delta_sigma = (1 - beta_interp)*c2_std + beta_interp*c1_std;
                sigma_eff   = std_A + delta_sigma;
                disp('  Correction region: 2 (high noise)')

            else
                % --- Region 1  (σ ≤ 0.28·μ, low-noise regime) ---
                %   Solve: 4x³ − 6σ²x + γ₃ = 0,  γ₃ = skew·σ³
                %   Then: δσ = 2.6 · Re(x₂)
                skew_A      = skewness(x_stat);
                gamma_skew  = skew_A * std_A^3;

                x2_roots    = roots([4,  0,  -6*std_A^2,  gamma_skew]);
                delta_sigma = 2.6 * real(x2_roots(3));
                sigma_eff   = sqrt(std_A^2 - 2*delta_sigma^2);
                disp('  Correction region: 1 (low noise)')
            end

    end   % switch CASE

    % ------------------------------------------------------------------
    % 7. igPC identification  (Eqs. 10-11, Silva et al. 2026)
    % ------------------------------------------------------------------

    [nu_hat, b_hat] = igpc_identify(mu_A, sigma_eff, c_hat);

    kappa_hat = 8 * b_hat;
    Gamma_hat = c_hat * 4 * omega^2;

    fprintf('  ν    identified: %8.4f  [rad/s]\n',   nu_hat);
    fprintf('  κ    identified: %8.4f  [m⁻²s⁻¹]\n', kappa_hat);
    fprintf('  c    identified: %8.3g  [Pa²/s]\n',   c_hat);
    fprintf('  Γ    identified: %8.3g  [Pa²/s³]\n',  Gamma_hat);

    params_hat = [nu_hat, b_hat, c_hat];
    xstats     = [mu_A, std_A];

    % ------------------------------------------------------------------
    % 8. Save results
    % ------------------------------------------------------------------

    if save_data
        file = ['../DATA/ns_', num2str(ns), '_', CASE, '_', BIN, '_fromigPC.mat'];
        save(file, 'params_hat', 'ns', 'omega', 'x_stat', 'xstats')
        fprintf('  Saved to: %s\n', file);
    end

end   % ns loop

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function [nu_hat, b_hat] = igpc_identify(mu, sigma, c)
% igpc_identify  Closed-form igPC identification of the Langevin parameters
%
%   [nu_hat, b_hat] = igpc_identify(mu, sigma, c)
%
%   INPUT
%     mu     — mean amplitude  E[A]  [Pa]
%     sigma  — (corrected) standard deviation  std[A]  [Pa]
%     c      — diffusion coefficient  c = Γ/(4ω₀²)  [Pa²·s⁻¹]
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

    A1     = 0.5*(9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
    B1     = 0.5*(3*sigma^2 - mu^2);
    D      = mu^2 * sigma^2 * (sigma^2 - mu^2);
    nu_hat = (A1/D) * c;
    b_hat  = (B1/D) * c;
end
