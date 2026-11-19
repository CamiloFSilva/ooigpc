% OOID_CorrectionFactors.m
% =========================================================================
% igPC identification accuracy maps and correction-factor validation
%
% PURPOSE
%   For a large ensemble of randomly sampled (ν, κ, c) triples drawn from
%   a uniform distribution over a physically relevant range, this script:
%     (a) [MODE = 'Calculate'] integrates the analytical stationary PDF of
%         the Langevin amplitude equation, extracts its moments, runs the
%         igPC identification (Eqs. 10-11 of the paper), applies the
%         two-level correction, and saves all results to a .mat file; or
%     (b) [MODE = 'Figures'] loads a previously computed dataset and
%         generates the scatter-map figures.
%
% PAPER FIGURE MAPPING
%   Fig. 1 + 2  →  paper Fig. 3(b)   reference (ν/c) and (κ/c) maps
%   Fig. 3 + 4  →  paper Fig. 3(c)   identification error maps  (id = 1, no correction)
%   Fig. 3 + 4  →  paper Fig. 3(d)   identification error maps  (id = 2, with correction)
%   Fig. 5      →  paper Fig. 4(a)   reference Region-2 correction map
%   Fig. 6      →  paper Fig. 4(b)   modelled Region-2 correction map
%   Fig. 9      →  paper Fig. 4(c)   Region-2 correction model error
%   Fig. 7      →  paper Fig. 5(a)   reference Region-1 correction map
%   Fig. 8      →  paper Fig. 5(b)   two-level model correction map
%
%   Silva, C. F. (2026).
%   "Output-only identification of thermoacoustic systems using polynomial
%    chaos expansions." Combustion and Flame.
% =========================================================================

clear all

% =========================================================================
%% USER OPTIONS
% =========================================================================

MODE      = 'Figures';  % 'Calculate' — run computation and save results
                        % 'Figures'   — load saved results and plot

id        = 2;          % selects which identification result to use in
                        %   the error maps (Figs. 3-4):
                        %   id = 1 → uncorrected igPC  (paper Fig. 3c)
                        %   id = 2 → two-level corrected igPC (paper Fig. 3d)

KONST     = 2.6;        % calibrated prefactor for the Region-1 (low-std)
                        % correction:  δσ_R1 = KONST · x2_root
                        % (also used in Fig. 6 and Fig. 9)

save_figs = true;       % true  → save all figures as PDF to ../Plotting/
                        % false → display only, no files written

% =========================================================================
%% PARAMETER SPACE  (used only in MODE = 'Calculate')
% =========================================================================
% (ν, κ, c) are drawn uniformly from the ranges below.
% ν  : linear growth rate  [rad/s]
% κ  : nonlinear saturation coefficient
% c  : diffusion coefficient of the amplitude envelope

nu_min    = 0;    nu_max    = 20;
kappa_min = 0.1;  kappa_max = 6;
c_min     = 0.5;  c_max     = 6;

num_cases = 100000;   % number of random (ν, κ, c) samples

% =========================================================================
%% MAIN LOOP
% =========================================================================
% The outer loop over nk allows processing multiple independent datasets
% (DATA1.mat, DATA2.mat, ...).  Currently only nk = 1 is active.

for nk = 1:1

    % Draw random parameter samples (new draw each outer iteration)
    nu_vec    = nu_min    + (nu_max    - nu_min)    * rand(num_cases, 1);
    kappa_vec = kappa_min + (kappa_max - kappa_min) * rand(num_cases, 1);
    c_vec     = c_min     + (c_max     - c_min)     * rand(num_cases, 1);

    switch MODE

        % =================================================================
        case 'Calculate'
        % =================================================================
        % For each sampled triple (ν, κ, c):
        %   Step 1 — Compute the analytical stationary PDF p(A) and its
        %            first four central moments (μ_A, σ_A, skew_A, kurt_A).
        %   Step 2 — Apply igPC identification without correction (σ_eff = σ_A).
        %   Step 3 — Grid-search for the optimal additive correction in each
        %            region (reference values stored in corr_R1_vec, corr_R2_vec).
        %   Step 4 — Apply the analytical two-level correction model and
        %            re-run igPC with the corrected σ_eff.
        % =================================================================

            for kc = 1:num_cases

                disp(['Case ' num2str(kc)])

                c     = c_vec(kc);
                nu    = nu_vec(kc);
                kappa = kappa_vec(kc);

                % ----------------------------------------------------------
                % Step 1: Analytical stationary PDF and its moments
                % ----------------------------------------------------------
                % The Langevin amplitude equation dA/dt = νA − (κ/8)A³ + c/A + ζ(t)
                % has stationary PDF  p(A) ∝ exp(−Φ(A)/c), where:
                %   Φ(A) = −(ν/2)A² + (κ/32)A⁴ − c·ln(A)
                % (see e.g. Noiray & Schuermans 2013)

                A_grid = linspace(0, 20, 100);
                dA     = A_grid(2) - A_grid(1);

                Phi         = -(nu/2)*A_grid.^2 + (kappa/32)*A_grid.^4 - c*log(A_grid);
                pdf_unnorm  = exp(-Phi / c);
                Z_norm      = trapz(pdf_unnorm) * dA;   % normalization constant
                pdf_A       = pdf_unnorm / Z_norm;       % normalized PDF

                % Central moments of p(A)
                mu_A   = trapz(A_grid .* pdf_A) * dA;
                var_A  = trapz((A_grid - mu_A).^2 .* pdf_A) * dA;
                std_A  = sqrt(var_A);
                skew_A = trapz((A_grid - mu_A).^3 .* pdf_A) * dA / std_A^3;
                kurt_A = trapz((A_grid - mu_A).^4 .* pdf_A) * dA / std_A^4;

                % ----------------------------------------------------------
                % Step 2: igPC identification without correction  (baseline)
                %
                % Eqs. (10)-(11) of the paper: given (μ, σ, c), recover ν and κ.
                % σ_eff = σ_A (no correction applied).
                % ----------------------------------------------------------
                [nu_hat_def, b_hat_def] = igpc_identify(mu_A, std_A, c);

                % ----------------------------------------------------------
                % Step 3: Grid search for optimal additive corrections
                %
                % For each candidate correction value δ in vec_scan, we ask:
                % "what δ, when added to σ_A (Region 1) or subtracted
                %  quadratically (Region 2), makes igPC recover the true ν
                %  exactly?"  The minimising δ is stored as the reference
                %  correction.
                %
                %   Region 2 (large std): σ_eff = sqrt(σ_A² − 2δ²)
                %   Region 1 (small std): σ_eff = σ_A + δ
                % ----------------------------------------------------------

                % Roots of the skewness cubic — computed once (independent of δ).
                % 4x³ − 6σ_A²x + γ = 0, where γ = skew_A · σ_A³.
                % The third root is the real solution used in the correction.
                gamma_skew = skew_A * std_A^3;
                if isnan(gamma_skew)
                    x2_roots = [NaN; NaN; NaN];
                    x1_roots = [NaN; NaN; NaN];
                else
                    x2_roots = roots([4,  0,  -6*std_A^2,  gamma_skew]);
                    x1_roots = real(sqrt(std_A^2 - 2*x2_roots.^2));
                end

                vec_scan  = -0.5 : 0.005 : 0.2;   % candidate δ values
                tol_R1    = 1;                      % running minimum error (Region 1)
                tol_R2    = 1;                      % running minimum error (Region 2)

                for v = 1:length(vec_scan)

                    delta_scan = vec_scan(v);

                    % -- Region 2: σ_eff = sqrt(σ_A² − 2δ²) ---------------
                    sigma_eff_R2 = sqrt(std_A^2 - 2*delta_scan^2);
                    [nu_hat_R2, ~] = igpc_identify(mu_A, sigma_eff_R2, c);
                    err_R2 = nu_hat_R2 - nu;

                    if abs(err_R2) < tol_R2
                        tol_R2              = abs(err_R2);
                        corr_R2_vec(kc)     = delta_scan;
                    end

                    % -- Region 1: σ_eff = σ_A + δ -------------------------
                    sigma_eff_R1 = std_A + delta_scan;
                    [nu_hat_R1, ~] = igpc_identify(mu_A, sigma_eff_R1, c);
                    err_R1 = nu_hat_R1 - nu;

                    if abs(err_R1) < tol_R1
                        tol_R1              = abs(err_R1);
                        corr_R1_vec(kc)     = delta_scan;
                    end

                end

                % ----------------------------------------------------------
                % Step 4: Two-level analytical correction model
                %
                % Two regions separated by the critical line σ_A = 0.28·μ_A:
                %
                %   Region 2  (σ_A > 0.28·μ_A):
                %     Additive linear interpolation between two anchor lines:
                %       c1(σ) = 0.144·σ + 0.0126   (at σ = 0.423·μ)
                %       c2(σ) = −0.035·σ            (at σ = 0.28·μ)
                %     BETA  = (σ − 0.28·μ) / (0.423·μ − 0.28·μ)
                %     δσ    = (1−BETA)·c2 + BETA·c1
                %     σ_eff = σ_A + δσ
                %
                %   Region 1  (σ_A ≤ 0.28·μ_A):
                %     Correction based on the cubic root x2_root:
                %     δσ    = KONST · x2_root   (KONST = 2.6, see USER OPTIONS)
                %     σ_eff = sqrt(σ_A² − 2·δσ²)
                % ----------------------------------------------------------

                std_crit = 0.28 * mu_A;   % boundary between Region 1 and Region 2

                if std_A > std_crit
                    % -- Region 2 ------------------------------------------
                    std_ref1   = 0.423 * mu_A;
                    std_ref2   = 0.28  * mu_A;
                    c1_std     = 0.144 * std_A + 0.0126;
                    c2_std     = -0.035 * std_A;
                    BETA       = (std_A - std_ref2) / (std_ref1 - std_ref2);
                    delta_sigma = (1 - BETA)*c2_std + BETA*c1_std;
                    sigma_eff   = std_A + delta_sigma;
                else
                    % -- Region 1 ------------------------------------------
                    delta_sigma = KONST * real(x2_roots(3));
                    sigma_eff   = sqrt(std_A^2 - 2*delta_sigma^2);
                end

                [nu_hat_corr, b_hat_corr] = igpc_identify(mu_A, sigma_eff, c);

                % ----------------------------------------------------------
                % Store results for this sample
                % ----------------------------------------------------------
                mu_vec(kc)   = mu_A;
                std_vec(kc)  = std_A;
                skew_vec(kc) = skew_A;
                kurt_vec(kc) = kurt_A;

                x1_vec(kc)   = real(x1_roots(3));
                x2_vec(kc)   = real(x2_roots(3));
                corr2(kc)    = delta_sigma;      % two-level model correction

                % Column 1: uncorrected igPC result
                nu_recov(kc,1)    = nu_hat_def;
                kappa_recov(kc,1) = b_hat_def * 8;

                % Column 2: corrected igPC result
                nu_recov(kc,2)    = nu_hat_corr;
                kappa_recov(kc,2) = b_hat_corr * 8;

                cparam(kc)   = c;

            end % kc loop

            save(['../DATA_2DMap/DATA', num2str(nk), '.mat'], ...
                 'nu_vec', 'kappa_vec', 'c_vec', ...
                 'mu_vec', 'std_vec', ...
                 'nu_recov', 'kappa_recov', 'cparam', ...
                 'x1_vec', 'x2_vec', ...
                 'corr_R1_vec', 'corr_R2_vec', 'corr2')

        % =================================================================
        case 'Figures'
        % =================================================================

            load(['../DATA_2DMap/DATA', num2str(nk), '.mat'], ...
                 'nu_vec', 'kappa_vec', 'c_vec', ...
                 'mu_vec', 'std_vec', ...
                 'nu_recov', 'kappa_recov', 'cparam', ...
                 'x1_vec', 'x2_vec', ...
                 'corr_R1_vec', 'corr_R2_vec', 'corr2')

    end % switch MODE

end % nk loop

% =========================================================================
%% COLORMAP AND SHARED PLOT SETTINGS
% =========================================================================

% Diverging blue-white-red colormap for error plots
nn = 256;
c_lo  = [linspace(0,1,nn/2)',  linspace(0,1,nn/2)',  ones(nn/2,1)];   % blue→white
c_hi  = [ones(nn/2,1),  linspace(1,0,nn/2)',  linspace(1,0,nn/2)'];   % white→red
cmap_diverge = [c_lo; c_hi];

% Reference boundary lines σ = 0.423·μ and σ = 0.28·μ
mu_draw    = 0 : 0.1 : 10;
sigma_hi   = 0.423 * mu_draw;   % upper boundary of Region 2
sigma_lo   = 0.280 * mu_draw;   % lower boundary / Region 1 threshold

% =========================================================================
%% FIGURES 1-2: Reference parameter maps  →  paper Fig. 3(b)
%
% Scatter plot of the (μ_A, σ_A) space, coloured by the true ν/c and κ/c.
% These are the "ground truth" maps against which identification errors
% are assessed.
% =========================================================================

figure(1)   % reference ν/c
scatter(mu_vec(:), std_vec(:), 20,  nu_vec(:)./cparam(:), 'filled')
axis([0 10 0 1]);  caxis([0 10]);  colormap(turbo)
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, '../Plotting/map_nu_ref_c.pdf');  end

figure(2)   % reference κ/c
scatter(mu_vec(:), std_vec(:), 20,  kappa_vec(:)./cparam(:), 'filled')
axis([0 10 0 1]);  caxis([0 10]);  colormap(turbo)
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, '../Plotting/map_kappa_ref_c.pdf');  end

% =========================================================================
%% FIGURES 3-4: Identification error maps  →  paper Fig. 3(c) or 3(d)
%
% Error = (true parameter / c) − (recovered parameter / c).
% Set id = 1 for uncorrected igPC (Fig. 3c).
% Set id = 2 for two-level corrected igPC (Fig. 3d).
% =========================================================================

Error_nu    = nu_vec(:)./cparam(:)    - nu_recov(:,id)./cparam(:);
Error_kappa = kappa_vec(:)./cparam(:) - kappa_recov(:,id)./cparam(:);

figure(3)   % error in ν/c
scatter(mu_vec(:), std_vec(:), 20,  Error_nu, 'filled')
axis([0 10 0 1]);  caxis([-1 1]);  colormap(cmap_diverge)
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, ['../Plotting/map_Error', num2str(id), '_nu.pdf']);  end

figure(4)   % error in κ/c
scatter(mu_vec(:), std_vec(:), 20,  Error_kappa, 'filled')
axis([0 10 0 1]);  caxis([-1 1]);  colormap(cmap_diverge)
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, ['../Plotting/map_Error', num2str(id), '_kappa.pdf']);  end

% =========================================================================
%% FIGURES 5-6: Region-2 correction maps  →  paper Fig. 4(a) and 4(b)
%
% Region 2 (σ > σ_lo): additive correction to σ.
%   Fig. 5 (reference):  corr_R2_vec — optimal δσ found by grid search.
%   Fig. 6 (model):      KONST · x2_root — analytical approximation of δσ.
%
% Both figures share the same colour axis for direct comparison.
% The two boundary lines σ = 0.423·μ and σ = 0.28·μ are overlaid.
% =========================================================================

figure(5)   % reference correction (Region 2) — paper Fig. 4(a)
scatter(mu_vec(:), std_vec(:), 20,  corr_R2_vec(:), 'filled')
axis([0 10 0 2]);  caxis([-0.5 0.12]);  colormap(turbo)
hold on
plot(mu_draw, sigma_hi, 'k', 'LineWidth', 3)
plot(mu_draw, sigma_lo, 'k', 'LineWidth', 3)
set(gca, 'FontSize', 22)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 7], 'PaperPosition', [0 0 20 7])
if save_figs;  saveas(gcf, '../Plotting/map_CorrFactor_ref.pdf');  end

figure(6)   % modelled correction (Region 2) — paper Fig. 4(b)
scatter(mu_vec(:), std_vec(:), 20,  KONST * x2_vec(:), 'filled')
axis([0 10 0 2]);  caxis([-0.5 0.12]);  colormap(turbo)
hold on
plot(mu_draw, sigma_hi, 'k', 'LineWidth', 3)
plot(mu_draw, sigma_lo, 'k', 'LineWidth', 3)
set(gca, 'FontSize', 22)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 7], 'PaperPosition', [0 0 20 7])
if save_figs
    saveas(gcf, ['../Plotting/map_CorrFactor_model_Konst_', num2str(KONST), '.pdf'])
end

% =========================================================================
%% FIGURES 7-8: Region-1 correction maps  →  paper Fig. 5(a) and 5(b)
%
% Region 1 (σ ≤ σ_lo): correction based on cubic roots of the skewness.
%   Fig. 7 (reference):  corr_R1_vec — optimal δσ found by grid search.
%   Fig. 8 (model):      corr2       — two-level model prediction of δσ.
% =========================================================================

figure(7)   % reference correction (Region 1) — paper Fig. 5(a)
scatter(mu_vec(:), std_vec(:), 20,  corr_R1_vec(:), 'filled')
axis([0 10 0 2]);  caxis([-0.3 0.3]);  colormap(turbo);  colorbar
hold on
plot(mu_draw, sigma_hi, 'k', 'LineWidth', 3)
plot(mu_draw, sigma_lo, 'k', 'LineWidth', 3)
set(gca, 'FontSize', 22)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 7], 'PaperPosition', [0 0 20 7])
if save_figs;  saveas(gcf, '../Plotting/map_CorrFactor2_ref.pdf');  end

figure(8)   % two-level model correction — paper Fig. 5(b)
scatter(mu_vec(:), std_vec(:), 20,  corr2(:), 'filled')
axis([0 10 0 2]);  caxis([-0.3 0.3]);  colormap(turbo);  colorbar
hold on
plot(mu_draw, sigma_hi, 'k', 'LineWidth', 3)
plot(mu_draw, sigma_lo, 'k', 'LineWidth', 3)
set(gca, 'FontSize', 22)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 7], 'PaperPosition', [0 0 20 7])
if save_figs;  saveas(gcf, '../Plotting/map_CorrFactor_model.pdf');  end

% =========================================================================
%% FIGURE 9: Region-2 correction model error  →  paper Fig. 4(c)
%
% Squared pointwise error between the reference correction (corr_R2_vec)
% and the analytical model (KONST · x2_vec).
% A small error across the whole (μ, σ) space validates KONST = 2.6.
% =========================================================================

Err_corr_R2 = (corr_R2_vec - KONST * x2_vec).^2;

figure(9)
scatter(mu_vec(:), std_vec(:), 20,  Err_corr_R2(:), 'filled')
axis([0 10 0 2]);  caxis([0 0.001]);  colormap(cmap_diverge)
hold on
plot(mu_draw, sigma_hi, 'k', 'LineWidth', 3)
plot(mu_draw, sigma_lo, 'k', 'LineWidth', 3)
set(gca, 'FontSize', 22)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 7], 'PaperPosition', [0 0 20 7])
if save_figs
    saveas(gcf, ['../Plotting/map_ErrorCorrFactor_', num2str(KONST), '.pdf'])
end

% =========================================================================
%% LOCAL FUNCTION
% =========================================================================

function [nu_hat, b_hat] = igpc_identify(mu, sigma, c)
% igpc_identify  Apply the igPC closed-form identification (Eqs. 10-11).
%
%   Given the mean (mu) and standard deviation (sigma) of the amplitude
%   envelope PDF, and the diffusion coefficient c, recover the linear
%   growth rate nu and the saturation parameter b = kappa/8.
%
%   Input:
%     mu    — mean of the amplitude PDF
%     sigma — standard deviation of the amplitude PDF (possibly corrected)
%     c     — diffusion coefficient of the Langevin amplitude equation
%
%   Output:
%     nu_hat — recovered linear growth rate  ν̂
%     b_hat  — recovered saturation parameter  b̂ = κ̂/8
%
%   Formulae (Silva et al. 2026, Eqs. 10-11):
%     A1 = ½(9σ⁴ + 2μ²σ² − μ⁴)
%     D  = μ²σ²(σ² − μ²)        [common denominator]
%     B1 = (3σ² − μ²)/2
%     ν̂  = (A1/D)·c
%     b̂  = (B1/D)·c

    A1    = 0.5*(9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
    B1    = 0.5*(3*sigma^2 - mu^2);
    D     = mu^2 * sigma^2 * (sigma^2 - mu^2);
    nu_hat = A1/D * c;
    b_hat  = B1/D * c;

end
