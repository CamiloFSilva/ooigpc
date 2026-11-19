% =========================================================================
% OOID_CorrectionFactors_Validation.m
%   Validation of the two-level igPC correction on randomly sampled
%   (ν, κ) pairs at a fixed diffusion coefficient c.
%
% PURPOSE
%   For a large ensemble of randomly sampled (ν, κ) pairs drawn uniformly
%   from a physically relevant range, with the diffusion coefficient c held
%   fixed, this script integrates the analytical stationary PDF of the
%   Langevin amplitude equation, extracts its moments (μ, σ), applies igPC
%   identification without and with the two-level correction, and plots the
%   results.  No data files are written or read.
%
% PAPER FIGURE MAPPING
%   Fig. 1 (ν)  →  Supplementary Material, Fig. 1
%   Fig. 2 (κ)  →  Supplementary Material, Fig. 2
%     Red circles  : uncorrected igPC
%     Blue crosses : two-level corrected igPC
%     Black line   : ideal 1:1 reference
%
% DEPENDENCIES
%   igpc_identify (local function, bottom of this file)
%
% Author : C. F. Silva  (camilo.silva@tum.de)
%          Professorship of Thermofluiddynamics, TU Munich
% Paper  : Silva et al. (2026). Output-only identification of
%          thermoacoustic systems using polynomial chaos expansions.
%          Combust. Flame.
% =========================================================================

clear all

% =========================================================================
%% USER OPTIONS
% =========================================================================

KONST     = 2.6;        % calibrated prefactor for the Region-1 (low-std)
                        % correction:  δσ_R1 = KONST · x2_root

save_figs = true;       % true  → save figures as PDF to ../Plotting/
                        % false → display only, no files written

% =========================================================================
%% PARAMETER SPACE
% =========================================================================
% ν  : linear growth rate  [rad/s]
% κ  : nonlinear saturation coefficient
% c  : diffusion coefficient (fixed for this validation)

nu_min    = 0;    nu_max    = 30;
kappa_min = 0.1;  kappa_max = 8;

c         = 6;           % fixed diffusion coefficient

num_cases = 120;         % number of random (ν, κ) samples

% =========================================================================
%% COMPUTATION
% =========================================================================
% For each sampled pair (ν, κ) at fixed c:
%   Step 1 — Compute the analytical stationary PDF p(A) and its first two
%            central moments (μ_A, σ_A) and third central moment γ₃.
%   Step 2 — Apply igPC identification without correction (σ_eff = σ_A).
%   Step 3 — Apply the two-level analytical correction model and re-run
%            igPC with the corrected σ_eff.

% Draw random parameter samples
nu_vec    = nu_min    + (nu_max    - nu_min)    * rand(num_cases, 1);
kappa_vec = kappa_min + (kappa_max - kappa_min) * rand(num_cases, 1);

% Pre-allocate output arrays
nu_recov    = zeros(num_cases, 2);
kappa_recov = zeros(num_cases, 2);

for kc = 1:num_cases

    disp(['Case ' num2str(kc)])

    nu    = nu_vec(kc);
    kappa = kappa_vec(kc);

    % ------------------------------------------------------------------
    % Step 1: Analytical stationary PDF and its moments
    %
    % The Langevin amplitude equation dA/dt = νA − (κ/8)A³ + c/A + ζ(t)
    % has stationary PDF  p(A) ∝ exp(−Φ(A)/c), where:
    %   Φ(A) = −(ν/2)A² + (κ/32)A⁴ − c·ln(A)
    % ------------------------------------------------------------------
    A_grid = linspace(0, 20, 100);
    dA     = A_grid(2) - A_grid(1);

    Phi        = -(nu/2)*A_grid.^2 + (kappa/32)*A_grid.^4 - c*log(A_grid);
    pdf_unnorm = exp(-Phi / c);
    Z_norm     = trapz(pdf_unnorm) * dA;   % normalization constant
    pdf_A      = pdf_unnorm / Z_norm;       % normalized PDF

    % Central moments
    mu_A    = trapz(A_grid .* pdf_A) * dA;
    var_A   = trapz((A_grid - mu_A).^2 .* pdf_A) * dA;
    std_A   = sqrt(var_A);
    gamma_3 = trapz((A_grid - mu_A).^3 .* pdf_A) * dA;  % third central moment

    % ------------------------------------------------------------------
    % Step 2: igPC identification without correction  (baseline)
    %
    % Eqs. (10)-(11) of the paper: given (μ, σ, c), recover ν and κ.
    % σ_eff = σ_A (no correction applied).
    % ------------------------------------------------------------------
    [nu_hat_def, b_hat_def] = igpc_identify(mu_A, std_A, c);

    % ------------------------------------------------------------------
    % Step 3: Two-level analytical correction model
    %
    % Two regions separated by the critical line σ_A = 0.28·μ_A:
    %
    %   Region 2  (σ_A > 0.28·μ_A):
    %     Additive linear interpolation between two anchor lines:
    %       c1(σ) = 0.144·σ + 0.0126   (at σ ≈ 0.423·μ)
    %       c2(σ) = −0.035·σ            (at σ ≈ 0.28·μ)
    %     BETA  = (σ − 0.28·μ) / (0.423·μ − 0.28·μ)
    %     δσ    = (1−BETA)·c2 + BETA·c1
    %     σ_eff = σ_A + δσ
    %
    %   Region 1  (σ_A ≤ 0.28·μ_A):
    %     Cubic root of the skewness equation:
    %     4x³ − 6σ_A²·x + γ₃ = 0  →  real root x2_root
    %     δσ    = KONST · x2_root
    %     σ_eff = sqrt(σ_A² − 2·δσ²)
    % ------------------------------------------------------------------
    std_crit = 0.28 * mu_A;

    if std_A > std_crit
        % -- Region 2 --------------------------------------------------
        std_ref1    = 0.423 * mu_A;
        std_ref2    = 0.28  * mu_A;
        c1_std      = 0.144 * std_A + 0.0126;
        c2_std      = -0.035 * std_A;
        BETA        = (std_A - std_ref2) / (std_ref1 - std_ref2);
        delta_sigma = (1 - BETA)*c2_std + BETA*c1_std;
        sigma_eff   = std_A + delta_sigma;
    else
        % -- Region 1 --------------------------------------------------
        if isnan(gamma_3)
            sigma_eff = NaN;
        else
            x2_roots    = roots([4,  0,  -6*std_A^2,  gamma_3]);
            delta_sigma = KONST * real(x2_roots(3));
            sigma_eff   = sqrt(std_A^2 - 2*delta_sigma^2);
        end
    end

    [nu_hat_corr, b_hat_corr] = igpc_identify(mu_A, sigma_eff, c);

    % ------------------------------------------------------------------
    % Store results for this sample
    % ------------------------------------------------------------------
    nu_recov(kc, 1)    = nu_hat_def;        % uncorrected
    kappa_recov(kc, 1) = b_hat_def * 8;

    nu_recov(kc, 2)    = nu_hat_corr;       % two-level corrected
    kappa_recov(kc, 2) = b_hat_corr * 8;

end % kc loop

% =========================================================================
%% FIGURES 1-2: Validation scatter plots  →  Supplementary Material
%
% Each figure compares the true parameter value (x-axis) with the value
% recovered by igPC (y-axis):
%   Red circles  : uncorrected igPC identification
%   Blue crosses : two-level corrected igPC identification
%   Black line   : perfect 1:1 correspondence
%
% A tighter clustering around the 1:1 line for the corrected results
% confirms that the two-level correction reduces the identification error.
% =========================================================================

id_valid = find(~isnan(nu_recov(:, 2)));   % discard NaN samples (Region-1 edge cases)

figure(1); clf
plot(nu_vec(id_valid), nu_recov(id_valid, 1), 'ro', ...
    'LineWidth', 2, 'MarkerSize', 5)
hold on
plot(nu_vec(id_valid), nu_recov(id_valid, 2), 'bx', ...
    'LineWidth', 2, 'MarkerSize', 5)
plot([nu_min, nu_max], [nu_min, nu_max], 'k', 'LineWidth', 1.5)
axis([nu_min, nu_max, nu_min, nu_max])
xlabel('\nu_{true}');  ylabel('\nu_{recovered}')
legend({'Uncorrected', 'Corrected'}, 'Location', 'best')
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs
    saveas(gcf, '../Plotting/Validation_nu_Supp1.pdf')
end

figure(2); clf
plot(kappa_vec(id_valid), kappa_recov(id_valid, 1), 'ro', ...
    'LineWidth', 2, 'MarkerSize', 5)
hold on
plot(kappa_vec(id_valid), kappa_recov(id_valid, 2), 'bx', ...
    'LineWidth', 2, 'MarkerSize', 5)
plot([kappa_min, kappa_max], [kappa_min, kappa_max], 'k', 'LineWidth', 1.5)
axis([kappa_min, kappa_max, kappa_min, kappa_max])
xlabel('\kappa_{true}');  ylabel('\kappa_{recovered}')
legend({'Uncorrected', 'Corrected'}, 'Location', 'best')
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs
    saveas(gcf, '../Plotting/Validation_kappa_Supp2.pdf')
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

    A1     = 0.5*(9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
    B1     = 0.5*(3*sigma^2 - mu^2);
    D      = mu^2 * sigma^2 * (sigma^2 - mu^2);
    nu_hat = A1/D * c;
    b_hat  = B1/D * c;

end
