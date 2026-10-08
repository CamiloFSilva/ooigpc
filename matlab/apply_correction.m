function [sigma, std_correc, region, mu, std_raw] = apply_correction(A)
% APPLY_CORRECTION  Two-level non-Gaussian correction for the amplitude PDF.
%
%   [sigma, std_correc, region, mu, std_raw] = APPLY_CORRECTION(A)
%
%   The raw sample standard deviation std(A) slightly under-estimates the
%   corrected value σ that enters the identification formulae (Eqs. 10–11).
%   This function implements the two-level correction derived in the
%   Supplementary material §5 (Fig. 4 of the paper).
%
%   Two regimes are distinguished by the threshold σ_crit = 0.280·μ:
%
%   Region 1 — marginally stable  (std_raw > σ_crit)
%       Linear interpolation between two empirical correction curves:
%           c1 =  0.144·std_raw + 0.0126       (Eq. S-8)
%           c2 = −0.035·std_raw                (Eq. S-9)
%       with weight β = (std_raw − σ₂) / (σ₁ − σ₂),  σ₁=0.423μ, σ₂=0.280μ
%       Result:  σ = std_raw + std_correc
%
%   Region 2 — linearly unstable  (std_raw ≤ σ_crit)
%       Uses the skewness of the PDF to solve the cubic (Eq. S-7):
%           4δ³ − 6·std_raw²·δ + γ = 0,   γ = skewness·std_raw³
%       Takes the third real root (index 3, consistent with MATLAB's roots).
%       Result:  σ = √(std_raw² − 2·std_correc²)
%
% INPUT
%   A  – 1-D array of amplitude envelope samples, all positive
%
% OUTPUTS
%   sigma      – Corrected standard deviation entering Eqs. (10)–(11)
%   std_correc – Additive (or sign-aware) correction applied
%   region     – 1 (marginally stable) or 2 (linearly unstable)
%   mu         – Sample mean of A
%   std_raw    – Raw sample standard deviation (ddof=1) of A
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame.
%   Supplementary material §5, Eqs. S-7–S-9.

% -------------------------------------------------------------------------
% Empirical thresholds (Suppl. §5, Fig. 4)
% -------------------------------------------------------------------------
SIGMA_CRIT_RATIO = 0.280;   % σ_crit / μ
SIGMA_1_RATIO    = 0.423;   % σ₁    / μ   (upper boundary, Region 1)
SIGMA_2_RATIO    = 0.280;   % σ₂    / μ   (lower boundary, Region 1)

% -------------------------------------------------------------------------
% Basic statistics
% -------------------------------------------------------------------------
A        = A(:);
mu       = mean(A);
std_raw  = std(A, 0);   % ddof = 1 (default in MATLAB)

sigma_crit = SIGMA_CRIT_RATIO * mu;

% -------------------------------------------------------------------------
% Region 1 — marginally stable
% -------------------------------------------------------------------------
if std_raw > sigma_crit

    c1_std = 0.144 * std_raw + 0.0126;   % Eq. S-8
    c2_std = -0.035 * std_raw;            % Eq. S-9

    sigma_1 = SIGMA_1_RATIO * mu;
    sigma_2 = SIGMA_2_RATIO * mu;

    beta       = (std_raw - sigma_2) / (sigma_1 - sigma_2);
    std_correc = (1 - beta) * c2_std  +  beta * c1_std;

    sigma  = std_raw + std_correc;
    region = 1;

% -------------------------------------------------------------------------
% Region 2 — linearly unstable
% -------------------------------------------------------------------------
else

    sk    = skewness(A, 0);   % Fisher's definition, bias-corrected (flag=0)
    gamma = sk * std_raw^3;

    % Cubic:  4δ³ − 6·std_raw²·δ + γ = 0
    poly_coeffs = [4, 0, -6*std_raw^2, gamma];
    r = roots(poly_coeffs);

    % Take index 3 (third element from MATLAB's roots, sorted by
    % descending magnitude), consistent with the reference implementation.
    std_correc = 2.6 * real(r(3));

    var_corrected = std_raw^2 - 2 * std_correc^2;
    if var_corrected <= 0
        var_corrected = 1e-12;   % fallback clamp
    end
    sigma  = sqrt(var_corrected);
    region = 2;

end

end
