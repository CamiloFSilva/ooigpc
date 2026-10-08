function result = identify_igpc(A, c, varargin)
% IDENTIFY_IGPC  Output-only identification via iterative gPC (igPC).
%
%   result = IDENTIFY_IGPC(A, c)
%   result = IDENTIFY_IGPC(A, c, 'Omega', omega)
%
%   Given the amplitude envelope A(t) and the diffusion coefficient c
%   (estimated from the MSD, see FIT_MSD), recovers the three physical
%   parameters of the Langevin SDE:
%
%       dA = (ν·A − (κ/8)·A³ + c/A) dt + √(2c) dW         (Eq. 1)
%
%   using the explicit identification formulae:
%
%       ν   = [9σ⁴ + 2μ²σ² − μ⁴] / [2μ²σ²(σ²−μ²)] · c   (Eq. 10)
%       κ/8 = [3σ² − μ²]          / [2μ²σ²(σ²−μ²)] · c   (Eq. 11)
%
%   where μ and σ come from the corrected PDF (APPLY_CORRECTION), and
%       Γ = c · 4ω₀²                                       (Eq. 12)
%
% INPUTS
%   A  – 1-D amplitude envelope time series (all values positive)
%   c  – Diffusion coefficient  (from  A_hat / (2·T_hat),  see FIT_MSD)
%
% OPTIONAL NAME–VALUE PAIRS
%   'Omega'  – Angular eigenfrequency ω₀ = 2π·f₀ (rad/s).
%              Required to compute Γ.  Omit or pass [] to skip.
%
% OUTPUT
%   result – struct with fields:
%       .nu        – Linear growth rate ν (s⁻¹)
%       .kappa     – Nonlinearity coefficient κ
%       .Gamma     – Noise intensity Γ (Pa²·s⁻³)  [NaN if omega not given]
%       .a         – a = ν
%       .b         – b = κ/8
%       .c         – Diffusion coefficient
%       .mu        – Sample mean of A
%       .sigma     – Corrected standard deviation
%       .std_raw   – Raw standard deviation
%       .region    – Correction region (1 or 2)
%
% EXAMPLE
%   omega = 2*pi*120;
%   A = simulate_langevin(16, 3.4, 1e7, omega, 2e6, 5e-5, 'Seed', 0);
%   tau = (2.5e-4 : 2.5e-4 : 0.2)';
%   [taus, MSD] = compute_msd(A, tau, 5e-5);
%   [A_hat, T_hat] = fit_msd(taus, MSD);
%   c_est = A_hat / (2 * T_hat);
%   res = identify_igpc(A(5001:end), c_est, 'Omega', omega);
%   fprintf('nu = %.1f s^{-1}\n', res.nu);
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame. Eqs. 10–12.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'Omega', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
parse(p, varargin{:});
omega = p.Results.Omega;

% -------------------------------------------------------------------------
% Two-level non-Gaussian correction (Suppl. §5)
% -------------------------------------------------------------------------
[sigma, std_correc, region, mu, std_raw] = apply_correction(A);

% -------------------------------------------------------------------------
% Identification formulae (Eqs. 10–11)
% -------------------------------------------------------------------------
mu2    = mu^2;
sigma2 = sigma^2;
denom  = mu2 * sigma2 * (sigma2 - mu2);

if abs(denom) < 1e-30
    error('identify_igpc:singularDenominator', ...
        'Denominator mu^2*sigma^2*(sigma^2-mu^2) ≈ 0 (mu=%.4g, sigma=%.4g). Cannot identify.', ...
        mu, sigma);
end

% Eq. (10): a = ν
A1 = 0.5 * (9*sigma2^2 + 2*mu2*sigma2 - mu2^2);
a  = (A1 / denom) * c;

% Eq. (11): b = κ/8
B1 = 0.5 * (3*sigma2 - mu2);
b  = (B1 / denom) * c;

nu    = a;
kappa = 8 * b;

if ~isempty(omega)
    Gamma = c * 4 * omega^2;
else
    Gamma = NaN;
end

% -------------------------------------------------------------------------
% Pack result struct
% -------------------------------------------------------------------------
result.nu       = nu;
result.kappa    = kappa;
result.Gamma    = Gamma;
result.a        = a;
result.b        = b;
result.c        = c;
result.mu       = mu;
result.sigma    = sigma;
result.std_raw  = std_raw;
result.region   = region;

end
