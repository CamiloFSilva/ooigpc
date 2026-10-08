function [A_hat, T_hat, MSD_fit] = fit_msd(taus, MSD, varargin)
% FIT_MSD  Fit the exponential MSD model and extract the diffusion coefficient c.
%
%   [A_hat, T_hat, MSD_fit] = FIT_MSD(taus, MSD)
%   [A_hat, T_hat, MSD_fit] = FIT_MSD(taus, MSD, 'P0', [A0, T0])
%
%   Fits the model (Supplementary §4, Eq. S-11):
%
%       MSD_mod(τ) = Â · (1 − exp(−τ/T̂))
%
%   to the empirical MSD using nonlinear least squares (lsqcurvefit).
%   The diffusion coefficient follows from Eq. 7:
%
%       c = Â / (2·T̂)
%
% INPUTS
%   taus  – Delay times τ (s),              vector of length n_tau
%   MSD   – Empirical MSD values,           vector of length n_tau
%
% OPTIONAL NAME–VALUE PAIRS
%   'P0'  – Initial guess [Â₀, T̂₀] (default: [0.1, 0.01])
%
% OUTPUTS
%   A_hat    – Fitted plateau amplitude Â
%   T_hat    – Fitted relaxation time T̂ (s)
%   MSD_fit  – Model MSD evaluated at taus
%
%   The diffusion coefficient is  c = A_hat / (2 * T_hat).
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame.  Eqs. 7, S-11.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'P0', [0.1, 0.01], @(x) isnumeric(x) && numel(x) == 2);
parse(p, varargin{:});
p0 = p.Results.P0;

% -------------------------------------------------------------------------
% Exponential MSD model
% -------------------------------------------------------------------------
msd_model = @(params, tau) params(1) .* (1 - exp(-tau ./ params(2)));

taus = taus(:);
MSD  = MSD(:);

% -------------------------------------------------------------------------
% Nonlinear least-squares fit
% -------------------------------------------------------------------------
opts = optimoptions('lsqcurvefit', ...
    'Display',       'off', ...
    'MaxIterations', 5000,  ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance',     1e-10);

lb = [0,  0  ];   % Â ≥ 0, T̂ ≥ 0
ub = [Inf, Inf];

popt = lsqcurvefit(msd_model, p0, taus, MSD, lb, ub, opts);

A_hat   = popt(1);
T_hat   = popt(2);
MSD_fit = msd_model(popt, taus);

end
