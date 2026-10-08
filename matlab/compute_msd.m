function [taus, MSD] = compute_msd(A, tau_values, dt_sim, varargin)
% COMPUTE_MSD  Mean-square displacement of the amplitude envelope.
%
%   [taus, MSD] = COMPUTE_MSD(A, tau_values, dt_sim)
%   [taus, MSD] = COMPUTE_MSD(A, tau_values, dt_sim, 'IStart', i_start)
%
%   Computes the empirical mean-square displacement
%
%       MSD(τ) = ⟨[A(t+τ) − A(t)]²⟩                       (Eq. 5)
%
%   for each delay time τ in TAU_VALUES.
%
% INPUTS
%   A          – Amplitude envelope time series, length N+1
%   tau_values – Delay times τ to evaluate (s), vector of length n_tau
%   dt_sim     – Time step Δt (s)
%
% OPTIONAL NAME–VALUE PAIRS
%   'IStart'  – First index to use (1-based, default: 5001, i.e. skip 5000
%               samples to discard the initial transient)
%
% OUTPUTS
%   taus  – Effective delay times (s), length n_tau  (rounded to dt_sim grid)
%   MSD   – Mean-square displacement, length n_tau
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame.  Eq. 5,
%   Supplementary §4.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'IStart', 5001, @(x) isnumeric(x) && x >= 1);
parse(p, varargin{:});
i_start = p.Results.IStart;   % 1-based MATLAB index

% -------------------------------------------------------------------------
% Convert delay times to integer step counts
% -------------------------------------------------------------------------
A          = A(:);
N          = numel(A) - 1;           % number of steps (length - 1)
tau_values = tau_values(:);
n_tau      = numel(tau_values);

tk_vec = max(1, round(tau_values / dt_sim));   % step counts
taus   = tk_vec * dt_sim;                      % effective delays

MSD = zeros(n_tau, 1);

for idx = 1:n_tau
    tk    = tk_vec(idx);
    i_end = N - tk + 1;           % last 1-based index for A(t)

    if i_end <= i_start
        continue
    end

    x1 = A(i_start : i_end);
    x2 = A(i_start + tk : i_end + tk);

    % Discard unphysical (non-positive) envelope values
    mask = (x1 > 0) & (x2 > 0);
    x1m  = x1(mask);
    x2m  = x2(mask);

    if isempty(x1m)
        continue
    end

    delta   = x1m - x2m;
    MSD(idx) = (delta' * delta) / numel(delta);
end

end
