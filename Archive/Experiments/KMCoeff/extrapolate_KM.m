% =========================================================================
% extrapolate_KM.m  —  2-coefficient exponential extrapolation of KM
%                       transition moments to tau = 0
%
% Description:
%   For each amplitude bin, fits a 2-parameter exponential model
%       f(tau) = a1 · exp(a2 · tau)
%   to the transition moment values KMt(tau) using fminsearch,
%   then evaluates the model on the extended grid [0, tval] to
%   obtain the Kramers-Moyal coefficient at tau = 0.
%
%   Typically used for the 2nd transition moment (diffusion coefficient),
%   which approaches a finite constant as tau → 0.
%
% Usage:
%   KM_ext = extrapolate_KM(tval, KMt, tmin, init_growth_guess)
%
% Inputs:
%   tval               — time-lag values [1 × ntau]
%   KMt                — transition moments [ntau × na]
%                        na: number of amplitude bins with sufficient data
%   tmin               — lower bound for the fitting range [s]
%   init_growth_guess  — initial guess for the exponential growth rate a2
%
% Output:
%   KM_ext  — extrapolated moments [ntau+1 × na]
%             Row 1 corresponds to tau = 0 (the KM coefficient)
%             Rows 2:end correspond to the original tval grid
%
% References:
%   Boujo & Noiray (2017). Robust identification of harmonic oscillator
%   parameters using the adjoint Fokker–Planck equation.
%   J. Fluid Mech. 828, 457–485.
%
%   Noiray & Schuermans (2013). Int. J. Nonlinear Mech. 50, 152–163.
%   Siegert, Friedrich & Peinke (1998). Phys. Lett. A 243, 275–280.
%
% See also: extrapolate_KM_3coeff.m
% =========================================================================

function KM_ext = extrapolate_KM(tval, KMt, tmin, init_growth_guess)

    [nt, na] = size(KMt);
    KM_ext   = zeros(nt + 1, na);

    % Index of tmin in the time-lag array
    t_ind  = find(tval >= tmin, 1, 'first');
    t_rest = tval(t_ind : end);
    t_ext  = [0, tval];                     % Extended grid including tau = 0

    options = optimset('MaxFunEvals', 2000 * nt);

    for ia = 1 : na

        to_ext = KMt(t_ind : end, ia);      % Data to fit

        % 2-coefficient exponential model: f(tau) = a1 * exp(a2 * tau)
        f_ext = @(a1, a2, x) a1 * exp(a2 * x);
        fopt  = @(coeff) sum( (to_ext(:) - f_ext(coeff(1), coeff(2), t_rest(:))).^2 );

        c_start = [KMt(t_ind, ia) * exp(0.1 * tmin), init_growth_guess];
        c       = fminsearch(fopt, c_start, options);

        KM_ext(:, ia) = c(1) * exp(c(2) * t_ext);
    end

end
