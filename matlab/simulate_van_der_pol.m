function [t, eta, eta_dot, A] = simulate_van_der_pol(nu, kappa, Gamma, omega, N, dt, varargin)
% SIMULATE_VAN_DER_POL  Euler–Maruyama integrator for the stochastic Van der Pol ODE.
%
%   [t, eta, eta_dot, A] = SIMULATE_VAN_DER_POL(nu, kappa, Gamma, omega, N, dt)
%   [t, eta, eta_dot, A] = SIMULATE_VAN_DER_POL(..., 'Eta0', e0, 'EtaDot0', ed0, 'Seed', s)
%
%   Integrates the stochastically forced Van der Pol oscillator (Eq. 2):
%
%       η̈ − 2ν·η̇ + κ·η·η̇ + ω₀²·η = Ξ(t)
%
%   rewritten as the first-order system:
%
%       d[η   ]   [η̇                           ]       [0        ]
%       d[η̇  ] = [-ω₀²η + 2ν·η̇ − κ·η·η̇  ] dt + [√(2Γ) dW ]
%
%   using the Euler–Maruyama scheme.  The amplitude envelope is computed as
%
%       A(t) = √[ η(t)² + (η̇(t)/ω₀)² ]
%
% INPUTS
%   nu     – Linear growth rate ν  (s⁻¹)
%   kappa  – Nonlinear coupling coefficient κ
%   Gamma  – White-noise intensity Γ  (Pa²·s⁻³)
%   omega  – Angular eigenfrequency ω₀ = 2π·f₀  (rad/s)
%   N      – Number of time steps
%   dt     – Time step Δt  (s)
%
% OPTIONAL NAME–VALUE PAIRS
%   'Eta0'    – Initial displacement (default: 0.01)
%   'EtaDot0' – Initial velocity     (default: 0.0)
%   'Seed'    – Random seed          (default: [] = random)
%
% OUTPUTS
%   t       – Time vector, length N+1
%   eta     – Displacement η(t), length N+1
%   eta_dot – Velocity η̇(t), length N+1
%   A       – Amplitude envelope √(η² + (η̇/ω₀)²), length N+1
%
% EXAMPLE
%   omega = 2*pi*120;
%   [t, eta, ~, A] = simulate_van_der_pol(16, 3.4, 1e7, omega, 2e6, 5e-5, 'Seed', 0);
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame. Eq. 2.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'Eta0',    0.01, @isnumeric);
addParameter(p, 'EtaDot0', 0.0,  @isnumeric);
addParameter(p, 'Seed',    [],   @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
eta0    = p.Results.Eta0;
etadot0 = p.Results.EtaDot0;
seed    = p.Results.Seed;

% -------------------------------------------------------------------------
% Setup
% -------------------------------------------------------------------------
if ~isempty(seed)
    rng(seed);
end

sigma_noise = sqrt(2 * Gamma * dt);
omega2      = omega^2;

t       = (0:N)' * dt;
eta     = zeros(N+1, 1);
eta_dot = zeros(N+1, 1);

eta(1)     = eta0;
eta_dot(1) = etadot0;

noise = randn(N, 1);

% -------------------------------------------------------------------------
% Euler–Maruyama integration
% -------------------------------------------------------------------------
for k = 1:N
    e  = eta(k);
    ed = eta_dot(k);

    eta_ddot   = -omega2 * e  +  2*nu * ed  -  kappa * e * ed;
    eta(k+1)   = e  + ed * dt;
    eta_dot(k+1) = ed + eta_ddot * dt  +  sigma_noise * noise(k);
end

% -------------------------------------------------------------------------
% Amplitude envelope
% -------------------------------------------------------------------------
A = sqrt(eta.^2 + (eta_dot / omega).^2);

end
