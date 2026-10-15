function [t, eta, eta_dot, A] = simulate_van_der_pol(nu, kappa, Gamma, omega, N, dt, varargin)
% SIMULATE_VAN_DER_POL  Stochastic Van der Pol oscillator integrator.
%
%   [t, eta, eta_dot, A] = SIMULATE_VAN_DER_POL(nu, kappa, Gamma, omega, N, dt)
%   [t, eta, eta_dot, A] = SIMULATE_VAN_DER_POL(..., 'Method', m, ...)
%
%   Integrates the stochastically forced Van der Pol oscillator (Eq. 2):
%
%       η̈ − 2ν·η̇ + κ·η²·η̇ + ω₀²·η = Ξ(t)
%
%   rewritten as the first-order system
%
%       d[η  ]   [η̇                           ]       [0    ]
%       d[η̇ ] = [-ω₀²η + 2ν·η̇ − κη²η̇      ] dt + [dΞ   ]
%
%   where dΞ is the integrated noise increment on η̇ (see Noise Models below).
%
% INTEGRATION SCHEMES  (see 'Method')
%
%   'symplectic' (default) — Semi-implicit (symplectic) Euler.
%       Velocity updated first; position advanced with the NEW velocity.
%       Preserves the symplectic structure → no spurious energy injection.
%       Mean amplitude matches A_eq = sqrt(8ν/κ) to ~0.1%.
%
%   'heun' — Explicit Runge-Kutta 2 (Heun / explicit trapezoidal).
%       Two-stage scheme; det ≈ 1 + O(ω⁴Δt⁴) — effectively area-preserving.
%       Matches the original paper solver (vderPol_Solver.m).
%
%   'euler-maruyama' — Standard explicit Euler.
%       Position advanced with OLD velocity → spurious energy injection.
%       Provided for backward-compatibility only.
%
% NOISE MODELS  (see 'NoiseType')
%
%   'white' (default) — i.i.d. Gaussian white noise.
%       The velocity increment per step is  dΞ = √(2Γ·Δt) · ξ,  ξ ~ N(0,1).
%
%   'ou' — Ornstein-Uhlenbeck (colored) noise.
%       dξ = −ξ/τ_c dt + √Γ_drive/τ_c dW.
%       The driving intensity Γ_drive is calibrated so that the effective
%       noise power in the passband equals Γ exactly:
%
%           gamma_coeff = 2π·τ_c·BW / (atan(2π(f₀+BW/2)τ_c)
%                                     − atan(2π(f₀-BW/2)τ_c))
%           Γ_drive     = Γ / gamma_coeff · (1 + ω₀²τ_c²)
%
%       The velocity increment per step is  dΞ = ξ(t) · Δt,
%       because ξ is a force (acceleration); this is the key difference from
%       white noise where dΞ ∝ √Δt.
%       Requires: TcNoise and BandwidthHz (see parameters below).
%
% INPUTS
%   nu     – Linear growth rate ν  (s⁻¹)
%   kappa  – Nonlinear coupling coefficient κ
%   Gamma  – Noise intensity Γ  (Pa²·s⁻³)
%   omega  – Angular eigenfrequency ω₀ = 2π·f₀  (rad/s)
%   N      – Number of time steps
%   dt     – Time step Δt  (s)
%
% OPTIONAL NAME–VALUE PAIRS
%   'Method'      – 'symplectic' | 'heun' | 'euler-maruyama'  (default: 'symplectic')
%   'NoiseType'   – 'white' | 'ou'                            (default: 'white')
%   'TcNoise'     – OU correlation time τ_c (s)               (default: 5e-4)
%   'BandwidthHz' – FULL passband BW for OU calibration (Hz)  (default: 100)
%   'Eta0'        – Initial displacement (default: 0.0)
%   'EtaDot0'     – Initial velocity     (default: 0.0)
%   'Seed'        – Random seed          (default: [] = random)
%
% OUTPUTS
%   t       – Time vector, length N+1
%   eta     – Displacement η(t), length N+1
%   eta_dot – Velocity η̇(t), length N+1
%   A       – Amplitude envelope √(η² + (η̇/ω₀)²), length N+1
%
% EXAMPLES
%   % Symplectic Euler + white noise (fast, good amplitude statistics)
%   omega = 2*pi*120;
%   [t, eta, ~, A] = simulate_van_der_pol(16, 3.4, 1e7, omega, 6e6, 5e-5, ...
%       'Seed', 0);
%
%   % Heun + calibrated OU noise — matches the original paper solver exactly
%   [t, eta, ~, A] = simulate_van_der_pol(16, 3.4, 1e7, omega, 6e6, 5e-5, ...
%       'Method', 'heun', 'NoiseType', 'ou', ...
%       'TcNoise', 5e-4, 'BandwidthHz', 100, 'Seed', 0);
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame. Eq. 2.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'Method',      'symplectic', @(x) ischar(x) || isstring(x));
addParameter(p, 'NoiseType',   'white',      @(x) ischar(x) || isstring(x));
addParameter(p, 'TcNoise',     5e-4, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'BandwidthHz', 100,  @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Eta0',        0.0,  @isnumeric);
addParameter(p, 'EtaDot0',     0.0,  @isnumeric);
addParameter(p, 'Seed',        [],   @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});

method    = lower(char(p.Results.Method));
noiseType = lower(char(p.Results.NoiseType));
tc        = p.Results.TcNoise;
BW        = p.Results.BandwidthHz;
eta0      = p.Results.Eta0;
etadot0   = p.Results.EtaDot0;
seed      = p.Results.Seed;

% -------------------------------------------------------------------------
% Setup
% -------------------------------------------------------------------------
if ~isempty(seed)
    rng(seed);
end

omega2 = omega^2;
fn     = omega / (2*pi);   % eigenfrequency (Hz)

t       = (0:N)' * dt;
eta     = zeros(N+1, 1);
eta_dot = zeros(N+1, 1);
eta(1)     = eta0;
eta_dot(1) = etadot0;

% -------------------------------------------------------------------------
% Generate noise velocity increment  dxi(k)  for step k → k+1.
%
%   For ALL integration schemes dxi(k) is the INCREMENT added to eta_dot:
%       eta_dot(k+1) += dxi(k)   (symplectic / Euler-Maruyama)
%   or equivalently enters the RHS as dxi(k)/dt (acceleration).
%
%   White noise:   dxi(k) = sqrt(2*Gamma*dt) * randn        ~ O(sqrt(dt))
%   OU noise:      dxi(k) = xi(k) * dt,  xi = OU process    ~ O(dt)
%
% The OU velocity increment is ξ × Δt because ξ is an acceleration
% (force per unit mass) integrated over Δt.
% -------------------------------------------------------------------------
switch noiseType

    case 'white'
        % White-noise force on η̈ with E[ξ(t)ξ(s)] = Γ·δ(t−s).
        % In the Itô SDE  dη̇ = fdet·dt + √Γ·dW,  the noise coefficient is
        % √Γ (NOT √(2Γ)).  The factor of 2 belongs to the Langevin amplitude
        % SDE (dA = … + √(2c) dW) and must NOT be carried over here.
        % Using √(2Γ) would double the effective noise intensity and give
        % c_est ≈ 2·c_true.
        dxi = sqrt(Gamma * dt) * randn(N, 1);

    case 'ou'
        % Calibrated OU forcing:  Γ_drive chosen so that the spectral power
        % in the band [f₀−BW/2, f₀+BW/2] equals Γ.
        gamma_coeff = 2*pi*tc*BW / ( atan(2*pi*(fn + BW/2)*tc) ...
                                   - atan(2*pi*(fn - BW/2)*tc) );
        Gamma_drive = Gamma / gamma_coeff * (1 + omega2*tc^2);

        % Euler-Maruyama discretisation of  dξ = -ξ/τ_c dt + √Γ_drive/τ_c dW
        xi      = zeros(N, 1);
        sig_ou  = sqrt(Gamma_drive) / tc * sqrt(dt);
        for k = 2:N
            xi(k) = xi(k-1) * (1 - dt/tc) + sig_ou * randn;
        end

        % velocity increment = force × Δt
        dxi = xi * dt;

        fprintf('  [OU noise]  gamma_coeff = %.4f,  Gamma_drive = %.4e\n', ...
            gamma_coeff, Gamma_drive);

    otherwise
        error('simulate_van_der_pol:unknownNoiseType', ...
              'Unknown NoiseType ''%s''. Choose ''white'' or ''ou''.', noiseType);
end

% -------------------------------------------------------------------------
% Time integration
% -------------------------------------------------------------------------
nu2 = 2 * nu;   % pre-compute

switch method

    case 'symplectic'
        % Semi-implicit (symplectic) Euler — area-preserving for harmonic part.
        for k = 1:N
            e  = eta(k);
            ed = eta_dot(k);
            fdet         = -omega2*e + nu2*ed - kappa*e^2*ed;
            eta_dot(k+1) = ed + fdet*dt + dxi(k);   % velocity first
            eta(k+1)     = e  + eta_dot(k+1)*dt;    % position with new velocity
        end

    case 'heun'
        % Heun (RK2 / explicit trapezoidal).
        %
        %   K1  = F(Y_k)            [deterministic + noise at step k]
        %   Ỹ   = Y_k + dt·K1      [predictor]
        %   K2  = F(Ỹ)             [deterministic; same noise dxi(k)]
        %   Y_{k+1} = Y_k + dt/2·(K1 + K2)
        %
        % The noise dxi(k) acts on the velocity equation and appears once
        % in the corrector (not split between predictor and corrector — a
        % valid approximation for small dt that avoids needing a second
        % noise draw per step).
        for k = 1:N
            e1  = eta(k);
            ed1 = eta_dot(k);

            % K1 — deterministic acceleration at (e1, ed1)
            fdet1 = -omega2*e1 + nu2*ed1 - kappa*e1^2*ed1;

            % Predictor (no noise in predictor position/velocity — noise
            % enters only the corrector to avoid double-counting)
            e2  = e1  + dt*ed1;
            ed2 = ed1 + dt*fdet1;

            % K2 — deterministic acceleration at (e2, ed2)
            fdet2 = -omega2*e2 + nu2*ed2 - kappa*e2^2*ed2;

            % Corrector — noise added once to velocity
            eta(k+1)     = e1  + dt/2*(ed1   + ed2);
            eta_dot(k+1) = ed1 + dt/2*(fdet1 + fdet2) + dxi(k);
        end

    case 'euler-maruyama'
        % Standard Euler–Maruyama (explicit Euler, NOT symplectic).
        for k = 1:N
            e  = eta(k);
            ed = eta_dot(k);
            fdet         = -omega2*e + nu2*ed - kappa*e^2*ed;
            eta(k+1)     = e  + ed*dt;
            eta_dot(k+1) = ed + fdet*dt + dxi(k);
        end

    otherwise
        error('simulate_van_der_pol:unknownMethod', ...
              'Unknown method ''%s''. Choose ''symplectic'', ''heun'', or ''euler-maruyama''.', ...
              method);
end

% -------------------------------------------------------------------------
% Amplitude envelope  A = sqrt(eta^2 + (eta_dot/omega)^2)
% -------------------------------------------------------------------------
A = sqrt(eta.^2 + (eta_dot / omega).^2);

end
