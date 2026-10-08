function A = simulate_langevin(nu, kappa, Gamma, omega, N, dt, varargin)
% SIMULATE_LANGEVIN  Euler–Maruyama integrator for the amplitude envelope SDE.
%
%   A = SIMULATE_LANGEVIN(nu, kappa, Gamma, omega, N, dt)
%   A = SIMULATE_LANGEVIN(nu, kappa, Gamma, omega, N, dt, 'A0', A0, 'Seed', s)
%
%   Integrates the Itô SDE for the amplitude envelope A(t) of a
%   self-excited thermoacoustic oscillator (Eq. 3 of the paper):
%
%       dA = (a·A − b·A³ + c/A) dt  +  √(2c) dW
%
%   using the Euler–Maruyama discretisation (Eq. 4):
%
%       A_{n+1} = A_n + a·Δt·A_n − b·Δt·A_n³ + (c/A_n)·Δt + √(2c·Δt)·ξ_n
%
%   with  a = ν,  b = κ/8,  c = Γ/(4ω₀²),  ξ_n ~ TN(0,1; ±4.5).
%
% INPUTS
%   nu     – Linear growth rate ν  (s⁻¹)
%   kappa  – Nonlinear damping coefficient κ  (Pa⁻²·s⁻¹)
%   Gamma  – Noise intensity Γ  (Pa²·s⁻³)
%   omega  – Angular eigenfrequency ω₀ = 2π·f₀  (rad/s)
%   N      – Number of time steps
%   dt     – Time step Δt  (s)
%
% OPTIONAL NAME–VALUE PAIRS
%   'A0'    – Initial amplitude (default: 0.65)
%   'Seed'  – Random seed for reproducibility (default: [] = random)
%
% OUTPUT
%   A  – Amplitude envelope, length N+1  (column vector)
%
% EXAMPLE
%   omega = 2*pi*120;
%   A = simulate_langevin(16, 3.4, 1e7, omega, 2e6, 5e-5, 'Seed', 0);
%
% REFERENCE
%   Silva, C.F. et al. (2026). Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions. Combust. Flame. Eqs. 3–4.

% -------------------------------------------------------------------------
% Parse optional arguments
% -------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'A0',   0.65, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Seed', [],   @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
A0   = p.Results.A0;
seed = p.Results.Seed;

% -------------------------------------------------------------------------
% Internal Langevin coefficients
% -------------------------------------------------------------------------
a  = nu;
b  = kappa / 8;
c  = Gamma / (4 * omega^2);
s1 = sqrt(2 * c * dt);   % noise amplitude  √(2c·Δt)

% -------------------------------------------------------------------------
% Truncated-normal noise  TN(0, 1; ±4.5)  — matches MATLAB makedist/truncate
% -------------------------------------------------------------------------
if ~isempty(seed)
    rng(seed);
end

xi = zeros(N, 1);
filled = 0;
trunc  = 4.5;
while filled < N
    needed = N - filled;
    batch  = randn(ceil(needed * 1.15) + 200, 1);
    batch  = batch(abs(batch) <= trunc);
    take   = min(numel(batch), needed);
    xi(filled+1 : filled+take) = batch(1:take);
    filled = filled + take;
end

% -------------------------------------------------------------------------
% Euler–Maruyama integration  (Eq. 4)
% -------------------------------------------------------------------------
A    = zeros(N+1, 1);
A(1) = A0;

for k = 1:N
    Ak     = A(k);
    A(k+1) = Ak ...
            + a  * dt * Ak     ...   % linear growth
            - b  * dt * Ak^3   ...   % nonlinear saturation
            + (c / Ak) * dt    ...   % diffusion correction
            + s1 * xi(k);            % stochastic forcing
end

end
