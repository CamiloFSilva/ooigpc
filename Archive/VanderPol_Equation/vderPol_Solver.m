% vderPol_Solver.m
% =========================================================================
% Stochastic Van der Pol oscillator — simulation and envelope extraction
%
% DESCRIPTION
%   Integrates the stochastic Van der Pol oscillator (second-order SDE)
%   using a second-order Runge-Kutta (Heun) scheme.  The broadband noise
%   is pre-generated as a coloured Ornstein-Uhlenbeck process and then
%   injected as a forcing.  After integration, the signal is band-pass
%   filtered around the natural frequency and the amplitude envelope is
%   extracted via the Hilbert transform.
%
%   This simulation produces the data used to generate Fig. 2 of:
%     Silva, C. F. (2026).
%     "Output-only identification of thermoacoustic systems using
%      polynomial chaos expansions." Combustion and Flame.
%
% CREDITS
%   Original script (sys_id_VDP_RK.m) contributed by N. Noiray (ETH Zurich).
%   Method based on:
%     Siegert, Friedrich, Peinke (1998), Phys. Lett. A 243, 275-280
%     Noiray, Schuermans (2013),  Int. J. Non-linear Mech. 50, 152-163
%   OU noise calibration from:
%     Bonciolini, Boujo, Noiray (2017), Phys. Rev. Applied 8, 015103
%
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

save_data = true;  % true  → save envelope and metadata to DATA/ folder
                    % false → run without writing any .mat file

save_figs = false;   % true  → save figures as PDF to Plotting/ folder
                    % false → display figures without saving

% =========================================================================
%% PHYSICAL PARAMETERS
% =========================================================================

% Van der Pol oscillator:
%   η̈ − 2ν η̇ + κη²η̇ + ω₀²η = Ξ(t)
%
%   ω₀  : natural angular frequency
%   α   : acoustic damping coefficient
%   β   : linear gain from flame response
%   ν   : effective linear growth rate  ν = (β − α)/2
%   κ   : nonlinear saturation coefficient (3rd-order flame response)
%   Γ   : intensity of the stochastic broadband noise forcing

wn      = 2*pi*120; % [rad/s] natural angular frequency (f₀ = 120 Hz)
alpha   = 48;       % [rad/s] acoustic damping
beta    = 96;       % [rad/s] linear flame gain (ν = (β−α)/2 = 4 rad/s → limit cycle)
%beta   = 88;       % [rad/s] alternative: ν = 20 rad/s (stronger instability)
kappa   = 3.4;      % [1/(m²·s)] nonlinear saturation coefficient
Gamma   = 1e7;      % [m²/s³]   white-noise intensity of the equivalent forcing

% =========================================================================
%% NUMERICAL PARAMETERS
% =========================================================================

T         = 300;    % [s]   total simulation time
Fs        = 20000;  % [Hz]  sampling frequency  (must satisfy Fs >> f₀)
bandwidth = 100;    % [Hz]  half-bandwidth of the band-pass filter around f₀
tc        = 0.0005; % [s]   correlation time of the OU noise
                    %       Conditions: tc >> 1/Fs  AND  tc << 2π/ω₀
                    %       Weak noise:  tc = 0.0005
                    %       Strong memory: tc = 0.002

% =========================================================================
%% FIGURE STYLE PARAMETERS
% =========================================================================

fig_start        = 1;    % starting figure index
fontSize         = 20;
lineWidthThin    = 1.0;
lineWidth        = 2;
lineWidthBorder  = 1.5;
lineWidthThick   = 2.4;
markSize         = 15;
markSizeBig      = 80;

% =========================================================================
%% DERIVED QUANTITIES
% =========================================================================

nu = (beta - alpha) / 2;  % [rad/s] effective linear growth rate
fn = wn / (2*pi);         % [Hz]    natural frequency

% Time vector
dt = 1/Fs;
nt = round(T/dt);
t  = (0 : nt-1) * dt;    % [s], row vector length nt

% Apply global figure defaults
fig = fig_start;
set(groot, 'defaultTextFontSize',               fontSize);
set(groot, 'defaultAxesFontSize',               fontSize);
set(groot, 'defaultLegendFontSize',             fontSize);
set(groot, 'defaultTextInterpreter',            'latex');
set(groot, 'defaultAxesTickLabelInterpreter',   'latex');
set(groot, 'defaultColorbarTickLabelInterpreter','latex');

% =========================================================================
%% OU NOISE CALIBRATION
% =========================================================================
%
% The oscillator is forced by a coloured noise ξ(t) obeying:
%   dξ = −ξ/τc dt  +  √(Γ_drive)/τc · dW
%
% The driving intensity Γ_drive is adjusted so that, after band-pass
% filtering around f₀ ± bandwidth/2, the effective noise level equals Γ.
% See Bonciolini, Boujo & Noiray (2017) for the derivation.

gamma_coeff = 2*pi*tc*bandwidth / ...
    ( atan(2*pi*(fn + bandwidth/2)*tc) - atan(2*pi*(fn - bandwidth/2)*tc) );

Gamma_drive_OU = Gamma / gamma_coeff * (1 + wn^2*tc^2);

% =========================================================================
%% SIMULATION LOOP
% =========================================================================
%
% NumSim: number of independent realisations.
%   NumSim = 1  for a single run (default).
%   NumSim = 100 for ensemble statistics (comment/uncomment below).

NumSim = 100;
%NumSim = 1;

% Initial conditions (amplitude and phase of the first cycle)
Ai    = 0;      % [m]   initial amplitude (0 → start from rest)
phi_i = 0;      % [rad] initial phase

for ns = 1:NumSim

    % ------------------------------------------------------------------
    % 1. Generate Ornstein-Uhlenbeck noise forcing
    % ------------------------------------------------------------------
    xi = zeros(1, nt);
    for it = 2:nt
        % Euler-Maruyama step for the OU process:
        %   ξ(t+dt) = ξ(t) + (−ξ(t)/τc)·dt + √(Γ_drive)/τc · √dt · N(0,1)
        xi(it) = xi(it-1) ...
                 - xi(it-1)/tc * dt ...
                 + sqrt(Gamma_drive_OU)/tc * randn * sqrt(dt);
    end

    % ------------------------------------------------------------------
    % 2. Integrate the Van der Pol oscillator (Heun / RK2 scheme)
    % ------------------------------------------------------------------
    % State vector: Y = [η, η̇]ᵀ
    % ODE written as:  Ẏ = F(Y, t)  where
    %   F(1) = η̇
    %   F(2) = −ω₀²η + 2ν·η̇ − κη²·η̇ + ξ(t)

    % Deterministic part of the restoring/damping force (sign: moved to RHS)
    fa = @(eta, dt_eta) wn^2*eta - 2*nu*dt_eta + kappa*(eta^2)*dt_eta;

    % Full RHS including noise forcing
    F  = @(Y, it) [Y(2); -fa(Y(1), Y(2)) + xi(it)];

    % Initial state
    Yt      = zeros(2, nt);
    Yt(1,1) = Ai * cos(phi_i);    % η(0)
    Yt(2,1) = -wn * Ai * sin(phi_i); % η̇(0)

    progression = 0;
    prog_step   = 10;
    disp('Simulation running...');
    tic
    for it = 2:nt
        % Heun (RK2) predictor-corrector:
        %   Y* = Y_n + dt·F(Y_n)        [predictor (Euler)]
        %   Y_{n+1} = Y_n + dt/2·[F(Y_n) + F(Y*)]  [corrector (trapezoidal)]
        Y1  = Yt(:, it-1);
        FY1 = F(Y1, it-1);
        Y2  = Y1 + dt * FY1;              % predictor
        Yt(:,it) = Y1 + dt/2*FY1 + dt/2*F(Y2, it);  % corrector

        if it/nt >= (progression + prog_step)/100
            progression = progression + prog_step;
            disp([num2str(progression) ' %']);
        end
    end
    toc

    % Raw oscillator signal η(t)
    eta = Yt(1,:);

    % ------------------------------------------------------------------
    % 3. Transient removal, filtering, and envelope extraction
    % ------------------------------------------------------------------

    % Index to discard the initial transient: 5× the characteristic
    % linear growth time  T_growth = 1/ν.
    % NOTE: ind_trans marks the start of the statistically stationary
    %       portion of the signal.  Use it when computing statistics, e.g.:
    %           eta_stat = eta(ind_trans:end);
    ind_trans = round(Fs * 5 / nu);

    % Band-pass filter around f₀ ± bandwidth/2 to suppress harmonics.
    % (The function BP_filter_rect is in the same folder.)
    [eta_f, ~, ~] = BP_filter_rect(eta, fn - bandwidth/2, fn + bandwidth/2, Fs);

    % Amplitude envelope via Hilbert transform of the filtered signal
    A{ns} = abs(hilbert(eta_f));

end

% =========================================================================
%% SAVE DATA  (controlled by save_data flag above)
% =========================================================================

if save_data
    file = ['./DATA/Data_nu', num2str(nu), '.mat'];
    save(file, 'A', 'dt', 'Gamma', 'wn', '-v7.3')
    disp(['Data saved to: ' file]);
end

% =========================================================================
%% PLOTS
% =========================================================================
%
% Figure 1 — Time series η(t) and envelope A(t)   [→ Fig. 2 of the paper]
% Figure 2 — Histogram of η
% Figure 3 — Histogram of amplitude envelope A

% Window for the time-series plot (a 5-second excerpt in the middle)
init = 2000000;
fin  = 2100000;

% ---- Figure 1: time series + envelope -----------------------------------
figure(1)
plot(t, eta,    'k', 'lineWidth', 0.5);  hold on
plot(t, A{ns},   'r', 'lineWidth', 2);
xlim(t([init fin]));
ylim([-10 10]);
set(gca, 'FontSize', 24)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 10], ...
         'PaperPosition', [0 0 20 10])
yticklabels([]);
if save_figs
    filefig = ['./Plotting/etaAndEnvelope_', num2str(nu), '.pdf'];
    saveas(gcf, filefig)
end

% ---- Figure 2: η histogram (narrow strip, axis labels removed) ----------
figure(2)
histogram(eta, 'FaceColor', 'k', 'EdgeColor', 'k');
xticklabels([]);  yticklabels([]);
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 2], ...
         'PaperPosition', [0 0 10 2])
xlim([-10 10]);
if save_figs
    filefig = ['./Plotting/etaHistogram_', num2str(nu), '.pdf'];
    saveas(gcf, filefig)
end

% ---- Figure 3: amplitude envelope histogram -----------------------------
x = A{1};

% For ν = 16 the equilibrium amplitude is large and the PDF bulk lies
% above 6 Pa.  Restricting the histogram to [6, 10] removes the initial
% transient tail and makes the PDF shape visible at the same xlim scale.
if nu == 16
    x_subset = x(x >= 6 & x <= 10);
else
    x_subset = x;
end

figure(3)
histogram(x_subset, 'FaceColor', 'r', 'EdgeColor', 'r');
xticklabels([]);  yticklabels([]);
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 2], ...
         'PaperPosition', [0 0 10 2])
xlim([-10 10]);
if save_figs
    filefig = ['./Plotting/AHistogram_', num2str(nu), '.pdf'];
    saveas(gcf, filefig)
end
