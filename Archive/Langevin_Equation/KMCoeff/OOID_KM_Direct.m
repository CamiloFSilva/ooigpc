% OOID_KM_Direct.m
% =========================================================================
% Kramers-Moyal (KM) identification of the Langevin amplitude equation
%   — direct conditional-statistics variant —
%
% DESCRIPTION
%   Integrates the stochastic amplitude (Langevin) equation via Euler-
%   Maruyama and identifies the system parameters using the conditional-
%   transition-moment approach (Kramers-Moyal coefficients):
%
%     dA/dt = ν·A − b·A³ + c/A + ζ(t),   b = κ/8,  c = Γ/(4ω₀²)
%
%   The identification proceeds in four steps:
%     1. Bin the amplitude sequence via discretize() and, for each bin,
%        directly collect the set of conditional jumps
%          Δ_i(τ) = { A(t+τ) − A(t) | A(t) ∈ bin i }.
%        No intermediate conditional PDF matrix is built.
%     2. Compute first (drift) and second (diffusion) transition moments
%        as the sample mean and mean-square of Δ_i, divided by τ.
%     3. Extrapolate moments to τ→0 → 1st and 2nd KM coefficients.
%     4. Fit the Langevin amplitude model to the drift coefficient.
%
%   The 2nd KM coefficient (additive noise) is used to estimate  Γ
%   directly; the 1st KM coefficient is fitted via least squares to
%   recover  ν  and  κ.
%
%   Compare with OOID_KM_PDFMatrix.m, which builds the full conditional PDF
%   PA(τ, A_i, A_j) first and then computes moments as weighted sums.
%   Both scripts use identical parameters and should yield equivalent
%   results; see README.md for a discussion of the trade-offs.
%
%   Results for all growth rates in nu_vec are saved to ../DATA/ for
%   downstream plotting.
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
%
% METHOD REFERENCES
%   Siegert, Friedrich, Peinke (1998).  Phys. Lett. A 243, 275-280.
%   Noiray & Schuermans (2013).  Int. J. Non-linear Mech. 50, 152-163.
%   Boujo & Noiray (2017).  Proc. R. Soc. A 473, 20160894.
%
% SEE ALSO
%   OOID_KM_PDFMatrix.m       — PDF-matrix variant (same parameters)
%   extrapolate_KM.m          — exponential extrapolation for the 2nd KM coeff
%   extrapolate_KM_3coeff.m   — 3-coefficient extrapolation for the drift
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

save_data = true;   % true  → save identified parameters to ../DATA/
                    % false → run without writing .mat file

% =========================================================================
%% PHYSICAL PARAMETERS  (ground truth for validation)
% =========================================================================
%
%   Amplitude equation:  dA/dt = ν·A − b·A³ + c/A + ζ(t)
%     ν  = linear growth rate        [rad/s]
%     b  = κ/8                       [m⁻²·s⁻¹]  saturation parameter
%     c  = Γ/(4ω₀²)                 [m²·s⁻¹]   diffusion coefficient

kappa = 3.4;        % [m⁻²·s⁻¹]  nonlinear saturation coefficient
Gamma = 1e7;        % [m²·s⁻³]   white-noise forcing intensity
omega = 2*pi*120;   % [rad/s]     natural angular frequency  (f₀ = 120 Hz)

b_true = kappa / 8;
c_true = Gamma / (4*omega^2);

% =========================================================================
%% KM ESTIMATION PARAMETERS
% =========================================================================

nbin = 30;          % number of amplitude bins

% Time-lag values for computing transition moments  [s]
% These lags must be large enough for the PDF to evolve, but small
% enough that the exponential extrapolation is valid.
% Best results:
tau_val = 0.005 : 0.01 : 0.1;
% Too short (not recommended):  tau_val = 0.00025 : 0.00025 : 0.005;

% Extrapolation settings
%   use_extrap = 1 → exponential extrapolation to τ→0 (default; see
%                    Boujo & Noiray 2017 for derivation)
%   use_extrap = 0 → constant extrapolation for the 1st KM coefficient
%                    (use only if exponential gives poor results)
use_extrap = 1;
tmin = tau_val(1);  % shortest lag used in the extrapolation

% =========================================================================
%% SCAN PARAMETERS
% =========================================================================

NumSim = 100;   % number of independent realisations per growth rate

% Growth rates to loop over  [rad/s]
nu_vec = [0, 4, 8, 12, 16, 20, 24];

% =========================================================================
%% IDENTIFICATION LOOP
% =========================================================================

for nk = 1:length(nu_vec)

    nu_true = nu_vec(nk);
    b_true  = kappa / 8;
    c_true  = Gamma / (4*omega^2);

    % Pre-allocate output vectors
    nu_hat_vec = zeros(NumSim, 1);
    b_hat_vec  = zeros(NumSim, 1);
    c_hat_vec  = zeros(NumSim, 1);

    for ns = 1:NumSim

        fprintf('\n--- Simulation %d / %d  (ν_true = %g rad/s) ---\n', ...
                ns, NumSim, nu_true);

        % ------------------------------------------------------------------
        % 1. Simulate the discrete Langevin amplitude equation
        % ------------------------------------------------------------------
        % Itô Euler-Maruyama step:
        %   A_{k+1} = A_k + ν·A_k·dt − b·A_k³·dt + (c/A_k)·dt + √(2c·dt)·ε_k
        % where ε_k ~ Truncated-Normal(0,1) on [−4.5, 4.5].

        N      = 2000000;
        dt_sim = 5e-5;                      % [s] integration time step
        s1     = sqrt(2*c_true*dt_sim);     % noise amplitude per step

        pd    = makedist('Normal');
        pd_tr = truncate(pd, -4.5, 4.5);
        noise = random(pd_tr, N, 1);

        x    = zeros(1, N);
        x(1) = 0.65;

        for k = 1:N
            x(k+1) = x(k) ...
                   + nu_true  * dt_sim * x(k) ...
                   - b_true   * dt_sim * x(k)^3 ...
                   + (c_true  / x(k)) * dt_sim ...
                   + s1 * noise(k);
        end

        % Discard transient and unphysical negative values
        x_all = x(5000:end);
        ampl  = x_all(x_all > 0);   % [m] stationary positive-amplitude vector

        Fs = 1 / dt_sim;    % sampling rate [Hz]
        nt = length(ampl);

        if isempty(ampl);  continue;  end

        % ------------------------------------------------------------------
        % 2. Transition moments — direct conditional-statistics approach
        % ------------------------------------------------------------------
        % For each lag τ and each amplitude bin i, collect the set of
        % conditional jumps:
        %
        %   Δ_i(τ) = { A(t+τ) − A(t) | A(t) ∈ bin i }
        %
        % and compute:
        %
        %   M₁(τ, i) = mean( Δ_i ) / τ          (drift proxy)
        %   M₂(τ, i) = mean( Δ_i² ) / τ         (diffusion proxy)
        %
        % These converge to the KM coefficients D₁ and D₂ as τ→0.
        % No PA(τ, i, j) matrix is ever stored.

        fprintf('  Computing transition moments (direct)...\n');  tic

        Amin  = min(ampl);
        Amax  = max(ampl);
        dA    = (Amax - Amin) / nbin;              % bin width [m]
        Abin  = linspace(Amin + dA, Amax, nbin);   % bin right-edges [m]
        edges = [Amin, Abin];                       % bin boundary vector

        ntau  = length(tau_val);
        DA1t  = zeros(ntau, nbin);   % 1st transition moment
        DAA2t = zeros(ntau, nbin);   % 2nd transition moment

        % Map every sample to its bin index and bin floor value
        bin_idx   = discretize(ampl, edges);
        bin_idx(isnan(bin_idx)) = nbin;   % clamp any boundary sample
        bin_floor = edges(bin_idx)';      % binned amplitude — column vector [m]
        % NOTE: both A(t) and A(t+τ) must use the discretised (binned) signal,
        % not raw ampl.  Using raw ampl adds within-bin variance to every jump,
        % inflating D₂ and distorting D₁.  Matches OOID_KM_PDFMatrix.m, which
        % also computes moments in bin space.

        for its = 1:ntau
            shift   = floor(tau_val(its) * Fs);   % lag in samples
            ref_idx = bin_idx(1 : nt - shift);    % bin index of A(t)

            for ib = 1:nbin
                I = find(ref_idx == ib);
                if isempty(I);  continue;  end

                % jumps: binned A(t+τ) − bin floor of bin ib
                % All ampl(I) lie inside bin ib, so their bin floor = edges(ib)
                jumps = bin_floor(I + shift) - edges(ib);

                DA1t (its, ib) = mean(jumps)    / tau_val(its);
                DAA2t(its, ib) = mean(jumps.^2) / tau_val(its);
            end
        end

        toc

        % Amplitude bins with sufficient statistics for reliable extrapolation
        % Criterion: each bin should contain at least (N/nbin)/50 counts.
        stats_A = histcounts(ampl, edges);
        Arange  = find(stats_A > nt / (nbin * 50));

        % ------------------------------------------------------------------
        % 3. Extrapolate transition moments to τ→0 → KM coefficients
        % ------------------------------------------------------------------
        % Fits an exponential of the form M(τ) = D + A·exp(B·τ) and returns
        % the τ=0 intercept D as the KM coefficient.

        fprintf('  Extrapolating KM coefficients...\n')

        range_drift = find(tau_val >= 0.0 & tau_val <= 0.5);
        range_diff  = find(tau_val >= 0.0 & tau_val <= 0.5);

        tau_rest_drift = tau_val(range_drift);
        tau_rest_diff  = tau_val(range_diff);

        % 1st KM coefficient: drift  D₁(A)  [m·s⁻¹]
        DA1_ext = extrapolate_KM_3coeff( ...
            tau_rest_drift, DA1t(range_drift, Arange), tmin, -0.005);
        D1_KM = DA1_ext(1, :);   % extrapolated drift at τ=0

        % 2nd KM coefficient: diffusion  D₂(A)  [m²·s⁻¹]
        DAA2_ext = extrapolate_KM( ...
            tau_rest_diff, DAA2t(range_diff, Arange), tmin, -0.01);
        D2_KM = DAA2_ext(1, :);  % extrapolated diffusion at τ=0

        % ------------------------------------------------------------------
        % 4. Parameter identification
        % ------------------------------------------------------------------

        % --- Noise intensity  Γ  from the 2nd KM coefficient --------------
        % For additive noise D₂(A) is constant (independent of A).
        % The simulation noise amplitude is √(2c·dt), so the per-unit-time
        % variance is 2c, meaning D₂ → 2c as τ→0.  Hence:
        %   Γ = D₂_mean · 2ω₀²
        % Average D₂, weighted by the amplitude PDF.
        pdf_ampl      = histcounts(ampl, edges, 'Normalization', 'probability');
        pdf_ampl_rest = pdf_ampl(Arange);
        pdf_ampl_rest = pdf_ampl_rest / sum(pdf_ampl_rest);   % normalise

        % D2_KM → 2c as τ→0 (noise variance per unit time is 2c);
        % divide by 2 to recover the diffusion coefficient c.
        c_hat     = sum(D2_KM .* pdf_ampl_rest) / 2;   % = c  [m²·s⁻¹]
        Gamma_hat = c_hat * 4 * omega^2;                % [m²·s⁻³]

        % --- Growth rate  ν  and saturation  κ  from the 1st KM coeff ----
        % The Langevin amplitude model gives:
        %   D₁(A) = ν·A − b·A³ + c/A
        % Fit c₁ (= ν) and c₂ (= −b) by weighted least squares.

        D1_model = @(c1, c2, a) c1*a + c2*a.^3 + Gamma_hat./(4*a.*omega^2);
        cost_fn  = @(coeff) sum( ...
            (D1_KM .* pdf_ampl_rest - D1_model(coeff(1), coeff(2), Abin(Arange)) .* pdf_ampl_rest).^2 );

        opts      = optimset('MaxFunEvals', 1000*2);
        coeff_fit = fminsearch(cost_fn, [20, 2e-5], opts);

        alpha_fit = coeff_fit(1);    % identified linear coefficient  (= ν)
        beta_fit  = coeff_fit(2);    % identified cubic coefficient   (= −b = −κ/8)

        nu_hat    = alpha_fit;
        b_hat     = -beta_fit;       % note sign: D₁ model uses +β·A³, Langevin uses −b·A³
        kappa_hat = 8 * b_hat;

        % Store for ensemble statistics
        nu_hat_vec(ns) = nu_hat;
        b_hat_vec(ns)  = b_hat;
        c_hat_vec(ns)  = c_hat;

        % Display identified vs. true values
        fprintf('  ν    identified: %8.4f   true: %8.4f  [rad/s]\n',   nu_hat,    nu_true);
        fprintf('  κ    identified: %8.4f   true: %8.4f  [m⁻²s⁻¹]\n', kappa_hat, kappa);
        fprintf('  Γ    identified: %8.3g   true: %8.3g  [m²s⁻³]\n',  Gamma_hat, Gamma);

    end   % ns loop

    % Save identified parameters (controlled by save_data flag above)
    params_true = [nu_true, b_true, c_true];
    params_hat  = [nu_hat_vec, b_hat_vec, c_hat_vec];

    if save_data
        file = ['../DATA/nu_', num2str(nu_true), '_fromKMCoeffs.mat'];
        save(file, 'params_true', 'params_hat')
        fprintf('Data saved to: %s\n', file);
    end

end   % nk loop
