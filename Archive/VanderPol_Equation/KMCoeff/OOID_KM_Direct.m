% OOID_KM_Direct.m  (Van der Pol version)
% =========================================================================
% Kramers-Moyal (KM) identification of Van der Pol data — direct method
%
% DESCRIPTION
%   Loads pre-computed amplitude envelope data from vderPol_Solver.m and
%   identifies the system parameters using direct conditional-moment
%   estimation of the Kramers-Moyal coefficients.
%
%   The direct method bins the amplitude axis and, for each bin, selects
%   all time pairs (x(t), x(t+τ)) where x(t) falls in that bin.  The
%   conditional mean and variance of the increment directly give the
%   first and second KM coefficients.  Compare with OOID_KM_PDFMatrix.m,
%   which builds the full joint PDF PA(τ, A_i, A_j) instead.
%
%   Identification steps:
%     1. Discretise amplitude trace into nbin bins via discretize()
%     2. For each bin and each lag τ, compute D₁(τ,A) and D₂(τ,A)
%        from the conditional mean and variance of the increments
%     3. Extrapolate moments to τ→0 → 1st and 2nd KM coefficients
%     4. Fit the Langevin amplitude model to the drift coefficient
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
%
% DATA INPUT
%   ../DATA/Data_nu<nu>.mat  — produced by vderPol_Solver.m
%   Expected variables: A (cell array of envelopes), dt, Gamma, wn
%
% METHOD REFERENCES
%   Siegert, Friedrich, Peinke (1998).  Phys. Lett. A 243, 275-280.
%   Noiray & Schuermans (2013).  Int. J. Non-linear Mech. 50, 152-163.
%   Boujo & Noiray (2017).  Proc. R. Soc. A 473, 20160894.
%
% SEE ALSO
%   OOID_KM_PDFMatrix.m  — alternative KM implementation using joint PDF
%   OOID_igPC.m          — igPC identification applied to same VdP data
%   vderPol_Solver.m     — generates the input data
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
%% PHYSICAL PARAMETERS  (ground truth; Gamma and omega loaded from file)
% =========================================================================

kappa = 3.4;        % [m⁻²·s⁻¹]  nonlinear saturation coefficient
                    % (used only for displaying true vs. identified values)

% =========================================================================
%% KM ESTIMATION PARAMETERS
% =========================================================================

nbin = 30;          % number of amplitude bins

% Time-lag values for computing transition moments  [s]
tau_val = 0.015 : 0.01 : 0.1;

% Extrapolation settings
use_extrap = 1;     % 1 = exponential extrapolation (default; Boujo & Noiray 2017)
tmin = tau_val(1);  % shortest lag used in the extrapolation

% =========================================================================
%% SCAN PARAMETERS
% =========================================================================

NumSim = 100;    % number of independent realisations (must match vderPol_Solver.m)

% Growth rates to loop over — must match the saved DATA/ files  [rad/s]
nu_vec = [16, 20, 24];
%nu_vec = [0, 4, 8, 12, 16, 20, 24];

% =========================================================================
%% IDENTIFICATION LOOP
% =========================================================================

for nk = 1:length(nu_vec)

    nu_true = nu_vec(nk);

    % Pre-allocate output vectors
    nu_hat_vec = zeros(NumSim, 1);
    b_hat_vec  = zeros(NumSim, 1);
    c_hat_vec  = zeros(NumSim, 1);

    for ns = 1:NumSim

        fprintf('\n--- Realisation %d / %d  (ν_true = %g rad/s) ---\n', ...
                ns, NumSim, nu_true);

        % ------------------------------------------------------------------
        % 1. Load pre-computed amplitude envelope data
        % ------------------------------------------------------------------

        file = ['../DATA/Data_nu', num2str(nu_true), '.mat'];
        load(file, 'A', 'dt', 'Gamma', 'wn')

        x      = A{ns};        % [m]  amplitude envelope from realisation ns
        dt_sim = dt;           % [s]  simulation time step
        N      = length(x);
        omega  = wn;           % [rad/s]  natural angular frequency

        b_true = kappa / 8;
        c_true = Gamma / (4*omega^2);

        tk_vec = floor(tau_val / dt_sim);

        % Bin edges and bin centres (shared across all lags)
        Amin = min(x);
        Amax = max(x);
        dA   = (Amax - Amin) / nbin;
        Abin = linspace(Amin + dA, Amax, nbin);
        edges      = [Amin, Abin];
        Achosen_vec = edges;         % one conditional value per edge

        % ------------------------------------------------------------------
        % 2. Direct conditional-moment estimation of D₁ and D₂
        % ------------------------------------------------------------------

        fprintf('  Computing conditional moments...\n');  tic

        D1  = zeros(length(tk_vec), length(Achosen_vec));
        D2  = zeros(length(tk_vec), length(Achosen_vec));

        for kk = 1:length(tk_vec)

            tk    = tk_vec(kk);
            dt_lag = tk * dt_sim;    % [s] physical time lag

            % Discretise the amplitude trace to bin floor values
            idx    = discretize(x, edges);
            Afloor = edges(idx);

            xvec1 = Afloor(1 : N - tk);
            xvec2 = Afloor(1 + tk : N);

            for k = 1:length(Achosen_vec)

                Amin_loc = Achosen_vec(k);
                indices  = find(xvec1 == Amin_loc);
                if isempty(indices);  continue;  end

                xvec2_cond = xvec2(indices);
                xvec1_cond = xvec1(indices);
                dx_cond    = xvec2_cond - xvec1_cond;

                % Transition moments  (without 1/2 factor; Noiray convention)
                D1(kk, k) = mean(dx_cond)              / dt_lag;
                D2(kk, k) = sum(dx_cond.^2) / length(dx_cond) / dt_lag;

            end
        end

        toc

        % Amplitude statistics for weighting and range selection
        stats_A = histcounts(x, edges);
        pdf_A   = histcounts(x, edges, 'Normalization', 'probability');
        Arange  = find(stats_A > N / (nbin * 50));

        % ------------------------------------------------------------------
        % 3. Extrapolate KM coefficients to τ→0
        % ------------------------------------------------------------------

        fprintf('  Extrapolating KM coefficients...\n')

        range_drift = find(tau_val >= 0.0 & tau_val <= 0.5);
        range_diff  = find(tau_val >= 0.0 & tau_val <= 0.5);

        tau_rest_drift = tau_val(range_drift);
        tau_rest_diff  = tau_val(range_drift);

        DA1_ext  = extrapolate_KM_3coeff( ...
            tau_rest_drift, D1(range_drift, Arange),  tmin, -0.005);
        DAA2_ext = extrapolate_KM( ...
            tau_rest_diff,  D2(range_diff,  Arange), tmin, -0.01);

        D1_KM = DA1_ext(1, :);
        D2_KM = DAA2_ext(1, :);

        % ------------------------------------------------------------------
        % 4. Parameter identification
        % ------------------------------------------------------------------

        pdf_ampl_rest = pdf_A(Arange);
        pdf_ampl_rest = pdf_ampl_rest / sum(pdf_ampl_rest);

        % Noise intensity from the 2nd KM coefficient.
        % As τ→0, M₂(τ,A) → 2·D₂(A) = 2c (additive noise → constant).
        % The PDF-weighted average of D2_KM estimates 2c; divide by 2 to
        % recover the physical diffusion coefficient c.
        c_hat     = sum(D2_KM .* pdf_ampl_rest) / sum(pdf_ampl_rest) / 2;
        Gamma_hat = c_hat * 4 * omega^2;   % [m²·s⁻³]  Γ = 4·ω²·c

        % Growth rate and saturation from least-squares fit to D₁(A)
        %   D₁(A) = ν·A − b·A³ + c/A
        % Note: element-wise operators ensure correct broadcasting with vector A.
        D1_model = @(c1, c2, a) c1*a + c2*a.^3 + Gamma_hat./(4*a.*omega^2);
        cost_fn  = @(coeff) sum( ...
            (D1_KM .* pdf_ampl_rest - D1_model(coeff(1), coeff(2), Abin(Arange)) .* pdf_ampl_rest).^2 );

        opts      = optimset('MaxFunEvals', 1000*2);
        coeff_fit = fminsearch(cost_fn, [20, 2e-5], opts);

        alpha_fit = coeff_fit(1);    % identified linear coefficient  (= ν)
        beta_fit  = coeff_fit(2);    % identified cubic coefficient   (= −b)

        nu_hat    = alpha_fit;
        b_hat     = -beta_fit;
        kappa_hat = 8 * b_hat;

        nu_hat_vec(ns) = nu_hat;
        b_hat_vec(ns)  = b_hat;
        c_hat_vec(ns)  = c_hat;

        fprintf('  ν    identified: %8.4f   true: %8.4f  [rad/s]\n',   nu_hat,    nu_true);
        fprintf('  κ    identified: %8.4f   true: %8.4f  [m⁻²s⁻¹]\n', kappa_hat, kappa);
        fprintf('  Γ    identified: %8.3g   true: %8.3g  [m²s⁻³]\n',  Gamma_hat, Gamma);

    end   % ns loop

    params_true = [nu_true, b_true, c_true];
    params_hat  = [nu_hat_vec, b_hat_vec, c_hat_vec];

    if save_data
        file = ['../DATA/nu_', num2str(nu_true), '_fromKMDirect.mat'];
        save(file, 'params_true', 'params_hat')
        fprintf('Data saved to: %s\n', file);
    end

end   % nk loop
