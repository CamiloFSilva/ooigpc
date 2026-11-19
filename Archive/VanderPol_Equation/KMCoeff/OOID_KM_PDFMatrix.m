% OOID_KM_PDFMatrix.m  (Van der Pol version)
% =========================================================================
% Kramers-Moyal (KM) identification of Van der Pol data
%
% DESCRIPTION
%   Loads pre-computed amplitude envelope data from vderPol_Solver.m and
%   identifies the system parameters using the conditional-transition-
%   moment approach (Kramers-Moyal coefficients).
%
%   The identification follows the same four-step procedure as the
%   Langevin KM script:
%     1. Build the conditional-jump probability density PA(τ, A_i, A_j)
%     2. Compute first (drift) and second (diffusion) transition moments
%     3. Extrapolate moments to τ→0 → 1st and 2nd KM coefficients
%     4. Fit the Langevin amplitude model to the drift coefficient
%
%   This script provides the KM benchmark for comparison with igPC
%   applied to Van der Pol simulation data (see OOID_igPC.m).
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
%   OOID_igPC.m       — igPC identification applied to same VdP data
%   vderPol_Solver.m  — generates the input data
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

nbin = 30;          % number of amplitude bins for conditional-jump PDFs

% Time-lag values for computing transition moments  [s]
tau_val = 0.005 : 0.01 : 0.1;

% Extrapolation settings
use_extrap = 1;     % 1 = exponential extrapolation (default; Boujo & Noiray 2017)
tmin = tau_val(1);  % shortest lag used in the extrapolation

% =========================================================================
%% SCAN PARAMETERS
% =========================================================================

NumSim = 1;    % number of independent realisations (must match vderPol_Solver.m)

% Growth rates to loop over — must match the saved DATA/ files  [rad/s]
nu_vec = [0];
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
        % Rename env_cell immediately to free the name 'A' for other use.

        file = ['../DATA/Data_nu', num2str(nu_true), '.mat'];
        load(file, 'A', 'dt', 'Gamma', 'wn')
        env_cell = A;  clear A

        ampl   = env_cell{ns};    % [m]  amplitude envelope from realisation ns
        dt_sim = dt;              % [s]  time step of the VdP simulation
        N      = length(ampl);
        omega  = wn;              % [rad/s]  natural angular frequency

        b_true = kappa / 8;
        c_true = Gamma / (4*omega^2);

        tk_vec = floor(tau_val / dt_sim);
        Fs     = 1 / dt_sim;
        nt     = N;

        % ------------------------------------------------------------------
        % 2. Conditional-jump probability density  PA(τ, A_i, A_j)
        % ------------------------------------------------------------------

        fprintf('  Building jump statistics...\n');  tic

        Amin = min(ampl);
        Amax = max(ampl);
        dA   = (Amax - Amin) / nbin;
        Abin = linspace(Amin + dA, Amax, nbin);

        ntau = length(tau_val);
        PA   = zeros(ntau, nbin, nbin);

        for its = 1:ntau
            shift = floor(tau_val(its) * Fs);

            for it = 1:nt - shift
                ib1 = floor((ampl(it)       - Amin) / dA);
                ib2 = floor((ampl(it+shift) - Amin) / dA);
                if ib1 == 0;  ib1 = 1;  end
                if ib2 == 0;  ib2 = 1;  end

                PA(its, ib1, ib2) = PA(its, ib1, ib2) + 1;
            end

            for ib1 = 1:nbin
                cnt = sum(squeeze(PA(its, ib1, :)));
                if cnt > 0
                    PA(its, ib1, :) = PA(its, ib1, :) / cnt;
                end
                if its == 1
                    stats_A(ib1) = cnt;
                end
            end
        end

        toc

        % ------------------------------------------------------------------
        % 3. Transition moments  M₁(τ, A_i)  and  M₂(τ, A_i)
        % ------------------------------------------------------------------

        fprintf('  Computing transition moments...\n');  tic

        DA1t  = zeros(ntau, nbin);
        DAA2t = zeros(ntau, nbin);

        for its = 1:ntau
            for ib1 = 1:nbin
                A_ref = Abin(ib1);
                for ib2 = 1:nbin
                    A_del = Abin(ib2);
                    p_ij  = squeeze(PA(its, ib1, ib2));
                    jump  = A_del - A_ref;

                    DA1t (its, ib1) = DA1t (its, ib1) + jump   * p_ij;
                    DAA2t(its, ib1) = DAA2t(its, ib1) + jump^2 * p_ij;
                end
            end

            DA1t (its, :) = DA1t (its, :) / tau_val(its);
            DAA2t(its, :) = DAA2t(its, :) / tau_val(its);
        end

        toc

        stats_A = histcounts(ampl, [Amin, Abin]);
        Arange  = find(stats_A > nt / (nbin * 50));

        % ------------------------------------------------------------------
        % 4. Extrapolate transition moments to τ→0 → KM coefficients
        % ------------------------------------------------------------------

        fprintf('  Extrapolating KM coefficients...\n')

        range_drift = find(tau_val >= 0.0 & tau_val <= 0.5);
        range_diff  = find(tau_val >= 0.0 & tau_val <= 0.5);

        tau_rest_drift = tau_val(range_drift);
        tau_rest_diff  = tau_val(range_diff);

        DA1_ext  = extrapolate_KM_3coeff( ...
            tau_rest_drift, DA1t(range_drift, Arange),  tmin, -0.005);
        DAA2_ext = extrapolate_KM( ...
            tau_rest_diff,  DAA2t(range_diff, Arange), tmin, -0.01);

        D1_KM = DA1_ext(1, :);
        D2_KM = DAA2_ext(1, :);

        % ------------------------------------------------------------------
        % 5. Parameter identification
        % ------------------------------------------------------------------

        pdf_ampl      = histcounts(ampl, [Amin, Abin], 'Normalization', 'probability');
        pdf_ampl_rest = pdf_ampl(Arange);
        pdf_ampl_rest = pdf_ampl_rest / sum(pdf_ampl_rest);

        % Noise intensity from the 2nd KM coefficient.
        % As τ→0, M₂(τ,A) → 2·D₂(A) = 2c (constant), so the PDF-weighted
        % average of D2_KM estimates 2c.  Divide by 2 to recover c.
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
        file = ['../DATA/nu_', num2str(nu_true), '_fromKMCoeffs.mat'];
        save(file, 'params_true', 'params_hat')
        fprintf('Data saved to: %s\n', file);
    end

end   % nk loop
