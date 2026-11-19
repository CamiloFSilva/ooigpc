% OOID_KM_PDFMatrix.m  (Experiments version)
% =========================================================================
% Kramers-Moyal (KM) identification from experimental data
% — PDF matrix method —
%
% DESCRIPTION
%   Loads pre-computed amplitude envelopes from readExpData.m and identifies
%   the Langevin amplitude-equation parameters (ν, b, c) for each operating
%   condition using the conditional-transition-moment approach (Kramers-Moyal
%   PDF matrix).
%
%   The identification follows four steps per operating condition:
%     1. Build the conditional-jump probability density PA(τ, A_i, A_j)
%     2. Compute first (drift) and second (diffusion) transition moments
%     3. Extrapolate moments to τ→0  →  1st and 2nd KM coefficients
%     4. Fit the Langevin amplitude model to the drift coefficient
%
%   Unlike the simulation scripts (VanderPol_Equation, Langevin_Equation),
%   there is no ensemble of realisations and no ground-truth parameter.
%   Each operating condition (index ns = 1…5) corresponds to a different
%   equivalence ratio / heat-release rate from Noiray & Denisov (2017).
%
% INPUT
%   ../DATA/envelopeRef_B<bandwidth>Exp.mat   — produced by readExpData.m
%       Variables:  A        — {1×5} amplitude envelopes  [Pa]
%                   dt_vec   — [1×5] sampling time steps  [s]
%                   fn_vec   — [1×5] dominant frequencies [Hz]
%
% OUTPUT  (one file per operating condition)
%   ../DATA/ns_<ns>_fromKMCoeffs.mat
%       Variables:  params_hat — [1×3]  (nu_hat, b_hat, c_hat)
%                   ns         — scalar  operating-condition index
%                   wn         — scalar  angular frequency  [rad/s]
%
% CONVENTIONS
%   The amplitude SDE (Itô, slow-flow) is:
%       dA = (ν·A − b·A³ + c/A) dt + √(2c) dW
%   where  b = κ/8  and  c = Γ/(4·ω₀²).
%
%   Factor of 2: M₂(τ,A)/τ → 2·D₂(A) = 2c as τ→0.
%   The PDF-weighted average DAAm therefore estimates 2c; we divide by 2
%   to recover c_hat, then Γ_hat = 4·ω₀²·c_hat.
%
% METHOD REFERENCES
%   Siegert, Friedrich, Peinke (1998).  Phys. Lett. A 243, 275-280.
%   Noiray & Schuermans (2013).  Int. J. Non-linear Mech. 50, 152-163.
%   Boujo & Noiray (2017).  Proc. R. Soc. A 473, 20160894.
%
% DATA REFERENCE
%   Noiray, N. & Denisov, A. (2017).  Proc. Combust. Inst. 36, 3843-3850.
%
% SEE ALSO
%   readExpData.m            — generates the input envelope file
%   OOID_KM_Direct.m         — direct conditional-binning alternative
%   ../igPC/OOID_igPC.m      — igPC identification of same data
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

bandwidth = 40;    % [Hz]  band-pass filter half-width; must match readExpData.m
%bandwidth = 10;   % alternative

ns_vec = 1:5;      % operating conditions to process (1 = lowest, 5 = highest ER)
%ns_vec = [3 4 5]; % process a subset only

save_data = true;  % true  → save params_hat to ../DATA/
                   % false → run without writing .mat files

% =========================================================================
%% KM ESTIMATION PARAMETERS
% =========================================================================

nbin = 30;           % number of amplitude bins for the conditional-jump PDFs

% Time-lag values for computing transition moments  [s]
%   (this set produces the best results for Noiray & Denisov data)
tau_val = 0.005 : 0.01 : 0.1;

% Extrapolation settings
use_extrap = 1;      % 1 = exponential extrapolation (Boujo & Noiray 2017)
tmin = tau_val(1);   % shortest lag used in the extrapolation

% =========================================================================
%% LOAD ENVELOPE DATA
% =========================================================================

filename = ['../DATA/envelope_BW', num2str(bandwidth), 'Exp.mat'];
load(filename, 'A', 'dt_vec', 'fn_vec')

% =========================================================================
%% IDENTIFICATION LOOP (over operating conditions)
% =========================================================================

for nn = 1:length(ns_vec)

    ns = ns_vec(nn);
    fprintf('\n=== Operating condition %d / %d ===\n', ns, max(ns_vec));

    % ------------------------------------------------------------------
    % 1. Extract envelope data for this condition
    % ------------------------------------------------------------------

    ampl   = A{ns};          % [Pa]  amplitude envelope time series
    ampl   = ampl(:);        % ensure column vector — defensive normalisation
                             % against row-vector storage in the .mat file
    dt_sim = dt_vec(ns);     % [s]   sampling time step
    fn     = fn_vec(ns);     % [Hz]  dominant frequency
    wn     = 2*pi*fn;        % [rad/s]  angular frequency
    N      = length(ampl);

    ntau   = length(tau_val);
    Fs     = 1/dt_sim;
    nt     = N;
    tk_vec = floor(tau_val/dt_sim);

    % ------------------------------------------------------------------
    % 2. Conditional-jump probability density  PA(τ, A_i, A_j)
    %    PA(its, ib1, ib2) = P(A(t+τ) ∈ bin_ib2 | A(t) ∈ bin_ib1)
    % ------------------------------------------------------------------

    fprintf('  Building jump statistics...\n');  tic

    Amin = min(ampl);
    Amax = max(ampl);
    dA   = (Amax - Amin)/nbin;
    Abin = linspace(Amin + dA, Amax, nbin);   % bin centres

    PA = zeros(ntau, nbin, nbin);

    for its = 1:ntau
        shift = floor(tau_val(its)*Fs);

        for it = 1:nt - shift
            ib1 = floor((ampl(it)       - Amin)/dA);
            ib2 = floor((ampl(it+shift) - Amin)/dA);
            if ib1 == 0;  ib1 = 1;  end
            if ib2 == 0;  ib2 = 1;  end

            PA(its, ib1, ib2) = PA(its, ib1, ib2) + 1;
        end

        for ib1 = 1:nbin
            cnt = sum(squeeze(PA(its, ib1, :)));
            if cnt > 0
                PA(its, ib1, :) = PA(its, ib1, :)/cnt;
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

        DA1t (its, :) = DA1t (its, :)/tau_val(its);
        DAA2t(its, :) = DAA2t(its, :)/tau_val(its);
    end

    toc

    stats_A = histcounts(ampl, [Amin, Abin]);
    pdf_A   = histcounts(ampl, [Amin, Abin], 'Normalization', 'probability');

    % Keep only bins with sufficient statistics (≥ 1/50 of uniform density)
    Arange = find(stats_A > nt/(nbin*50));

    % ------------------------------------------------------------------
    % 4. Extrapolate transition moments to τ→0  →  KM coefficients
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

    D1_KM = DA1_ext(1, :);     % 1st KM coefficient: drift  [Pa/s]
    D2_KM = DAA2_ext(1, :);    % 2nd KM coefficient: estimates 2c  [Pa²/s]

    % ------------------------------------------------------------------
    % 5. Parameter identification
    % ------------------------------------------------------------------

    fprintf('  Identifying parameters...\n')

    pdf_ampl_rest = pdf_A(Arange);
    pdf_ampl_rest = pdf_ampl_rest/sum(pdf_ampl_rest);

    % Noise intensity from the 2nd KM coefficient
    %   D₂(A) = c  (constant, additive noise)
    %   M₂(τ,A)/τ → 2·D₂(A) = 2c  as τ→0
    %   DAAm = PDF-weighted mean of D2_KM ≈ 2c  →  divide by 2 to get c
    weight = pdf_ampl_rest.^1;
    DAAm   = sum(D2_KM .* weight)/sum(weight);  % ≈ 2c

    c_hat     = DAAm/2;                  % [Pa²/s]  physical diffusion coefficient
    Gamma_hat = c_hat * 4 * wn^2;        % [Pa²/s³] noise intensity  Γ = 4·ω₀²·c

    % Growth rate and saturation from least-squares fit to D₁(A)
    %   D₁(A) = ν·A − b·A³ + c/A
    %   coeff(1) = ν,  coeff(2) = −b  (note sign)
    D1_model = @(c1, c2, a) c1.*a + c2.*a.^3 + Gamma_hat./(4.*a.*wn^2);
    cost_fn  = @(coeff) sum( ...
        (D1_KM .* pdf_ampl_rest - D1_model(coeff(1), coeff(2), Abin(Arange)) .* pdf_ampl_rest).^2 );

    opts      = optimset('MaxFunEvals', 1000*2);
    coeff_fit = fminsearch(cost_fn, [20, 2e-5], opts);

    alpha_fit = coeff_fit(1);    % identified linear coefficient  (= ν)
    beta_fit  = coeff_fit(2);    % identified cubic coefficient   (= −b)

    nu_hat    = alpha_fit;
    b_hat     = -beta_fit;
    kappa_hat = 8*b_hat;

    fprintf('  ν    identified: %8.4f  [rad/s]\n',   nu_hat);
    fprintf('  κ    identified: %8.4f  [m⁻²s⁻¹]\n', kappa_hat);
    fprintf('  c    identified: %8.3g  [Pa²/s]\n',   c_hat);
    fprintf('  Γ    identified: %8.3g  [Pa²/s³]\n',  Gamma_hat);

    params_hat = [nu_hat, b_hat, c_hat];

    % ------------------------------------------------------------------
    % 6. Save results
    % ------------------------------------------------------------------

    if save_data
        file = ['../DATA/ns_', num2str(ns), '_fromKMMatrix.mat'];
        save(file, 'params_hat', 'ns', 'wn')
        fprintf('  Saved to: %s\n', file);
    end

end   % ns loop
