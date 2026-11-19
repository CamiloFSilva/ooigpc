% Plot_AmplitudePDF.m  (Experiments version)
% =========================================================================
% Marginal amplitude PDFs for all 5 operating conditions
% — histogram from measured data vs. theoretical Fokker-Planck density —
% → paper Fig. 12
%
% DESCRIPTION
%   For each operating condition (ns = 1…5) loads the stationary amplitude
%   time series  x_stat  and the identified parameters  params_hat  from
%   the igPC output file, then plots:
%     (a) a normalised histogram of  x_stat
%     (b) the theoretical Fokker-Planck stationary density with identified
%         parameters overlaid as a smooth curve
%
%   The theoretical stationary PDF of the Itô SDE
%       dA = (ν·A − b·A³ + c/A) dt + √(2c) dW
%   is obtained by solving the Fokker-Planck equation (additive noise):
%       p(A) = C · A · exp(ν·A²/(2c) − b·A⁴/(4c)),   A ≥ 0
%   where C is determined by normalisation on [0, ∞).
%
%   The five panels are arranged in a single row (1 × 5 subplots).
%
% INPUT
%   ../DATA/ns_<ns>_<CASE>_<BIN>_fromigPC.mat
%       Variables loaded:  params_hat — [1×3]  (nu_hat, b_hat, c_hat)
%                          x_stat     — amplitude time series  [Pa]
%
% OUTPUT
%   ./Fig12_ExpAmplitudePDF.pdf
%
% NOTES
%   • For conditions deep in the noise-driven regime (ν ≤ 0) the PDF peaks
%     near zero and is close to a Rayleigh distribution.
%   • For conditions in the oscillatory regime (ν > 0) the PDF peaks at
%     the limit-cycle amplitude A_eq = √(ν/b).
%
% DATA REFERENCE
%   Noiray, N. & Denisov, A. (2017).  Proc. Combust. Inst. 36, 3843-3850.
%
% SEE ALSO
%   OOID_igPC.m         — produces the input data files
%   Plot_BarResults.m   — paper Fig. 11
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

ns_vec = 1:5;      % operating conditions to plot

% igPC settings — must match the files produced by OOID_igPC.m
CASE = 'CorrectionTwoLevels';   % 'NoCorrection' | 'CorrectionTwoLevels'
BIN  = 'NOP';                   % 'NOP'           | 'YES'

n_bins     = 35;   % number of histogram bins (matches original Plot_Results_Histo.m)
n_pdf_pts  = 2000; % points used to evaluate the theoretical PDF

save_figs = true;  % true  → save PDF to current folder

% Axis limits (set to [] for automatic)
% For the Noiray & Denisov data the amplitude is in Pa; adjust as needed.
x_lim_A   = [];    % e.g. [0, 10]  — amplitude axis limit  [Pa]
y_lim_pdf = [];    % e.g. [0, 0.6] — PDF axis limit  [Pa⁻¹]

% =========================================================================
%% LOAD DATA AND PLOT
% =========================================================================

Nc = length(ns_vec);

fig12 = figure(12);
set(fig12, 'Units', 'centimeters', 'Position', [2 2 24 7])

for nn = 1:Nc

    ns = ns_vec(nn);

    % ------------------------------------------------------------------
    % Load igPC output
    % ------------------------------------------------------------------

    file_igPC = ['../DATA/ns_', num2str(ns), '_', CASE, '_', BIN, '_fromigPC.mat'];

    if ~exist(file_igPC, 'file')
        warning('File not found: %s — skipping condition %d', file_igPC, ns)
        continue
    end

    load(file_igPC, 'params_hat', 'x_stat')

    nu_hat = params_hat(1);
    b_hat  = params_hat(2);
    c_hat  = params_hat(3);

    % ------------------------------------------------------------------
    % Theoretical Fokker-Planck stationary PDF
    %   p(A) ∝ A · exp(ν·A²/(2c) − b·A⁴/(4c)),   A ≥ 0
    %
    %   Derivation: for the Itô SDE with additive noise  g = √(2c),
    %   the FPK stationary solution with reflecting BC at A=0 is
    %     p ∝ g⁻²·exp(2∫f/g² dA)
    %   where f(A) = ν·A − b·A³ + c/A.
    %   Integrating: ∫f/(g²) dA = (ν·A² − b·A⁴/2)/4c + ln(A)/2 → p ∝ A·exp(...)
    % ------------------------------------------------------------------

    A_max    = max(x_stat) * 1.3;
    A_range  = linspace(0, A_max, n_pdf_pts);

    exponent = nu_hat * A_range.^2 / (2*c_hat) - b_hat * A_range.^4 / (4*c_hat);

    % Guard against numerical overflow (exponent > 700 → Inf)
    exponent = min(exponent, 700);

    p_unnorm    = A_range .* exp(exponent);
    p_unnorm(1) = 0;                           % p(0) = 0 by boundary condition
    norm_const  = trapz(A_range, p_unnorm);

    if norm_const <= 0 || isnan(norm_const)
        warning('PDF normalisation failed for ns = %d; theory curve omitted.', ns)
        p_theory = NaN(size(A_range));
    else
        p_theory = p_unnorm / norm_const;
    end

    % ------------------------------------------------------------------
    % Subplot
    % ------------------------------------------------------------------

    ax = subplot(1, Nc, nn);

    % Histogram
    histogram(x_stat, n_bins, ...
        'FaceColor',    [0.5  0.5  0.5], ...
        'EdgeColor',    'none', ...
        'Normalization', 'pdf')

    hold on

    % Theoretical PDF
    plot(A_range, p_theory, '-', ...
        'Color',     [0.15  0.35  0.75], ...
        'LineWidth',  2.0)

    hold off

    xlabel('A  [Pa]', 'FontSize', 12)
    if nn == 1
        ylabel('p(A)  [Pa^{-1}]', 'FontSize', 12)
    end
    title(['ns = ', num2str(ns)], 'FontSize', 12)

    if ~isempty(x_lim_A)
        xlim(x_lim_A)
    else
        xlim([0, A_max])
    end
    if ~isempty(y_lim_pdf)
        ylim(y_lim_pdf)
    end

    % Annotate identified parameters inside each panel
    text(0.60, 0.85, ...
        {sprintf('\\nu  = %.1f', nu_hat), ...
         sprintf('\\kappa = %.1e', 8*b_hat), ...
         sprintf('c  = %.2e', c_hat)}, ...
        'Units',          'normalized', ...
        'FontSize',        9, ...
        'VerticalAlignment', 'top')

    set(ax, 'FontSize', 11, 'TickDir', 'out')
    grid on;  ax.YGrid = 'on';  ax.XGrid = 'off';

end

sgtitle('Marginal amplitude PDFs — experiment vs. FPK stationary density (igPC)', ...
        'FontSize', 13, 'FontWeight', 'bold')

% =========================================================================
%% SAVE
% =========================================================================

if save_figs
    out_file = './Fig12_ExpAmplitudePDF.pdf';
    set(fig12, 'PaperUnits', 'centimeters', 'PaperSize', [24 7], ...
               'PaperPosition', [0 0 24 7])
    saveas(fig12, out_file)
    fprintf('Saved: %s\n', out_file)
end
