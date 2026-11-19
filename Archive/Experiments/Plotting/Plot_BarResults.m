% Plot_BarResults.m  (Experiments version)
% =========================================================================
% Grouped bar charts comparing identified Langevin parameters
% — igPC vs. KM method, one figure per operating condition —
% → paper Fig. 12
%
% DESCRIPTION
%   For each operating condition (ns = 1…5) loads the identified parameter
%   vectors from the igPC and KM output files and plots a grouped bar chart
%   with three parameter groups on the x-axis:
%     (1)  ν  — linear growth rate               [rad/s]
%     (2)  κ  — nonlinear saturation  κ = 8·b
%     (3)  c  — diffusion coefficient
%   Each group has two bars: KM (red) and igPC (blue).
%
% INPUT
%   ../DATA/ns_<ns>_<CASE>_<BIN>_fromigPC.mat
%   ../DATA/ns_<ns>_fromKMCoeffs.mat          — if KM_SOURCE = 'PDFMatrix'
%   ../DATA/ns_<ns>_fromKMDirect.mat          — if KM_SOURCE = 'Direct'
%
% OUTPUT  (one PDF per operating condition)
%   ./Fig12_ns<ns>_ExpBars_<KM_SOURCE>.pdf
%
% DATA REFERENCE
%   Noiray, N. & Denisov, A. (2017).  Proc. Combust. Inst. 36, 3843-3850.
%
% SEE ALSO
%   OOID_igPC.m          — igPC identification
%   OOID_KM_PDFMatrix.m  — KM PDF-matrix identification
%   OOID_KM_Direct.m     — KM direct-binning identification
%   Plot_AmplitudePDF.m  — paper Fig. 11
% =========================================================================

clear all
close all
clc

% =========================================================================
%% USER OPTIONS
% =========================================================================

%ns_vec = 1:5;      % operating conditions to include
ns_vec = [3,5];      % operating conditions to include

% igPC settings — must match the files produced by OOID_igPC.m
CASE = 'CorrectionTwoLevels';   % 'NoCorrection' | 'CorrectionTwoLevels'
BIN  = 'NOP';                   % 'NOP'           | 'YES'

% KM source for the benchmark comparison
KM_SOURCE = 'Direct';   % 'PDFMatrix' | 'Direct'

save_figs = true;  % true  → save one PDF per condition to current folder
                   % false → display without saving

% =========================================================================
%% COLLECT IDENTIFIED PARAMETERS
% =========================================================================

Nc = length(ns_vec);

nu_igPC    = NaN(1, Nc);
kappa_igPC = NaN(1, Nc);
c_igPC     = NaN(1, Nc);

nu_KM      = NaN(1, Nc);
kappa_KM   = NaN(1, Nc);
c_KM       = NaN(1, Nc);

for nn = 1:Nc

    ns = ns_vec(nn);

    % --- igPC ---
    file_igPC = ['../DATA/ns_', num2str(ns), '_', CASE, '_', BIN, '_fromigPC.mat'];
    if exist(file_igPC, 'file')
        tmp = load(file_igPC, 'params_hat');
        nu_igPC(nn)    = tmp.params_hat(1);
        kappa_igPC(nn) = 8 * tmp.params_hat(2);
        c_igPC(nn)     = tmp.params_hat(3);
    else
        warning('File not found: %s', file_igPC)
    end

    % --- KM ---
    switch KM_SOURCE
        case 'PDFMatrix'
            file_KM = ['../DATA/ns_', num2str(ns), '_fromKMMatrix.mat'];
        case 'Direct'
            file_KM = ['../DATA/ns_', num2str(ns), '_fromKMDirect.mat'];
        otherwise
            error('KM_SOURCE must be ''PDFMatrix'' or ''Direct''.')
    end

    if exist(file_KM, 'file')
        tmp = load(file_KM, 'params_hat');
        nu_KM(nn)    = tmp.params_hat(1);
        kappa_KM(nn) = 8 * tmp.params_hat(2);
        c_KM(nn)     = tmp.params_hat(3);
    else
        warning('File not found: %s', file_KM)
    end

end

% =========================================================================
%% PLOT  —  one figure per operating condition
%
%   Each figure: grouped bar chart with 3 parameter groups × 2 bars
%     group 1: ν      (linear growth rate)
%     group 2: κ = 8b (nonlinear saturation)
%     group 3: c      (diffusion coefficient)
%   Bar 1 (red):  KM
%   Bar 2 (blue): igPC
% =========================================================================

cKM   = [0.75  0.15  0.15];   % dark red  — KM
cigPC = [0.15  0.35  0.75];   % dark blue — igPC
bar_w = 0.5;                  % relative bar width

for nn = 1:Nc

    ns = ns_vec(nn);

    % data(1,:) = KM   values: [ν, κ, c]
    % data(2,:) = igPC values: [ν, κ, c]
    % bar(data', 'grouped') → 3 groups on x-axis, 2 bars per group
    data = [nu_KM(nn),    kappa_KM(nn),    c_KM(nn);
            nu_igPC(nn),  kappa_igPC(nn),  c_igPC(nn)];

    fig_h = figure(100 + ns);
    clf
    set(fig_h, 'Units', 'centimeters', 'Position', [2 2 10 10])

    b = bar(data', bar_w, 'grouped');
    b(1).FaceColor = cKM;
    b(2).FaceColor = cigPC;
    b(1).EdgeColor = 'none';
    b(2).EdgeColor = 'none';

    set(gca, 'XTick', [1 2 3], 'FontSize', 20)
    xticklabels({'\nu', '\kappa', 'c'})
    legend({'KM', 'igPC'}, 'Location', 'northwest', 'FontSize', 14)

    grid on
    ax = gca;
    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.TickDir = 'out';

    title(['Operating condition  ns = ', num2str(ns)], 'FontSize', 14)

    % ---- Save ----
    if save_figs
        out_file = ['./Fig12_ns', num2str(ns), '_ExpBars_', KM_SOURCE, '.pdf'];
        set(fig_h, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                   'PaperPosition', [0 0 10 10])
        saveas(fig_h, out_file)
        fprintf('Saved: %s\n', out_file)
    end

end
