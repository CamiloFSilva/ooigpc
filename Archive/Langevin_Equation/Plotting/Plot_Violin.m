% =========================================================================
% Plot_Violin.m  —  Violin plots of identified parameters
%                    vs. true values: igPC vs. KM method
%
% DESCRIPTION
%   Loads ensemble identification results (from OOID_igPC.m and
%   OOID_KM_PDFMatrix.m) and builds violin plots comparing the two methods
%   for each growth rate nu in nu_vec.  Quantile markers (Q1, median, Q3)
%   are overlaid on each violin.  The true parameter value is shown as a
%   horizontal black reference line.
%
%   Three figures are produced (one per identified parameter):
%       Figure 100  — nu    (linear growth rate)        → paper Fig. 6(a)
%       Figure 200  — kappa (nonlinear damping = 8·b)   → paper Fig. 6(b)
%       Figure 300  — c     (diffusion coefficient)     → paper Fig. 6(c)
%
% INPUT FILES  (relative path from this script's location)
%   ../DATA/nu_<nu>_fromKMCoeffs.mat
%       Variables: params_true [1×3], params_hat [NumSim×3]
%                  columns: (nu, b, c)
%   ../DATA/nu_<nu>_<CASE>_fromigPC.mat
%       Variables: params_true [1×3], params_hat [NumSim×3]
%
%   NOTE: legacy files using ideal_param/recov_param are handled
%   automatically via a compatibility check.
%
% DEPENDENCIES
%   violinplot()  — MATLAB Statistics Toolbox (R2023b+) or File Exchange
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
% =========================================================================

clear all

% =========================================================================
%% USER OPTIONS
% =========================================================================

CASE = 'CorrectionTwoLevels';  % Variance-correction mode used to generate
%CASE = 'NoCorrection';         % the igPC DATA files; must match the CASE
%CASE = 'CorrectionOneLevel';   % used when running OOID_igPC.m

nu_vec = [0, 4, 8, 12, 16, 20, 24];  % Growth-rate cases to plot  [rad/s]
%nu_vec = [0, 2, 4, 6, 8, 10, 12];   % Alternative finer grid

save_figs = true;        % true  → save figures as PDF to current folder
                         % false → display only

% =========================================================================
%% PHYSICAL PARAMETERS  (reference values for the true-value lines)
% =========================================================================

kappa = 3.4;
Gamma = 1e7;
omega = 2*pi*120;

b_true = kappa / 8;
c_true = Gamma / (4 * omega^2);

% =========================================================================
%% LOAD AND PROCESS IDENTIFICATION RESULTS
% =========================================================================

% Pre-allocate concatenated data arrays for violin construction
DatagPC_nu = [];  DataKM_nu = [];
DatagPC_b  = [];  DataKM_b  = [];
DatagPC_c  = [];  DataKM_c  = [];

NgPC_nu = zeros(1, length(nu_vec));
NgPC_b  = zeros(1, length(nu_vec));
NgPC_c  = zeros(1, length(nu_vec));
NKM_nu  = zeros(1, length(nu_vec));
NKM_b   = zeros(1, length(nu_vec));
NKM_c   = zeros(1, length(nu_vec));

for nn = 1 : length(nu_vec)

    nu = nu_vec(nn);

    % --- Load KM results (new and legacy naming convention) ---------------
    fileKM   = ['../DATA/nu_', num2str(nu), '_fromKMCoeffs.mat'];
    %fileKM   = ['../DATA/nu_', num2str(nu), '_fromKMCoeffs_PDFMatrix.mat'];
    loadedKM = load(fileKM);
    if isfield(loadedKM, 'params_hat')
        params_hat_KM  = loadedKM.params_hat;
        params_true_KM = loadedKM.params_true;
    else
        params_hat_KM  = loadedKM.recov_param;
        params_true_KM = loadedKM.ideal_param;
    end

    % --- Load igPC results ------------------------------------------------
    filegPC   = ['../DATA/nu_', num2str(nu), '_', CASE, '_fromigPC.mat'];
    loadedgPC = load(filegPC);
    if isfield(loadedgPC, 'params_hat')
        params_hat_gPC  = loadedgPC.params_hat;
        params_true_gPC = loadedgPC.params_true;
    else
        params_hat_gPC  = loadedgPC.recov_param;
        params_true_gPC = loadedgPC.ideal_param;
    end

    % --- igPC ensemble: remove unfilled slots, then outliers jointly ------
    % Outliers are identified from c_data_gPC (the diffusion estimate that
    % feeds directly into the igPC formulas).  The same row indices are
    % removed from nu and b so that all three vectors stay the same length.
    nu_data_gPC = params_hat_gPC(:, 1);
    b_data_gPC  = params_hat_gPC(:, 2);
    c_data_gPC  = params_hat_gPC(:, 3);

    valid_gPC   = (nu_data_gPC ~= 0) & (b_data_gPC ~= 0) & (c_data_gPC ~= 0);
    nu_data_gPC = nu_data_gPC(valid_gPC);
    b_data_gPC  =  b_data_gPC(valid_gPC);
    c_data_gPC  =  c_data_gPC(valid_gPC);

    out_gPC     = isoutlier(c_data_gPC);
    nu_data_gPC = nu_data_gPC(~out_gPC);
    b_data_gPC  =  b_data_gPC(~out_gPC);
    c_data_gPC  =  c_data_gPC(~out_gPC);

    % --- KM ensemble: remove NaNs, then outliers jointly -----------------
    % Same logic: outlier mask is built from c_data_KM and applied to all.
    nu_data_KM = real(params_hat_KM(:, 1));
    b_data_KM  = real(params_hat_KM(:, 2));
    c_data_KM  = real(params_hat_KM(:, 3));

    valid_KM   = ~isnan(nu_data_KM) & ~isnan(b_data_KM) & ~isnan(c_data_KM);
    nu_data_KM = nu_data_KM(valid_KM);
    b_data_KM  =  b_data_KM(valid_KM);
    c_data_KM  =  c_data_KM(valid_KM);

    out_KM     = isoutlier(c_data_KM);
    nu_data_KM = nu_data_KM(~out_KM);
    b_data_KM  =  b_data_KM(~out_KM);
    c_data_KM  =  c_data_KM(~out_KM);

    % --- Accumulate for violin construction -------------------------------
    NgPC_nu(nn) = length(nu_data_gPC);
    NgPC_b(nn)  = length( b_data_gPC);
    NgPC_c(nn)  = length( c_data_gPC);
    DatagPC_nu  = [DatagPC_nu;  nu_data_gPC];
    DatagPC_b   = [DatagPC_b;    b_data_gPC];
    DatagPC_c   = [DatagPC_c;    c_data_gPC];

    NKM_nu(nn)  = length(nu_data_KM);
    NKM_b(nn)   = length( b_data_KM);
    NKM_c(nn)   = length( c_data_KM);
    DataKM_nu   = [DataKM_nu;  nu_data_KM];
    DataKM_b    = [DataKM_b;    b_data_KM];
    DataKM_c    = [DataKM_c;    c_data_KM];

end

% =========================================================================
%% BUILD CATEGORICAL GROUP LABELS FOR violinplot()
%   Each sample is assigned a string label matching its nu_vec value.
%   The order of nu_labels is preserved as the x-axis category order.
% =========================================================================

nu_labels = string(nu_vec);   % e.g. ["0","4","8","12","16","20","24"]

GroupgPC_nu = build_groups(NgPC_nu, nu_labels);
GroupgPC_b  = build_groups(NgPC_b,  nu_labels);
GroupgPC_c  = build_groups(NgPC_c,  nu_labels);
GroupKM_nu  = build_groups(NKM_nu,  nu_labels);
GroupKM_b   = build_groups(NKM_b,   nu_labels);
GroupKM_c   = build_groups(NKM_c,   nu_labels);

% =========================================================================
%% FIGURE 100 — nu (linear growth rate)      [→ paper Fig. 6(a)]
%   The true value varies per group, so it is drawn as a short horizontal
%   tick at each x-position rather than a single yline.
% =========================================================================

figure(100); clf
tbl_gPC = table(GroupgPC_nu, DatagPC_nu, 'VariableNames', ["nu", "gPC"]);
tbl_KM  = table(GroupKM_nu,  DataKM_nu,  'VariableNames', ["nu", "KM"]);

violinplot(tbl_gPC, "nu", "gPC");  hold on
violinplot(tbl_KM,  "nu", "KM");
overlay_quantiles(tbl_gPC, 'gPC', nu_labels, 'b');
overlay_quantiles(tbl_KM,  'KM',  nu_labels, 'r');

for i = 1 : length(nu_vec)
    plot([i-0.5, i+0.5], [nu_vec(i), nu_vec(i)], 'k-', 'LineWidth', 1)
end
hold off

xticks([])
ylim([-2 32])
ylabel('\nu  [1/s]')
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 10], ...
    'PaperPosition', [0 0 20 10])
if save_figs
    saveas(gcf, ['./param_nu_', CASE, '.pdf'])
end

% =========================================================================
%% FIGURE 200 — kappa (nonlinear damping = 8·b)   [→ paper Fig. 6(b)]
% =========================================================================

figure(200); clf
tbl_gPC = table(GroupgPC_b, 8*DatagPC_b, 'VariableNames', ["nu", "gPC"]);
tbl_KM  = table(GroupKM_b,  8*DataKM_b,  'VariableNames', ["nu", "KM"]);

violinplot(tbl_gPC, "nu", "gPC");  hold on
violinplot(tbl_KM,  "nu", "KM");
overlay_quantiles(tbl_gPC, 'gPC', nu_labels, 'b');
overlay_quantiles(tbl_KM,  'KM',  nu_labels, 'r');
hold off

xticks([])
yline(kappa, 'k-', 'LineWidth', 1)
ylim([0 10])
ylabel('\kappa')
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 10], ...
    'PaperPosition', [0 0 20 10])
if save_figs
    saveas(gcf, ['./param_b_', CASE, '.pdf'])
end

% =========================================================================
%% FIGURE 300 — c (diffusion coefficient)         [→ paper Fig. 6(c)]
% =========================================================================

figure(300); clf
tbl_gPC = table(GroupgPC_c, DatagPC_c, 'VariableNames', ["nu", "gPC"]);
tbl_KM  = table(GroupKM_c,  DataKM_c,  'VariableNames', ["nu", "KM"]);

violinplot(tbl_gPC, "nu", "gPC");  hold on
violinplot(tbl_KM,  "nu", "KM");
overlay_quantiles(tbl_gPC, 'gPC', nu_labels, 'b');
overlay_quantiles(tbl_KM,  'KM',  nu_labels, 'r');
hold off

xticks([])
yline(c_true, 'k-', 'LineWidth', 1)
ylim([0 6])
ylabel('c  [a.u.]')
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [20 10], ...
    'PaperPosition', [0 0 20 10])
if save_figs
    saveas(gcf, ['./param_c_', CASE, '.pdf'])
end

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function grp = build_groups(N_per_group, labels)
% BUILD_GROUPS  Build a categorical label vector for violinplot().
%
%   Each sample in group i receives the label labels(i).  The category
%   order in the output follows the order of labels, so violinplot()
%   displays groups left-to-right in that order regardless of the
%   lexicographic ordering of the label strings.
%
%   Uses repelem() for a direct, loop-free construction:
%     repelem(["0","4","8"], [3 2 4]) → ["0","0","0","4","4","8","8","8","8"]
%
%   Input:
%     N_per_group  — [1×nGroups] count of samples in each group
%     labels       — [1×nGroups] string labels (one per group)
%   Output:
%     grp          — categorical column vector, length sum(N_per_group)
    grp = categorical(repelem(labels(:), N_per_group(:)), labels);
end

function overlay_quantiles(tbl, varName, nu_labels, col)
% OVERLAY_QUANTILES  Overlay Q1, median, Q3 markers on a violin plot.
%
%   The x-position of each group is its index in nu_labels (1, 2, 3, …),
%   which matches the order that violinplot() uses for the category axis.
%   Groups are looked up by label (not by unique()), so the display order
%   is guaranteed to follow nu_labels regardless of lexicographic sorting.
%
%   Input:
%     tbl       — table with 'nu' (categorical) and varName columns
%     varName   — name of the data column (string)
%     nu_labels — string array of labels in the desired display order
%     col       — colour character ('b' for igPC, 'r' for KM)
    for i = 1 : length(nu_labels)
        mask = (tbl.nu == nu_labels(i));
        data = tbl.(varName)(mask);
        if isempty(data);  continue;  end
        q = quantile(data, [0.25, 0.5, 0.75]);
        plot(i, q(1), [col, '^'], 'MarkerSize', 6,  'LineWidth', 1.2)
        plot(i, q(2), [col, 's'], 'MarkerSize', 10, 'LineWidth', 1.5)
        plot(i, q(3), [col, 'v'], 'MarkerSize', 6,  'LineWidth', 1.2)
    end
end
