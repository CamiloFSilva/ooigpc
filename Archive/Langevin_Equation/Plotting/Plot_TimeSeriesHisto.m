% =========================================================================
% Plot_TimeSeriesHisto.m  —  Amplitude histograms and
%                             concatenated time series
%
% DESCRIPTION
%   Loads igPC identification results for each nu in nu_vec and produces:
%     1. Per-case marginal amplitude PDFs (histograms)   → Marg_PDF_<nu>.pdf
%     2. Concatenated amplitude time-series plot         → time_series0.pdf
%
%   These are diagnostic plots, not directly in the paper.
%
% INPUT FILES  (relative path from this script's location)
%   ../DATA/nu_<nu>_<CASE>_fromigPC.mat
%       Variables: x_stat (or legacy xvec1)
%
% PAPER
%   Silva, C. F. (2026).  "Output-only identification of thermoacoustic
%   systems using polynomial chaos expansions."  Combust. Flame.
% =========================================================================

clear all

% =========================================================================
%% USER OPTIONS
% =========================================================================

CASE = 'CorrectionTwoLevels';  % Must match the CASE used in OOID_igPC_V10.m
%CASE = 'NoCorrection';
%CASE = 'CorrectionOneLevel';

nu_vec = [0, 4, 8, 12, 16, 20, 24];  % Growth-rate cases to plot  [1/s]
dt_sim = 5e-5;                        % Simulation time step  [s]
N_plot = 1e6;                         % Max samples to use from each case

save_figs = true;        % true  → save PDF figures
                         % false → display only

% =========================================================================
%% AMPLITUDE HISTOGRAMS
% =========================================================================

xvec_vec = [];   % Accumulator for the concatenated time series

for nn = 1 : length(nu_vec)

    nu   = nu_vec(nn);
    file = ['../DATA/nu_', num2str(nu), '_', CASE, '_fromigPC.mat'];
    loaded = load(file);

    % Accept both current (x_stat) and legacy (xvec1) variable names
    if isfield(loaded, 'x_stat')
        xvec = loaded.x_stat;
    elseif isfield(loaded, 'xvec1')
        xvec = loaded.xvec1;
    else
        error(['No amplitude data found in ', file]);
    end
    xvec = xvec(1 : min(N_plot, length(xvec)));

    if save_figs
        figure(nn); clf
        histogram(xvec, 35, 'FaceColor', [0.3 0.3 0.3], ...
                  'Normalization', 'pdf')
        xlabel('A  [a.u.]');  ylabel('PDF')
        title(['\nu = ', num2str(nu)])
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
            'PaperPosition', [0 0 10 10])
        saveas(gcf, ['./Marg_PDF_', num2str(nu), '.pdf'])
    end

    xvec_vec = [xvec_vec, xvec'];

end

% =========================================================================
%% CONCATENATED TIME SERIES
% =========================================================================

N_total  = length(xvec_vec);
time_vec = linspace(0, dt_sim * N_total, N_total);

figure(10); clf
plot(time_vec, xvec_vec, '-')
axis([0 700 0 10])
xlabel('t  [s]');  ylabel('A  [a.u.]')
set(gca, 'FontSize', 20)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [30 10], ...
    'PaperPosition', [0 0 30 10])
if save_figs
    saveas(gcf, './time_series0.pdf')
end
