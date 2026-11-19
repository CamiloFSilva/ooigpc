% readExpData.m
% =========================================================================
% Load, filter, and extract amplitude envelopes from experimental data
%
% DESCRIPTION
%   Reads the raw pressure time series from the Noiray & Denisov (2017)
%   experiments, identifies the dominant frequency of each operating
%   condition via Welch's PSD, band-pass-filters the signal around that
%   frequency, and extracts the instantaneous amplitude envelope using the
%   Hilbert transform.
%
%   The five operating conditions stored in the input .mat file correspond
%   to increasing equivalence ratios / heat-release rates; the loop index
%   ns runs from 1 to 5.
%
%   Output: envelopeRef_B<bandwidth>Exp.mat in the DATA/ subfolder, with
%   one cell entry per operating condition.
%
% INPUT
%   ./DATA/data_Noiray_Denisov_PROCI2017.mat
%       Variables:  X     — [Nt × 5] raw pressure signals  [Pa]
%                   time  — [Nt × 1] time vector            [s]
%
% OUTPUT
%   ./DATA/envelopeRef_B<bandwidth>Exp.mat
%       Variables:  A        — {1×5} cell of amplitude envelopes [Pa]
%                   time_vec — {1×5} cell of time vectors        [s]
%                   dt_vec   — [1×5] sampling time steps         [s]
%                   fn_vec   — [1×5] dominant frequencies        [Hz]
%                   wn_vec   — [1×5] dominant angular freqs      [rad/s]
%
% REFERENCES
%   Noiray, N. & Denisov, A. (2017).  A method to identify thermoacoustic
%   growth rates in combustion chambers from dynamic pressure time series.
%   Proc. Combust. Inst. 36, 3843-3850.
%
%   BP_filter_rect.m  — band-pass filter utility used here
%
% SEE ALSO
%   Experiments/igPC/OOID_igPC.m          — igPC identification
%   Experiments/KMCoeff/OOID_KM_PDFMatrix.m  — KM identification
% =========================================================================

clear all

% =========================================================================
%% USER OPTIONS
% =========================================================================

bandwidth = 40;    % [Hz]  band-pass filter half-width around fn
%bandwidth = 10;   % alternative narrower bandwidth

NumCond   = 5;     % number of operating conditions in the data file

save_data = true;  % true  → save envelopes to DATA/
save_figs = true;  % true  → save PSD figure(s) to current folder

% =========================================================================
%% LOAD RAW DATA
% =========================================================================

load('./DATA/data_Noiray_Denisov_PROCI2017.mat', 'X', 'time')

% =========================================================================
%% ENVELOPE EXTRACTION LOOP
% =========================================================================

A        = cell(1, NumCond);
time_vec = cell(1, NumCond);
dt_vec   = zeros(1, NumCond);
fn_vec   = zeros(1, NumCond);
wn_vec   = zeros(1, NumCond);

for ns = 1 : NumCond

    eta = X(:, ns);              % [Pa]  raw pressure signal
    nt  = length(eta);
    dt  = time(2) - time(1);     % [s]   sampling interval
    Fs  = 1 / dt;                % [Hz]  sampling frequency

    % ------------------------------------------------------------------
    % 1. Dominant frequency via Welch PSD
    % ------------------------------------------------------------------

    [Pxx, f] = pwelch(eta, floor(nt/4), [], [], Fs);

    figure(ns); clf
    semilogy(f(2:end), Pxx(2:end), 'b')
    xlim([1 500])
    xlabel('f  [Hz]');  ylabel('PSD  [Pa^2 / Hz]')
    title(['Operating condition ', num2str(ns)])
    set(gca, 'FontSize', 14)
    if save_figs
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [15 10], ...
            'PaperPosition', [0 0 15 10])
        saveas(gcf, ['./Plotting/PSD_ns', num2str(ns), '_B', num2str(bandwidth), '.pdf'])
    end

    [~, idx] = max(Pxx);
    fn = round(f(idx));          % [Hz]      dominant frequency
    wn = fn * 2 * pi;            % [rad/s]   angular frequency

    fprintf('ns = %d:  fn = %d Hz,  dt = %.2e s\n', ns, fn, dt)

    % ------------------------------------------------------------------
    % 2. Band-pass filter and Hilbert envelope
    % ------------------------------------------------------------------

    eta_f = BP_filter_rect(eta, fn - bandwidth/2, fn + bandwidth/2, Fs);

    A{ns}        = abs(hilbert(eta_f));   % instantaneous amplitude  [Pa]
    time_vec{ns} = time;
    dt_vec(ns)   = dt;
    fn_vec(ns)   = fn;
    wn_vec(ns)   = wn;

end   % ns loop

% =========================================================================
%% SAVE OUTPUT
% =========================================================================

if save_data
    filename = ['./DATA/envelope_BW', num2str(bandwidth), 'Exp.mat'];
    save(filename, 'A', 'time_vec', 'dt_vec', 'fn_vec', 'wn_vec')
    fprintf('Envelopes saved to: %s\n', filename)
end
