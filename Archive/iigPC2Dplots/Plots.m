% =========================================================================
% Plots_V3.m  —  Analytical igPC parameter maps  (normalized, no overlays)
%
% Description:
%   Computes the analytical igPC parameter maps — ν/c and κ/c — over a
%   (mu, sigma) grid and plots them without scatter overlays (c = 1,
%   i.e. the maps show the ratio to the diffusion coefficient).
%
%   This is a compact version of Plots_V2.m focused on the two
%   "no-correction" normalized maps used in preliminary analysis.
%
%   Two figures are produced:
%       Fig. 5  — ν/c map (pcolor, turbo colormap)  → map_nu_NoCorrection.pdf
%       Fig. 6  — κ/c map (pcolor, turbo colormap)  → map_kappa_NoCorrection.pdf
%
%   For maps with simulation/experimental scatter overlays, use Plots_V2.m.
%
% Input files:
%   None — only analytical igPC formulas are used.
%
% Author : C. F. Silva  (camilo.silva@tum.de)
%          Professorship of Thermofluiddynamics, TU Munich
% Paper  : Silva et al. (2026). Output-only identification of
%          thermoacoustic systems using polynomial chaos expansions.
%          Combust. Flame.
% =========================================================================

clear all

% =========================================================================
% USER OPTIONS
% =========================================================================

% Grid extents for the analytical maps
mu_range    = [0, 10];     % Mean amplitude axis
sigma_range = [0.1, 2];    % Std deviation axis (wider than Plots_V2)
n_mu        = 100;
n_sigma     = 80;

% Colour-axis limits
vis_nu_nc  = [0, 10];   % ν/c colour range
vis_kap_nc = [0, 10];   % κ/c colour range

save_figs = true;        % true  → save PDF figures

% =========================================================================
% ANALYTICAL igPC MAPS
%   ν/c = A1/D,   b/c = B1/D,   κ/c = 8·B1/D
%   where A1 = ½(9σ⁴ + 2μ²σ² − μ⁴),  B1 = ½(3σ² − μ²),
%         D  = μ²σ²(σ² − μ²)
% =========================================================================

mu_vec    = linspace(mu_range(1),    mu_range(2),    n_mu);
sigma_vec = linspace(sigma_range(1), sigma_range(2), n_sigma);

[MU_grid, SIGMA_grid] = meshgrid(mu_vec, sigma_vec);

nu_c_map    = zeros(size(MU_grid));
kappa_c_map = zeros(size(MU_grid));

for kxo = 1 : n_mu
    mu = mu_vec(kxo);
    for kx1 = 1 : n_sigma
        sigma = sigma_vec(kx1);
        A1 = 0.5 * (9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
        B1 = 0.5 * (3*sigma^2 - mu^2);
        D  = mu^2 * sigma^2 * (sigma^2 - mu^2);
        nu_c_map(kx1, kxo)    = A1 / D;
        kappa_c_map(kx1, kxo) = 8 * B1 / D;
    end
end

% =========================================================================
% FIGURE 5 — ν/c map (normalized)
% =========================================================================

figure(5); clf
pcolor(MU_grid, SIGMA_grid, nu_c_map)
axis([mu_range, 0, 1])
caxis(vis_nu_nc);  shading interp;  colormap(turbo)
xlabel('\mu');  ylabel('\sigma');  title('\nu / c  (analytical)')
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, './map_nu_NoCorrection.pdf');  end

% =========================================================================
% FIGURE 6 — κ/c map (normalized)
% =========================================================================

figure(6); clf
pcolor(MU_grid, SIGMA_grid, kappa_c_map)
axis([mu_range, 0, 1])
caxis(vis_kap_nc);  shading interp;  colormap(turbo)
xlabel('\mu');  ylabel('\sigma');  title('\kappa / c  (analytical)')
set(gca, 'FontSize', 16)
set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], 'PaperPosition', [0 0 10 10])
if save_figs;  saveas(gcf, './map_kappa_NoCorrection.pdf');  end
