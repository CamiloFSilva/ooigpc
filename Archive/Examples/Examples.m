% Examples.m
%
% Tutorial script for PC (polynomial chaos) algebraic operations.
%
% Companion to Section 1 of the Supplementary Material of:
%   Silva, C. F. (2026).
%   "Output-only identification of thermoacoustic systems using polynomial
%    chaos expansions." Combustion and Flame.
%
% Usage: set the variable 'Example' below to one of:
%   'base'           – single 1-D PC random variable and its basis functions
%   'sum'            – addition of two independent PC random variables
%   'multiplication' – multiplication of two independent PC random variables
%   'division'       – inversion via 1-D Galerkin projection (x = 1/a)
%   'axb'            – general linear equation ax = b via 2-D Galerkin projection
%
% Figures are saved to ./Figs/.
%
% See also: Tensor_1D_comput.m (precomputes C and D tensors),
%           DisplayCtensor.m, DisplayDtensor.m (inspect tensor entries)

clear all

%% -----------------------------------------------------------------------
%  Hermite polynomial basis (probabilist's convention, order 0-2)
%
%  ψ₀(ξ) = 1
%  ψ₁(ξ) = ξ
%  ψ₂(ξ) = ξ² − 1
%
%  where ξ ~ N(0,1).  These are orthonormal w.r.t. the standard Gaussian
%  weight, with norms  ‹ψ₀²›=1, ‹ψ₁²›=1, ‹ψ₂²›=2.
%
%  A scalar random variable x is represented as the truncated expansion:
%    x = x₀ψ₀ + x₁ψ₁ + x₂ψ₂
%  Its variance is Var[x] = x₁²·1 + x₂²·2  (mean term x₀ does not contribute).
% ------------------------------------------------------------------------
p0 = 1;
p1 = @(x)  x;
p2 = @(x)  x.^2-1;

%%
Example = 'multiplication'; % set to: 'base', 'sum', 'multiplication', 'division', 'axb'

switch Example

    % =====================================================================
    case 'base'
    % =====================================================================
    % Single 1-D PC random variable.
    % (Supplementary Material, Section 1.1)
    %
    % Demonstrates:
    %   • Construction of a PC expansion x = x₀ψ₀ + x₁ψ₁ + x₂ψ₂
    %   • Monte Carlo evaluation of its marginal PDF
    %   • Verification of the analytical variance formula  Var[x] = x₁² + 2x₂²
    %   • Visualization of the PC basis functions ψ₀, ψ₁, ψ₂
    % =====================================================================

        q = 2;           % polynomial order (second-order gPC)
        x = zeros(q+1,1);

        % PC coefficients:  x₀ = mean, x₁ = linear coeff., x₂ = quadratic coeff.
        x(1) = 5;    % x₀ (zeroth mode — contributes to the mean)
        x(2) = 0.6;  % x₁ (first  mode — drives variance, skewness)
        x(3) = 0.06; % x₂ (second mode — drives variance, kurtosis)

        % Draw N samples of the standard Gaussian seed ξ
        N = 100000;
        etaRand1 = randn(N, 1);

        % Evaluate basis functions at the drawn samples
        psi0 = ones(N, 1);       % ψ₀ = 1  (constant)
        psi1 = p1(etaRand1);     % ψ₁ = ξ  (Gaussian)
        psi2 = p2(etaRand1);     % ψ₂ = ξ²−1  (non-Gaussian: shifted chi-squared)

        PSI = [psi0, psi1, psi2];  % [N×3] basis matrix

        % Reconstruct each modal contribution and the total PC variable
        pdfo = PSI(:,1)*x(1);   % zeroth mode contribution (constant → Dirac delta)
        pdf1 = PSI(:,2)*x(2);   % first  mode contribution (Gaussian)
        pdf2 = PSI(:,3)*x(3);   % second mode contribution (non-Gaussian)
        pdf  = pdfo + pdf1 + pdf2;  % total PC variable

        % ------------------------------------------------------------------
        % Variance check:  Var[x] = x₁²‹ψ₁²› + x₂²‹ψ₂²› = x₁² + 2x₂²
        % ------------------------------------------------------------------
        var_ref = var(pdf);                   % Monte Carlo reference
        var_gPC = x(2)^2 + 2*x(3)^2;         % analytical PC formula
        disp(['Reference variance: ' num2str(var_ref)]);
        disp(['gPC variance:       ' num2str(var_gPC)]);

        %% Plots
        fsize = 24;

        % — zeroth mode distribution (all mass at x₀, appears as a narrow histogram) —
        figure(1)
        histogram(pdfo, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_pdfo_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — first mode: x₁ψ₁ ~ N(0, x₁²) —
        figure(2)
        histogram(pdf1, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_pdf1_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — second mode: x₂ψ₂ = x₂(ξ²−1), shifted chi-squared —
        figure(3)
        histogram(pdf2, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_pdf2_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — total PC variable x = x₀ + x₁ψ₁ + x₂ψ₂ (non-Gaussian mixture) —
        figure(4)
        histogram(pdf, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_pdf_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — basis function ψ₀ = 1 (Dirac delta at 1) —
        figure(5)
        histogram(psi0, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/psi0_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — basis function ψ₁ = ξ ~ N(0,1) —
        figure(6)
        histogram(psi1, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/psi1_', Example, '.pdf'];
        saveas(gcf, figfile)

        % — basis function ψ₂ = ξ²−1  (shifted chi-squared with 1 d.o.f.) —
        figure(7)
        histogram(psi2, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/psi2_', Example, '.pdf'];
        saveas(gcf, figfile)

    % =====================================================================
    case 'sum'
    % =====================================================================
    % Addition of two independent 1-D PC random variables.
    % (Supplementary Material, Section 1.2)
    %
    % Problem:  z = x + y,  with
    %   x = x₀ψ₀(ξ₁) + x₁ψ₁(ξ₁) + x₂ψ₂(ξ₁)   [driven by ξ₁]
    %   y = y₀ψ₀(ξ₂) + y₁ψ₁(ξ₂) + y₂ψ₂(ξ₂)   [driven by ξ₂]
    %
    % Because ξ₁ and ξ₂ are independent, z lives in a 2-D PC space with
    % 9 basis elements  Ψ_{mn}(ξ₁,ξ₂) = ψₘ(ξ₁)·ψₙ(ξ₂),  m,n ∈ {0,1,2}.
    %
    % For addition, the PC coefficients of z are obtained by simple linear
    % combination — no tensor products are needed.
    % =====================================================================

        % PC coefficients of x and y
        xo = 5;  x1 = 0.6;  x2 = 0.06;
        yo = 2;  y1 = 0.4;  y2 = -0.04;

        % -- Analytical PC representation of z = x + y ------------------
        % 2-D basis ordering (row-major in [ξ₁, ξ₂]):
        %   z(1)=Ψ₀₀, z(2)=Ψ₁₀, z(3)=Ψ₂₀,
        %   z(4)=Ψ₀₁, z(5)=Ψ₁₁, z(6)=Ψ₂₁,
        %   z(7)=Ψ₀₂, z(8)=Ψ₁₂, z(9)=Ψ₂₂
        zo = xo + yo;   % mean term
        z1 = x1;        % x's first mode in ξ₁ direction
        z2 = x2;        % x's second mode in ξ₁ direction
        z3 = y1;        % y's first mode in ξ₂ direction
        z6 = y2;        % y's second mode in ξ₂ direction
        % (mixed cross-terms z4=z5=z7=z8=z9=0 since addition does not couple ξ₁ and ξ₂)

        z = zeros(9,1);
        z(1) = zo;  z(2) = z1;  z(3) = z2;  z(4) = z3;  z(7) = z6;

        % -- Monte Carlo reference ----------------------------------------
        N = 100000;
        etaRand1 = randn(N, 1);
        etaRand2 = randn(N, 1);

        % 1-D basis in ξ₁
        psi0A = ones(N, 1);  psi1A = p1(etaRand1);  psi2A = p2(etaRand1);
        % 1-D basis in ξ₂
        psi0B = ones(N, 1);  psi1B = p1(etaRand2);  psi2B = p2(etaRand2);

        % 2-D tensor-product basis
        PSI = [psi0A.*psi0B, psi1A.*psi0B, psi2A.*psi0B, ...
               psi0A.*psi1B, psi1A.*psi1B, psi2A.*psi1B, ...
               psi0A.*psi2B, psi1A.*psi2B, psi2A.*psi2B];

        % Reconstruct z from PC coefficients
        pdfz = PSI * z;

        % Reconstruct x and y individually for reference plots
        x = xo*psi0A + x1*psi1A + x2*psi2A;
        y = yo*psi0B + y1*psi1B + y2*psi2B;
        z_ref = x + y;    % direct Monte Carlo result for validation

        % -- Variance check -----------------------------------------------
        % 2-D basis norms: ‹Ψ_{mn}²› = ‹ψₘ²›·‹ψₙ²›
        % With ‹ψ₀²›=1, ‹ψ₁²›=1, ‹ψ₂²›=2:
        Norm = [0, 1, 2, 1, 1, 2, 2, 2, 4];  % (index 1 = mean, excluded from variance)

        var_ref  = var(z_ref);
        var_gPC  = sum(Norm(2:end) .* (z(2:end).^2)');
        disp(['Reference variance: ' num2str(var_ref)]);
        disp(['gPC variance:       ' num2str(var_gPC)]);

        %% Plots
        fsize = 20;

        % — z = x+y: PC (yellow) vs Monte Carlo (grey) —
        figure(5)
        histogram(pdfz,  35, 'FaceColor', [1.0 1.0 0.0], 'Normalization', 'pdf')
        hold on
        histogram(z_ref, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_z_', Example, '.pdf'];
        saveas(gcf, figfile)

        figure(6)
        histogram(x, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_x_', Example, '.pdf'];
        saveas(gcf, figfile)

        figure(7)
        histogram(y, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', fsize)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_y_', Example, '.pdf'];
        saveas(gcf, figfile)

    % =====================================================================
    case 'multiplication'
    % =====================================================================
    % Multiplication of two independent 1-D PC random variables.
    % (Supplementary Material, Section 1.3)
    %
    % Problem:  z = x · y,  with x and y as in the 'sum' case.
    %
    % For multiplication every product ψₘ(ξ₁)·ψₙ(ξ₂) remains a 2-D basis
    % element Ψ_{mn}, so ALL 9 PC coefficients of z are nonzero and are
    % obtained by simple algebra (no Galerkin linear system required).
    % =====================================================================

        % PC coefficients of x and y
        xo = 5;  x1 = 0.6;  x2 = 0.06;
        yo = 2;  y1 = 0.4;  y2 = -0.04;

        % -- Analytical PC coefficients of z = x·y -----------------------
        % (x₀ψ₀A + x₁ψ₁A + x₂ψ₂A)·(y₀ψ₀B + y₁ψ₁B + y₂ψ₂B)
        % gives 9 basis terms Ψ_{mn} = ψₘ(ξ₁)·ψₙ(ξ₂) with coefficient xₘ·yₙ:
        zo = xo*yo;   % Ψ₀₀ coefficient
        z1 = x1*yo;   % Ψ₁₀ coefficient
        z2 = x2*yo;   % Ψ₂₀ coefficient
        z3 = xo*y1;   % Ψ₀₁ coefficient
        z4 = x1*y1;   % Ψ₁₁ coefficient
        z5 = x2*y1;   % Ψ₂₁ coefficient
        z6 = xo*y2;   % Ψ₀₂ coefficient
        z7 = x1*y2;   % Ψ₁₂ coefficient
        z8 = x2*y2;   % Ψ₂₂ coefficient

        z = [zo; z1; z2; z3; z4; z5; z6; z7; z8];

        % -- Monte Carlo reference ----------------------------------------
        N = 100000;
        etaRand1 = randn(N, 1);
        etaRand2 = randn(N, 1);

        psi0A = ones(N,1);  psi1A = p1(etaRand1);  psi2A = p2(etaRand1);
        psi0B = ones(N,1);  psi1B = p1(etaRand2);  psi2B = p2(etaRand2);

        PSI = [psi0A.*psi0B, psi1A.*psi0B, psi2A.*psi0B, ...
               psi0A.*psi1B, psi1A.*psi1B, psi2A.*psi1B, ...
               psi0A.*psi2B, psi1A.*psi2B, psi2A.*psi2B];

        pdfz  = PSI * z;

        x     = xo*psi0A + x1*psi1A + x2*psi2A;
        y     = yo*psi0B + y1*psi1B + y2*psi2B;
        z_ref = x .* y;    % direct Monte Carlo multiplication

        % -- Variance check -----------------------------------------------
        Norm = [0, 1, 2, 1, 1, 2, 2, 2, 4];
        var_ref  = var(z_ref);
        var_gPC  = sum(Norm(2:end) .* (z(2:end)'.^2));
        disp(['Reference variance: ' num2str(var_ref)]);
        disp(['gPC variance:       ' num2str(var_gPC)]);

        %% Plots
        % — z = x·y: PC (yellow) vs Monte Carlo (grey) —
        figure(5)
        histogram(pdfz,  35, 'FaceColor', [1.0 1.0 0.0], 'Normalization', 'pdf')
        hold on
        histogram(z_ref, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_z_', Example, '.pdf'];
        saveas(gcf, figfile)

        figure(6)
        histogram(x, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_x_', Example, '.pdf'];
        saveas(gcf, figfile)

        figure(7)
        histogram(y, 35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_y_', Example, '.pdf'];
        saveas(gcf, figfile)

    % =====================================================================
    case 'division'
    % =====================================================================
    % Inversion of a 1-D PC random variable via Galerkin projection.
    % (Supplementary Material, Section 1.4)
    %
    % Problem: given a = a₀ψ₀ + a₁ψ₁, find x such that  a·x = 1.
    %
    % Galerkin projection: multiply both sides by ψᵢ and take expectation.
    % Using the triple-product C tensor Cᵢⱼₖ = ‹ψᵢψⱼψₖ›, the equation
    %   Σⱼ (Σₖ aₖ Cᵢⱼₖ) xⱼ = ‹ψᵢ›
    % becomes a 3×3 linear system for x = [x₀, x₁, x₂]ᵀ.
    %
    % For this specific 1-D case the matrix is worked out analytically:
    %   row 0:  a₀x₀ + a₁x₁             = 1
    %   row 1:  a₁x₀ + a₀x₁ + 2a₁x₂    = 0
    %   row 2:               a₁x₁ + a₀x₂ = 0
    % =====================================================================

        % PC coefficients of a
        ao = 0.5;  a1 = 0.1;  a2 = 0;   % a₀ = mean, a₁ = std coeff

        % -- Galerkin linear system: Mat·x = bvec -------------------------
        % (derived from the C-tensor Galerkin projection of a·x = 1)
        Mat  = [ao, a1,    0  ; ...
                a1, ao,  2*a1 ; ...
                 0, a1,    ao ];

        bvec = [1; 0; 0];   % right-hand side: ‹ψᵢ·1› = δᵢ₀

        x = Mat \ bvec;     % PC coefficients of x = 1/a

        % -- Monte Carlo reference ----------------------------------------
        N = 100000;
        etaRand1 = randn(N, 1);

        psi0A = ones(N,1);  psi1A = p1(etaRand1);  psi2A = p2(etaRand1);

        PSI  = [psi0A, psi1A, psi2A];
        pdfx = PSI * x;    % PC approximation of 1/a

        a   = ao*psi0A + a1*psi1A;
        xmt = 1 ./ a;      % Monte Carlo reference (exact inversion)

        %% Plots
        % — PC (yellow) vs Monte Carlo (grey) for x = 1/a —
        figure(5)
        histogram(pdfx, 35, 'FaceColor', [1.0 1.0 0.0], 'Normalization', 'pdf')
        hold on
        histogram(xmt,  35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_a_', Example, '.pdf'];
        saveas(gcf, figfile)

    % =====================================================================
    case 'axb'
    % =====================================================================
    % General linear PC equation ax = b with two independent random inputs.
    % (Supplementary Material, Section 1.5)
    %
    % Problem: given a (driven by ξ₁) and b (driven by ξ₂), find x such
    % that  a·x = b,  i.e.,  x = b/a.
    %
    % Here x lives in a 2-D PC space with basis
    %   {ψ₀(ξ₁)ψ₀(ξ₂), ψ₁(ξ₁)ψ₀(ξ₂), ψ₀(ξ₁)ψ₁(ξ₂), ψ₁(ξ₁)ψ₁(ξ₂)}.
    %
    % Galerkin projection of a·x = b onto this basis yields a 4×4 system.
    % The matrix has a block structure exploiting the independence of ξ₁,ξ₂.
    % =====================================================================

        % PC coefficients of a and b
        ao = 5;   a1 = 0.6;  a2 = 0;    % a = a₀ψ₀(ξ₁) + a₁ψ₁(ξ₁)
        bo = 2;   b1 = 0;    b2 = -0.4; % b = b₀ψ₀(ξ₂) + b₂ψ₁(ξ₂)
        % Note: b uses ξ₂ as its random dimension.

        % -- Galerkin linear system: Mat·x = bvec -------------------------
        % 2-D basis for x: Ψ₀₀, Ψ₁₀, Ψ₀₁, Ψ₁₁
        % (since a only depends on ξ₁, the ξ₂ dimension decouples into
        %  two independent 2×2 blocks with the same coefficient matrix)
        Mat  = [ao, a1,  0,  0; ...
                a1, ao,  0,  0; ...
                 0,  0, ao, a1; ...
                 0,  0, a1, ao];

        bvec = [bo; 0; b2; 0];  % RHS from Galerkin projection of b

        x = Mat \ bvec;         % PC coefficients of x = b/a

        % -- Monte Carlo reference ----------------------------------------
        N = 100000;
        etaRand1 = randn(N, 1);
        etaRand2 = randn(N, 1);

        psi0A = ones(N,1);  psi1A = p1(etaRand1);
        psi0B = ones(N,1);  psi1B = p1(etaRand2);

        PSI  = [psi0A.*psi0B, psi1A.*psi0B, psi0A.*psi1B, psi1A.*psi1B];
        pdfx = PSI * x;    % PC approximation of b/a

        a   = ao*psi0A + a1*psi1A;
        b   = bo*psi0B + b2*psi1B;
        xmt = b ./ a;      % Monte Carlo reference (direct division)

        %% Plots
        % — PC (yellow) vs Monte Carlo (grey) for x = b/a —
        figure(5)
        histogram(pdfx, 35, 'FaceColor', [1.0 1.0 0.0], 'Normalization', 'pdf')
        hold on
        histogram(xmt,  35, 'FaceColor', [0.3 0.3 0.3], 'Normalization', 'pdf')
        set(gca, 'FontSize', 16)
        set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [10 10], ...
                 'PaperPosition', [0 0 10 10])
        figfile = ['./Figs/Marg_xmt_', Example, '.pdf'];
        saveas(gcf, figfile)

end
