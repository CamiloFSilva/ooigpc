
clear all

% =========================================================================
%  SOURCE — choose where the amplitude signal comes from
%    'FILE' : load pre-computed data from ../DATA/  (original behaviour)
%    'VDP'  : generate on-the-fly with simulate_van_der_pol (symplectic Euler)
%             then band-pass filter + Hilbert envelope
% =========================================================================
SOURCE = 'VDP';
%SOURCE = 'FILE';

% ---- VdP-only settings (ignored when SOURCE = 'FILE') ------------------
SEED   = 0;    % random seed — change for a different realisation
BW_HZ  = 100;   % half-bandwidth of the band-pass filter around f0  (Hz)
% ------------------------------------------------------------------------

%CASE = 'NoCorrection';
CASE = 'CorrectionTwoLevels';

%BIN = 'YES';
BIN = 'NOP';

FIG = 'YES';
%FIG = 'NOP';


% Parameters of the simulation for the Langevin Equation
nu_true    = 16;      % linear growth rate
kappa_true = 3.4;     % 3rd-order nonlinearity coefficient
Gamma_true = 1e7;     % noise intensity
omega_true = 2*pi*120;

a_true = nu_true;
b_true = kappa_true/8;
c_true = Gamma_true/(4*omega_true^2);

tau_val = [0.00025:0.00025:0.2];

%% Carrying out the numerical simulations

NumSim = 1;
nu_vec = [16];

for nk = 1:length(nu_vec)

    nu = nu_vec(nk)

    for ns = 1:NumSim

        disp(['Simulation ',num2str(ns)])

        % =================================================================
        %  Load or generate the amplitude signal
        % =================================================================

        if strcmp(SOURCE, 'FILE')
            % ------ original behaviour: load pre-computed .mat file ------
            file = ['./Data_nu',num2str(nu),'.mat'];
            load(file, 'A', 'dt', 'Gamma', 'wn')

            x        = A{ns};
            dt_sim   = dt;
            N        = length(x);
            fn       = wn/(2*pi);
            omega    = wn;

            disp('Signal loaded from file.');
            disp(['  dt = ', num2str(dt_sim), ' s,   N = ', num2str(N)]);

        elseif strcmp(SOURCE, 'VDP')
            % ------ generate with Van der Pol solver + Hilbert -----------
            dt_sim = 5e-5;          % time step (s)
            N_steps = 2e6;          % number of integration steps
            fs      = 1/dt_sim;     % sampling frequency (Hz)
            omega   = omega_true;
            fn      = omega/(2*pi);
            Gamma   = Gamma_true;

            disp(sprintf('Simulating VdP (symplectic Euler, seed=%d) ...', SEED));
            [~, eta, ~, ~] = simulate_van_der_pol( ...
                nu, kappa_true, Gamma_true, omega, N_steps, dt_sim, ...
                'Method', 'symplectic', 'Seed', SEED + ns - 1);

            % Band-pass filter around f0 ± BW_HZ
            f_low  = fn - BW_HZ;
            f_high = fn + BW_HZ;
            nyq    = fs/2;
            [b_bp, a_bp] = butter(4, [f_low/nyq, f_high/nyq], 'bandpass');
            eta_bp = filtfilt(b_bp, a_bp, double(eta));

            % Hilbert amplitude envelope
            x      = abs(hilbert(eta_bp));
            N      = length(x);
            dt_sim = dt_sim;

            disp(sprintf('  Done.  BW = ±%g Hz   mean(A_H) = %.4f   A_eq = %.4f', ...
                BW_HZ, mean(x(5001:end)), sqrt(8*nu/kappa_true)));
        else
            error('Unknown SOURCE: choose ''FILE'' or ''VDP''.');
        end

        % =================================================================
        %  MSD computation  (identical for both sources)
        % =================================================================

        tk_vec = floor(tau_val / dt_sim);

        for kk = 1:length(tk_vec)

            tk = tk_vec(kk);

            xvec1 = x(5000:1:N-tk)';
            xvec2 = x(5000+tk:1:N)';

            mu_global  = mean(xvec2);
            var_global = var(xvec2);
            std_global = std(xvec2);

            %%%%

            if strcmp(BIN, 'YES')

                nbin = 30;
                Amin = min(x);
                Amax = max(x);
                dA   = (Amax-Amin)/nbin;
                Abin = linspace(Amin+dA, Amax, nbin);

                edges = [Amin Abin];
                idx   = discretize(x, edges);
                Afloor = edges(idx);

                clear xvec1 xvec2
                xvec1 = Afloor(1:1:N-tk);
                xvec2 = Afloor(1+tk:1:N);

            end

            %%%%

            deltax  = (xvec1 - xvec2);
            MSD(kk) = sum(deltax.^2)/length(deltax);
            dMSD(kk)= sum(deltax.^2)/(length(deltax)*(tk*dt_sim));

        end

        % =================================================================
        %  Fit exponential model  MSD(τ) = A_fit·(1 − exp(−τ/T))
        % =================================================================

        taus = tk_vec * dt_sim;

        ft = fittype('A*(1 - exp(-taus/T))', ...
            'independent', 'taus', ...
            'coefficients', {'A','T'});

        f = fit(taus(:), MSD(:), ft, 'StartPoint', [0.1 0.01]);

        A_fit      = f.A;
        tau_drift  = f.T;
        MSDfit     = A_fit*(1 - exp(-taus/tau_drift));
        dMSDfit    = MSDfit ./ taus;
        dMSD_plot  = MSD   ./ taus;

        c_recov = (A_fit/2) * (1/tau_drift);

        disp(sprintf('  A_fit     = %.6f', A_fit));
        disp(sprintf('  tau_drift = %.6f s', tau_drift));
        disp(sprintf('  c_recov   = %.6f   (c_true = %.6f,  ratio = %.3f)', ...
            c_recov, c_true, c_recov/c_true));

        % =================================================================
        %  Figures
        % =================================================================

        if strcmp(FIG, 'YES')

            label_src = [SOURCE, sprintf('  seed=%d  BW=±%gHz', SEED, BW_HZ)];
            if strcmp(SOURCE, 'FILE'), label_src = SOURCE; end

            figure(20); clf
            plot(taus, MSD,    'rx', 'MarkerSize',4, 'LineWidth',1); hold on
            plot(taus, MSDfit, 'b-', 'MarkerSize',8, 'LineWidth',2)
            grid on; ax = gca; ax.YGrid='on'; ax.YMinorGrid='on'; ax.XGrid='off';
            axis([0 0.2 0 max(MSD)*1.15])
            xlabel('\tau  (s)'); ylabel('MSD  (Pa^2)');
            title(sprintf('MSD — %s — \\nu=%d', label_src, nu));
            ax.XAxis.Exponent = 0;
            set(gca,'FontSize',16)

            figure(21); clf
            plot(taus, dMSD_plot/2,  'rx', 'MarkerSize',4, 'LineWidth',1); hold on
            plot(taus, dMSDfit/2,    'b-', 'MarkerSize',8, 'LineWidth',2)
            yline(c_recov, 'k--', 'LineWidth',1.5, ...
                  'Label', sprintf('c_{recov}=%.4f', c_recov), ...
                  'LabelHorizontalAlignment','left')
            yline(c_true,  'r:',  'LineWidth',1.5, ...
                  'Label', sprintf('c_{true}=%.4f', c_true), ...
                  'LabelHorizontalAlignment','left')
            grid on; ax = gca; ax.YGrid='on'; ax.YMinorGrid='on'; ax.XGrid='off';
            axis([0 0.2 0 max(dMSD_plot/2)*1.15])
            xlabel('\tau  (s)'); ylabel('MSD(\tau)/(2\tau)');
            title(sprintf('Extrapolation \\tau\\rightarrow0  —  %s  —  \\nu=%d', ...
                label_src, nu));
            ax.XAxis.Exponent = 0;
            set(gca,'FontSize',16)

        end

        % =================================================================
        %  Two-level correction + identification  (Eqs. 10–12)
        % =================================================================

        switch CASE

            case 'NoCorrection'

                sigma      = std_global;
                std_correc = 0;

            case 'CorrectionTwoLevels'

                std_crit = 0.28 * mu_global;

                if std_global > std_crit

                    c1_std = 0.144*std_global + 0.0126;
                    c2_std = -0.035*std_global;

                    std1 = 0.423*mu_global;
                    std2 = 0.28*mu_global;

                    BETA       = (std_global - std2)/(std1 - std2);
                    std_correc = (1 - BETA)*c2_std + BETA*c1_std;

                    sigma = std_global + std_correc;
                    disp('Region 1  (marginally stable)')

                else

                    Skewness   = skewness(xvec2);
                    gamma_sk   = Skewness * std_global^3;

                    x2         = roots([4 0 -6*std_global^2 gamma_sk]);
                    std_correc = 2.6 * real(x2(3));

                    var_corrected = std_global^2 - 2*std_correc^2;
                    if var_corrected <= 0
                        % Fallback: cubic broke down (large |skewness|);
                        % "no correction" preserves information in std_raw.
                        disp('  WARNING: var_corrected <= 0 — reverting to sigma = std_raw');
                        std_correc    = 0;
                        var_corrected = std_global^2;
                    end

                    sigma = sqrt(var_corrected);
                    disp('Region 2  (linearly unstable)')

                end

        end

        mu = mu_global;

        A1 = 1/2*(9*sigma^4 + 2*mu^2*sigma^2 - mu^4);
        A2 = mu^2*sigma^2*(sigma^2 - mu^2);

        B1 = (3*sigma^2 - mu^2)/2;
        B2 = mu^2*sigma^2*(sigma^2 - mu^2);

        if abs(A2) < 1e-30
            error('Singular denominator (mu=%.4g, sigma=%.4g). Cannot identify.', mu, sigma);
        end

        a_recov = A1/A2 * c_recov;
        b_recov = B1/B2 * c_recov;

        a_recov_vec(ns,1) = a_recov;
        b_recov_vec(ns,1) = b_recov;
        c_recov_vec(ns,1) = c_recov;

        xstats(ns,1) = mu_global;
        xstats(ns,2) = std_global;

        nu_SI    = a_recov;
        kappa_SI = 8*b_recov;
        Gamma_SI = c_recov * 4*omega^2;

        fprintf('\n--- Identified parameters (ns=%d) ---\n', ns);
        fprintf('  nu    = %8.3f s^-1     (true: %g,   error: %+.1f %%)\n', ...
            nu_SI,    nu,         100*(nu_SI    - nu)         /abs(nu));
        fprintf('  kappa = %8.3f           (true: %g,  error: %+.1f %%)\n', ...
            kappa_SI, kappa_true, 100*(kappa_SI - kappa_true) /abs(kappa_true));
        fprintf('  Gamma = %8.3e           (true: %g,  error: %+.1f %%)\n', ...
            Gamma_SI, Gamma_true, 100*(Gamma_SI - Gamma_true) /abs(Gamma_true));
        fprintf('  c     = %8.6f           (true: %.6f, error: %+.1f %%)\n', ...
            c_recov,  c_true,     100*(c_recov  - c_true)     /abs(c_true));
        fprintf('  mu        = %.4f\n', mu);
        fprintf('  std_raw   = %.4f\n', std_global);
        fprintf('  sigma     = %.4f  (corrected)\n', sigma);

    end

    ideal_param = [a_true, b_true, c_true];
    recov_param = [a_recov_vec, b_recov_vec, c_recov_vec];

    file = ['../DATA/nu_',num2str(nu),'_',CASE,'_',BIN,'_',SOURCE,'_fromigPC.mat'];
    %save(file, 'ideal_param', 'recov_param', 'xstats');

end
