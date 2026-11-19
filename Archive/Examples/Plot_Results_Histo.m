
clear all

nu_vec=[0,4,8,12,16,20,24];

time_vec=[]; xvec_vec=[];


file =['../DATA/fromigPC.mat'];
load(file,'recov_param','xvec1','xstats');

%% plot time series
dt_sim = 5e-5;
dt=dt_sim;

xvec = xvec1(1:3e5);

N = length(xvec);

xvec_vec = [xvec_vec xvec'];


%% plot parameters

plot2 = figure(1);
histogram(xvec,35,'FaceColor',[0.3 0.3 0.3],'Normalization','pdf')
set(gca,'FontSize',16)
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [10 10]);
set(gcf,'PaperPosition', [0 0 10 10])
figfile = ['./Marg_PDF.pdf']
saveas(gcf,figfile)
%






