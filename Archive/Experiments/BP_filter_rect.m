function [sig_filt,spec_sig_filt,freq_vec]=BP_filter_rectangular(sig,FHP,FLP,Fs) 
% return filtered signal (rectangular window specified by FHP and FLP)
% together with the fft of the filtered signal 
% The input parameters are the signal(column vector), the cut-on and off
% frequencies and the sampling rate
%
% Example
% length_test=51;xtest=rand(length_test,1);
% spec_xtest=fft(xtest);Fs_test=1000;
% fqvec_xtest=Fs_test*linspace(0,1,length_test);
% [eta_filt,spect_eta_filt,Fqvec]=BP_filter_rect(xtest,0.13*Fs_test,0.4*Fs_test,Fs_test);
% figure
% plot(fqvec_xtest,abs(spec_xtest));
% hold on
% plot(Fqvec,abs(spect_eta_filt),'--r');

nsamp=size(sig,1);
spec_sig_filt=fft(sig)/nsamp;
freq_vec = Fs*linspace(0,1,nsamp);
freq_vec0 = Fs/2*linspace(0,1,floor(nsamp/2));
if isempty(FHP)==0
    indexHP=find((freq_vec0(2:end)>=FHP)&(freq_vec0(1:end-1)<FHP));
end
indexLP=find((freq_vec0(2:end)>=FLP)&(freq_vec0(1:end-1)<FLP));

spec_sig_filt(indexLP+1:nsamp-indexLP+1,:)=...
    zeros(nsamp-2*indexLP+1,size(sig,2));

if isempty(FHP)==0
    spec_sig_filt(1:indexHP-1,:)=...
        zeros(indexHP-1,size(sig,2));
    spec_sig_filt(nsamp-indexHP+3:nsamp,:)=...
        zeros(indexHP-2,size(sig,2));
end
sig_filt=real(ifft(spec_sig_filt)*nsamp);